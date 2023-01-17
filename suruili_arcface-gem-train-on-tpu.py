import math

EPOCHS = 10
STEPS_PER_TPU_CALL = 1  # set this number higher for less TPU idle time
batch_size = 16
image_size = 500
learning_rate = 1e-5  # should be smaller than training on single GPU
feature_size = 2048  # Embedding size before the output layer

# ArcFace params
margin = 0.1  # DELG used 0.1, original ArcFace paper used 0.5. When margin is 0, it should be the same as doing a normal softmax but with embedding and weight normalised.
logit_scale = int(math.sqrt(feature_size))

# GeM params
gem_p = 3.
train_p = False  # whether to learn gem_p or not

# Google storage settings
# Set your own project id here
PROJECT_ID = 'YOUR_PROJECT_ID'
from google.cloud import storage
storage_client = storage.Client(project=PROJECT_ID)

# Step 1: Get the credential from the Cloud SDK
# Get the credential from the Cloud SDK
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
user_credential = user_secrets.get_gcloud_credential()

# Step 2: Set the credentials
# Set the credentials
user_secrets.set_tensorflow_credential(user_credential)

# Step 3: Take note of the GCS path
# Take note of the GCS path
from kaggle_datasets import KaggleDatasets
data_dir = KaggleDatasets().get_gcs_path()

import pandas as pd
import functools
from tqdm import tqdm

import tensorflow as tf
from pathlib import Path
import os

# detect and init the TPU
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

# instantiate a distribution strategy
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

# instantiating the model in the strategy scope creates the model on the TPU


def _parse_example(example, name_to_features, image_size, augmentation, unique_landmark_ids):
    parsed_example = tf.io.parse_single_example(example, name_to_features)
    # Parse to get image.
    image = parsed_example['image/encoded']
    image = tf.io.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = tf.math.divide(tf.subtract(image, 128.0), 128.0)
    # if augmentation:
    # image = _ImageNetCrop(image)
    # else:
    image = tf.image.resize(image, [image_size, image_size])
    image.set_shape([image_size, image_size, 3])
    # Parse to get label.
    label = parsed_example['image/class/label']
    label = tf.reduce_min(tf.where(tf.equal(unique_landmark_ids, label)))

    return image, label


def create_dataset(file_pattern, unique_landmark_ids, augmentation: bool = False):
    AUTO = tf.data.experimental.AUTOTUNE
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    filenames = tf.io.gfile.glob(file_pattern)

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO).shuffle(1000)

    # Create a description of the features.
    feature_description = {
        'image/height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/channels': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/format': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/id': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'image/class/label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    }

    customized_parse_func = functools.partial(
        _parse_example,
        name_to_features=feature_description,
        image_size=image_size,
        augmentation=augmentation,
        unique_landmark_ids=unique_landmark_ids,
    )
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(customized_parse_func)
    dataset = dataset.batch(batch_size)
    return dataset
import tensorflow as tf
import functools
import math


class GeMPoolingLayer(tf.keras.layers.Layer):
    def __init__(self, p=1., train_p=False):
        super().__init__()
        if train_p:
            self.p = tf.Variable(p, dtype=tf.float32)
        else:
            self.p = p
        self.eps = 1e-6

    def call(self, inputs: tf.Tensor, **kwargs):
        inputs = tf.clip_by_value(inputs, clip_value_min=1e-6, clip_value_max=tf.reduce_max(inputs))
        inputs = tf.pow(inputs, self.p)
        inputs = tf.reduce_mean(inputs, axis=[1, 2], keepdims=False)
        inputs = tf.pow(inputs, 1./self.p)
        return inputs


class DelfArcFaceModel(tf.keras.Model):
    def __init__(self, input_shape, n_classes, margin, logit_scale, feature_size, p=None, train_p=False):
        super().__init__()
        self.backbone = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
        self.backbone.summary()

        if p is not None:
            self.global_pooling = GeMPoolingLayer(p, train_p=train_p)
        else:
            self.global_pooling = functools.partial(tf.reduce_mean, axis=[1, 2], keepdims=False)
        self.dense1 = tf.keras.layers.Dense(feature_size, activation=None, kernel_initializer="glorot_normal")
        # self.bn1 = tf.keras.layers.BatchNormalization()
        self.arcface = ArcFaceLayer(n_classes, margin, logit_scale)

    def call(self, inputs, training=True, mask=None):
        images, labels = inputs
        x = self.extract_feature(images)
        x = self.arcface((x, labels))
        return x

    def extract_feature(self, inputs):
        x = self.backbone(inputs)
        x = self.global_pooling(x)
        x = self.dense1(x)
        return x


class ArcFaceLayer(tf.keras.layers.Layer):
    def __init__(self, num_classes, margin, logit_scale):
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.logit_scale = logit_scale

    def build(self, input_shape):
        self.w = self.add_weight("weights", shape=[int(input_shape[0][-1]), self.num_classes], initializer=tf.keras.initializers.get("glorot_normal"))
        self.cos_m = tf.identity(tf.cos(self.margin), name='cos_m')
        self.sin_m = tf.identity(tf.sin(self.margin), name='sin_m')
        self.th = tf.identity(tf.cos(math.pi - self.margin), name='th')
        self.mm = tf.multiply(self.sin_m, self.margin, name='mm')

    def call(self, inputs, training=True, mask=None):
        embeddings, labels = inputs
        normed_embeddings = tf.nn.l2_normalize(embeddings, axis=1, name='normed_embd')
        normed_w = tf.nn.l2_normalize(self.w, axis=0, name='normed_weights')

        cos_t = tf.matmul(normed_embeddings, normed_w, name='cos_t')
        sin_t = tf.sqrt(1. - cos_t ** 2, name='sin_t')

        cos_mt = tf.subtract(cos_t * self.cos_m, sin_t * self.sin_m, name='cos_mt')

        cos_mt = tf.where(cos_t > self.th, cos_mt, cos_t - self.mm)

        mask = tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes,
                          name='one_hot_mask')

        logits = tf.where(mask == 1., cos_mt, cos_t)
        logits = tf.multiply(logits, self.logit_scale, 'arcface_logist')
        return logits



def train_step(images, labels):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model((images, labels), training=True)

        loss = compute_loss(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss * tpu_strategy.num_replicas_in_sync)
    train_accuracy(labels, predictions)
    return loss


@tf.function
def distributed_train_steps(training_set_iter, steps_per_call):
    for _ in tf.range(steps_per_call):
        per_replica_losses = tpu_strategy.run(train_step, next(training_set_iter))
    # return tpu_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


def test_step(images, labels):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


@tf.function
def distributed_test_step(images, labels):
    return tpu_strategy.run(test_step, args=(images, labels, ))

checkpoint_dir = "gs://{YOUR_GS_BUCKET}/checkpoints"
train_tf_records_dir = "gs://{YOUR_GS_BUCKET}/{YOUR_DATA_SHARD_DIR}/train*"
test_tf_records_dir = "gs://{YOUR_GS_BUCKET}/{YOUR_DATA_SHARD_DIR}/validation*"


training_csv_path = os.path.join(data_dir, "train.csv")
train_csv = pd.read_csv(str(training_csv_path))
num_samples = len(train_csv["id"].tolist())
unique_landmark_ids = train_csv["landmark_id"].unique().tolist()
unique_landmark_ids = tf.convert_to_tensor(unique_landmark_ids, dtype=tf.int64)

training_set = create_dataset(train_tf_records_dir, unique_landmark_ids)
training_set = tpu_strategy.experimental_distribute_dataset(training_set)

test_set = create_dataset(test_tf_records_dir, unique_landmark_ids)
test_set = tpu_strategy.experimental_distribute_dataset(test_set)

with tpu_strategy.scope():
    model = DelfArcFaceModel(
        input_shape=(image_size, image_size, 3), n_classes=len(unique_landmark_ids), margin=margin, logit_scale=logit_scale,
        p=gem_p, train_p=train_p, feature_size=feature_size
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=batch_size)

training_set_iter = iter(training_set)

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    step = 0

    with tqdm(total=int(num_samples)*0.8) as pbar:
        while True:
            distributed_train_steps(training_set_iter, tf.convert_to_tensor(STEPS_PER_TPU_CALL))
            template = 'Epoch {}, Training, Loss: {:.4f}, Accuracy: {:.4f}'
            pbar.set_description(template.format(epoch + 1, train_loss.result(), train_accuracy.result() * 100))
            if step % save_interval == 0:
                if step == 0:
                    model.summary()
                    print()
                    print("\nlearning rate: {}\nmargin: {}\nlogit_scale: {}\ngem_p: {}\ntrain_p{}\n".format(learning_rate, margin, logit_scale, gem_p, train_p))

                checkpoint_path = str(os.path.join(checkpoint_dir, "cp_epoch_{}_step_{}".format(epoch, step)))
                model.save_weights(checkpoint_path)
                print("Model saved to {}".format(checkpoint_path))
            step += batch_size * STEPS_PER_TPU_CALL
            pbar.update(batch_size * STEPS_PER_TPU_CALL)
            if step >= int(num_samples)*0.8:
                break

    with tqdm(total=int(num_samples)*0.2) as pbar:
        for test_images, test_labels in test_set:
            distributed_test_step(test_images, test_labels)
            template = 'Epoch {}, Validation, Loss: {:.4f}, Accuracy: {:.4f}'
            pbar.set_description(template.format(epoch + 1, test_loss.result(), test_accuracy.result() * 100))
            pbar.update(batch_size)

    template = 'Epoch {}, \nTraining Loss: {}, Accuracy: {}\nTest Loss: {}, Accuracy: {}'
    print(template.format(epoch + 1, train_loss.result(), train_accuracy.result() * 100, test_loss.result(), test_accuracy.result() * 100))