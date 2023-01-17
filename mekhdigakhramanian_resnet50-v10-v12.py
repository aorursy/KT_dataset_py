import numpy as np
import tensorflow as tf
import math
import pandas as pd
from sklearn import model_selection
import glob
import os
from zipfile import ZipFile
import shutil
import tqdm.notebook as tqdm

import logging
tf.get_logger().setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

gpus = tf.config.experimental.list_physical_devices('GPU')
num_gpus = len(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(num_gpus, "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

#     policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
#     tf.keras.mixed_precision.experimental.set_policy(policy)
#     print('Compute dtype: %s' % policy.compute_dtype)
#     print('Variable dtype: %s' % policy.variable_dtype)

    
if num_gpus == 0:
    strategy = tf.distribute.OneDeviceStrategy(device='CPU')
    print("Setting strategy to OneDeviceStrategy(device='CPU')")
elif num_gpus == 1:
    strategy = tf.distribute.OneDeviceStrategy(device='GPU')
    print("Setting strategy to OneDeviceStrategy(device='GPU')")
else:
    strategy = tf.distribute.MirroredStrategy()
    print("Setting strategy to MirroredStrategy()")
config = {
    'learning_rate': 1e-3,
    'momentum': 0.9,
    'scale': 30,
    'margin': 0.5,
    'clip_grad': 10.0,
    'n_epochs': 20,
    'batch_size': 48,
    'input_size': (384, 384, 3),
    'n_classes': 81313,
    'dense_units': 512,
    'dropout_rate': 0.0,
}
def read_submission_file(input_path, alpha=0.5):
    files_paths = glob.glob(input_path + 'test/*/*/*/*')
    mapping = {}
    for path in files_paths:
        mapping[path.split('/')[-1].split('.')[0]] = path
    df = pd.read_csv(input_path + 'sample_submission.csv')
    df['path'] = df['id'].map(mapping)
    df['label'] = -1
    df['prob'] = -1
    return df

def read_train_file(input_path, alpha=0.5):
    files_paths = glob.glob(input_path + 'train/*/*/*/*')
    mapping = {}
    for path in files_paths:
        mapping[path.split('/')[-1].split('.')[0]] = path
    df = pd.read_csv(input_path + 'train.csv')
    df['path'] = df['id'].map(mapping)
    
    counts_map = dict(
        df.groupby('landmark_id')['path'].agg(lambda x: len(x)))
    df['counts'] = df['landmark_id'].map(counts_map)
    df['prob'] = (
        (1/df.counts**alpha) / (1/df.counts**alpha).max()).astype(np.float32)
    uniques = df['landmark_id'].unique()
    df['label'] = df['landmark_id'].map(dict(zip(uniques, range(len(uniques)))))
    return df, dict(zip(range(len(uniques)), uniques))


submission_df = read_submission_file('../input/landmark-recognition-2020/')
train_df, mapping = read_train_file('../input/landmark-recognition-2020/')
train_df.head(10)

def _get_transform_matrix(rotation, shear, hzoom, wzoom, hshift, wshift):

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])

    # convert degrees to radians
    rotation = math.pi * rotation / 360.
    shear    = math.pi * shear    / 360.

    one  = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')

    c1   = tf.math.cos(rotation)
    s1   = tf.math.sin(rotation)
    rot_mat = get_3x3_mat([c1,    s1,   zero ,
                           -s1,   c1,   zero ,
                           zero,  zero, one ])

    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_mat = get_3x3_mat([one,  s2,   zero ,
                             zero, c2,   zero ,
                             zero, zero, one ])

    zoom_mat = get_3x3_mat([one/hzoom, zero,      zero,
                            zero,      one/wzoom, zero,
                            zero,      zero,      one])

    shift_mat = get_3x3_mat([one,  zero, hshift,
                             zero, one,  wshift,
                             zero, zero, one   ])

    return tf.matmul(
        tf.matmul(rot_mat, shear_mat),
        tf.matmul(zoom_mat, shift_mat)
    )

def _spatial_transform(image,
                       rotation=3.0,
                       shear=2.0,
                       hzoom=8.0,
                       wzoom=8.0,
                       hshift=8.0,
                       wshift=8.0):

    ydim = tf.gather(tf.shape(image), 0)
    xdim = tf.gather(tf.shape(image), 1)
    xxdim = xdim % 2
    yxdim = ydim % 2

    # random rotation, shear, zoom and shift
    rotation = rotation * tf.random.normal([1], dtype='float32')
    shear = shear * tf.random.normal([1], dtype='float32')
    hzoom = 1.0 + tf.random.normal([1], dtype='float32') / hzoom
    wzoom = 1.0 + tf.random.normal([1], dtype='float32') / wzoom
    hshift = hshift * tf.random.normal([1], dtype='float32')
    wshift = wshift * tf.random.normal([1], dtype='float32')

    m = _get_transform_matrix(
        rotation, shear, hzoom, wzoom, hshift, wshift)

    # origin pixels
    y = tf.repeat(tf.range(ydim//2, -ydim//2,-1), xdim)
    x = tf.tile(tf.range(-xdim//2, xdim//2), [ydim])
    z = tf.ones([ydim*xdim], dtype='int32')
    idx = tf.stack([y, x, z])

    # destination pixels
    idx2 = tf.matmul(m, tf.cast(idx, dtype='float32'))
    idx2 = tf.cast(idx2, dtype='int32')
    # clip to origin pixels range
    idx2y = tf.clip_by_value(idx2[0,], -ydim//2+yxdim+1, ydim//2)
    idx2x = tf.clip_by_value(idx2[1,], -xdim//2+xxdim+1, xdim//2)
    idx2 = tf.stack([idx2y, idx2x, idx2[2,]])

    # apply destinations pixels to image
    idx3 = tf.stack([ydim//2-idx2[0,], xdim//2-1+idx2[1,]])
    d = tf.gather_nd(image, tf.transpose(idx3))
    image = tf.reshape(d, [ydim, xdim, 3])
    return image

def _pixel_transform(image,
                     saturation_delta=0.3,
                     contrast_delta=0.1,
                     brightness_delta=0.2):
    image = tf.image.random_saturation(
        image, 1-saturation_delta, 1+saturation_delta)
    image = tf.image.random_contrast(
        image, 1-contrast_delta, 1+contrast_delta)
    image = tf.image.random_brightness(
        image, brightness_delta)
    return image

def preprocess_input(image, target_size, augment=False):
    
    image = tf.image.resize(
        image, target_size, method='bilinear')

    image = tf.cast(image, tf.uint8)
    if augment:
        image = _spatial_transform(image)
        image = _pixel_transform(image)
    image = tf.cast(image, tf.float32)
    image /= 255.
    return image

def create_dataset(df, training, batch_size, input_size):

    def read_image(image_path):
        image = tf.io.read_file(image_path)
        return tf.image.decode_jpeg(image, channels=3)
    
    def filter_by_probs(x, y, p):
        if p > np.random.uniform(0, 1):
            return True
        return False

    image_paths, labels, probs = df.path, df.label, df.prob

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels, probs))
    if training:
        dataset = dataset.shuffle(100_000)
    dataset = dataset.map(
        lambda x, y, p: (read_image(x), y, p),
        tf.data.experimental.AUTOTUNE)
    if training:
        dataset = dataset.filter(filter_by_probs)
    dataset = dataset.map(
        lambda x, y, p: (preprocess_input(x, input_size[:2], training), y),
        tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset
class ArcMarginProduct(tf.keras.layers.Layer):
    '''
    Implements large margin arc distance.

    Reference:
        https://arxiv.org/pdf/1801.07698.pdf
        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/
            blob/master/src/modeling/metric_learning.py
    '''
    def __init__(self, n_classes, s=30, m=0.50, easy_margin=False,
                 ls_eps=0.0, **kwargs):

        super(ArcMarginProduct, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.easy_margin = easy_margin
        self.cos_m = tf.math.cos(m)
        self.sin_m = tf.math.sin(m)
        self.th = tf.math.cos(math.pi - m)
        self.mm = tf.math.sin(math.pi - m) * m

    def build(self, input_shape):
        super(ArcMarginProduct, self).build(input_shape[0])

        self.W = self.add_weight(
            name='W',
            shape=(int(input_shape[0][-1]), self.n_classes),
            initializer='glorot_uniform',
            dtype='float32',
            trainable=True,
            regularizer=None)

    def call(self, inputs):
        X, y = inputs
        y = tf.cast(y, dtype=tf.int32)
        cosine = tf.matmul(
            tf.math.l2_normalize(X, axis=1),
            tf.math.l2_normalize(self.W, axis=0)
        )
        sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine)
        else:
            phi = tf.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = tf.cast(
            tf.one_hot(y, depth=self.n_classes),
            dtype=cosine.dtype
        )
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.n_classes

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


def create_model(input_shape,
                 n_classes,
                 dense_units=512,
                 dropout_rate=0.0,
                 scale=30,
                 margin=0.3):

    backbone = tf.keras.applications.ResNet50(
        include_top=False,
        input_shape=input_shape,
        weights=('../input/imagenet-weights/' +
                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    )

    pooling = tf.keras.layers.GlobalAveragePooling2D(name='head/pooling')
    dropout = tf.keras.layers.Dropout(dropout_rate, name='head/dropout')
    dense = tf.keras.layers.Dense(dense_units, name='head/dense')

    margin = ArcMarginProduct(
        n_classes=n_classes,
        s=scale,
        m=margin,
        name='head/arc_margin',
        dtype='float32')

    softmax = tf.keras.layers.Softmax(dtype='float32')

    image = tf.keras.layers.Input(input_shape, name='input/image')
    label = tf.keras.layers.Input((), name='input/label')

    x = backbone(image)
    x = pooling(x)
    x = dropout(x)
    x = dense(x)
    x = margin([x, label])
    x = softmax(x)
    return tf.keras.Model(
        inputs=[image, label], outputs=x)


class DistributedModel:

    def __init__(self,
                 input_size,
                 n_classes,
                 batch_size,
                 finetuned_weights,
                 dense_units,
                 dropout_rate,
                 scale,
                 margin,
                 optimizer,
                 strategy,
                 mixed_precision,
                 clip_grad):

        self.model = create_model(
            input_shape=input_size,
            n_classes=n_classes,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            scale=scale,
            margin=margin,)

        self.input_size = input_size
        self.global_batch_size = batch_size * strategy.num_replicas_in_sync

        if finetuned_weights:
            self.model.load_weights(finetuned_weights)

        self.mixed_precision = mixed_precision
        self.optimizer = optimizer
        self.strategy = strategy
        self.clip_grad = clip_grad

        # loss function
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.NONE)

        # metrics
        self.mean_loss_train = tf.keras.metrics.SparseCategoricalCrossentropy(
            from_logits=False)
        self.mean_accuracy_train = tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=5)

        if self.optimizer and self.mixed_precision:
            self.optimizer = \
                tf.keras.mixed_precision.experimental.LossScaleOptimizer(
                    optimizer, loss_scale='dynamic')

    def _compute_loss(self, labels, probs):
        per_example_loss = self.loss_object(labels, probs)
        return tf.nn.compute_average_loss(
            per_example_loss, global_batch_size=self.global_batch_size)

    def _backprop_loss(self, tape, loss, weights):
        gradients = tape.gradient(loss, weights)
        if self.mixed_precision:
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        clipped, _ = tf.clip_by_global_norm(gradients, clip_norm=self.clip_grad)
        self.optimizer.apply_gradients(zip(clipped, weights))

    def _train_step(self, inputs):
        with tf.GradientTape() as tape:
            probs = self.model(inputs, training=True)
            loss = self._compute_loss(inputs[1], probs)
            if self.mixed_precision:
                loss = self.optimizer.get_scaled_loss(loss)
        self._backprop_loss(tape, loss, self.model.trainable_weights)
        self.mean_loss_train.update_state(inputs[1], probs)
        self.mean_accuracy_train.update_state(inputs[1], probs)
        return loss
    
    def _predict_step(self, inputs):
        probs = self.model(inputs, training=False)
        return probs
    
    @tf.function
    def _distributed_train_step(self, dist_inputs):
        per_replica_loss = self.strategy.run(self._train_step, args=(dist_inputs,))
        return self.strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
    
    @tf.function
    def _distributed_predict_step(self, dist_inputs):
        probs = self.strategy.run(self._predict_step, args=(dist_inputs,))
        return probs
    
    def train(self, train_ds, epochs, save_path):
        for epoch in range(epochs):
            dist_train_ds = self.strategy.experimental_distribute_dataset(train_ds)
            dist_train_ds = tqdm.tqdm(dist_train_ds)
            for i, inputs in enumerate(dist_train_ds):
                loss = self._distributed_train_step(inputs)
                dist_train_ds.set_description(
                    "TRAIN: Loss {:.3f}, Accuracy {:.3f}".format(
                        self.mean_loss_train.result().numpy(),
                        self.mean_accuracy_train.result().numpy()
                    )
                )
            if save_path:
                self.model.save_weights(save_path)

            self.mean_loss_train.reset_states()
            self.mean_accuracy_train.reset_states()
    
    def predict(self, test_ds):
        dist_test_ds = self.strategy.experimental_distribute_dataset(test_ds)
        dist_test_ds = tqdm.tqdm(dist_test_ds)
        # initialize accumulators
        predictions = np.zeros([0,], dtype='int32')
        confidences = np.zeros([0,], dtype='float32')
        for inputs in dist_test_ds:
            probs_replicates = self._distributed_predict_step(inputs)
            probs_replicates = self.strategy.experimental_local_results(probs_replicates)
            for probs in probs_replicates:
                m = tf.gather(tf.shape(probs), 0)
                probs_argsort = tf.argsort(probs, direction='DESCENDING')
                # obtain predictions
                idx1 = tf.stack([tf.range(m), tf.zeros(m, dtype='int32')], axis=1)
                preds = tf.gather_nd(probs_argsort, idx1)
                # obtain confidences
                idx2 = tf.stack([tf.range(m), preds], axis=1)
                confs = tf.gather_nd(probs, idx2)
                # add to accumulator
                predictions = np.concatenate([predictions, preds], axis=0)
                confidences = np.concatenate([confidences, confs], axis=0)
        return predictions, confidences

# skip training and predicting on public data
if submission_df.id[0] != '00084cdf8f600d00':

    train_ds = create_dataset(
        df=train_df,
        training=True,
        batch_size=config['batch_size'],
        input_size=config['input_size'],
    )

    test_ds = create_dataset(
        df=submission_df,
        training=False,
        batch_size=config['batch_size'],
        input_size=config['input_size'],
    )

    with strategy.scope():

        optimizer = tf.keras.optimizers.SGD(
            config['learning_rate'], momentum=config['momentum'])

        dist_model = DistributedModel(
            input_size=config['input_size'],
            n_classes=config['n_classes'],
            batch_size=config['batch_size'],
            finetuned_weights=None,
            dense_units=config['dense_units'],
            dropout_rate=config['dropout_rate'],
            scale=config['scale'],
            margin=config['margin'],
            optimizer=optimizer,
            strategy=strategy,
            mixed_precision=False,
            clip_grad=config['clip_grad'])

        dist_model.train(
            train_ds=train_ds, 
            epochs=config['n_epochs'], 
            save_path=None)#'model.h5')

        preds, confs = dist_model.predict(
            test_ds=test_ds)

    
    for i, (pred, conf) in enumerate(zip(preds, confs)):
        if conf < 0.1:
            submission_df.at[i, 'landmarks'] = ''
        else:
            submission_df.at[i, 'landmarks'] = f'{mapping[pred]} {conf}'
            
    submission_df = submission_df.set_index('id')
    submission_df = submission_df.drop('label', axis=1)
    submission_df = submission_df.drop('prob', axis=1)
    submission_df = submission_df.drop('path', axis=1)
    submission_df.to_csv('submission.csv')
else:
    submission_df = submission_df.set_index('id')
    submission_df = submission_df.drop('label', axis=1)
    submission_df = submission_df.drop('prob', axis=1)
    submission_df = submission_df.drop('path', axis=1)
    submission_df.to_csv('submission.csv')
    
    
