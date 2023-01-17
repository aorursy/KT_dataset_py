from ast import literal_eval
import csv
import os
import shutil
import tarfile

import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
tf.__version__
tf.logging.set_verbosity(tf.logging.INFO)
top_k = 3

max_steps = 50000

batch_size = 128
submission_batch_size = 1
eval_dataset_size = 5000
train_dataset_prefetch_size = 10000
train_dataset_shuffle_buffer_size = 100000

image_conv_layers = [(64, (3, 3), 2), (192, (3, 3), 2), (512, (3, 3), 2)]
image_dense_layers = [1024, 128]

sequence_conv_layers = [(256, 5), (256, 3)]
sequence_rnn_num_layers = 2
sequence_rnn_state_size = 256
sequence_rnn_direction = "bidirectional"
sequence_rnn_dropout_prob = 0.3
sequence_dense_layers = [256,]

learning_rate = 0.001
learning_rate_decay_steps = 50000
learning_rate_decay_rate = 0.01
gradient_clipping_norm = 9.0

model_dir = "task"
save_checkpoints_secs = 2400
save_summary_steps = 1000
train_files = os.listdir("../input/train_simplified")
n_classes = len(train_files)
index = { x.split(".")[0]: i for i, x in enumerate(sorted(train_files)) }
inv_index = { v: k.replace(" ", "_") for k, v in index.items() }
# https://www.kaggle.com/huyenvyvy/bidirectional-lstm-using-data-generator-lb-0-825
def _parse_sequence(v):
    a = literal_eval(v.decode("ascii"))
    strokes = [(xi, yi, i) for i, (x, y) in enumerate(a) for xi, yi in zip(x, y)]
    strokes = np.stack(strokes)
    strokes[:, 2] = [1] + np.diff(strokes[:, 2]).tolist()
    # 2 for a new start and 1 for each stroke, as 0 is used for padding/masking
    strokes[:, 2] += 1
    return np.float32(strokes)

# https://www.kaggle.com/marikekoch/quick-draw-mp
def _parse_image(v):
    image = Image.new("P", (256, 256), color=255)
    image_draw = ImageDraw.Draw(image)
    a = literal_eval(v.decode("ascii"))
    for stroke in a:
        for i in range(len(stroke[0]) - 1):
            image_draw.line([stroke[0][i], stroke[1][i], stroke[0][i + 1], stroke[1][i + 1]], fill=0, width=5)
    image = image.resize((32, 32))
    return np.float32(np.array(image) / 255)

def parse_train_row(_, v, k, __, ___, l):
    f = tf.py_func(_parse_sequence, [v], tf.float32, stateful=False)
    f.set_shape((None, 3))
    
    i = tf.py_func(_parse_image, [v], tf.float32, stateful=False)
    i.set_shape((32, 32))
    
    l = tf.py_func(lambda i: np.int32(index[i.decode("ascii")]), [l], tf.int32, stateful=False)
    l.set_shape(())
    
    return { "strokes": f, "images": i, "keys": k, "lengths": tf.shape(f)[0] }, l
dataset = tf.data.Dataset.list_files([
    os.path.join("../input/train_simplified", x) 
    for x in os.listdir("../input/train_simplified")
])

dataset = dataset.repeat()

dataset = dataset.interleave(lambda x: tf.contrib.data.CsvDataset(
    x, 
    [tf.constant([""], dtype=tf.string), tf.string, tf.string, tf.constant([""], dtype=tf.string), tf.constant([""], dtype=tf.string), tf.string],
).skip(1), cycle_length=n_classes, block_length=1)

dataset = dataset.map(parse_train_row)
eval_dataset = dataset
dataset = dataset.prefetch(train_dataset_prefetch_size)
dataset = dataset.shuffle(train_dataset_shuffle_buffer_size)
dataset = dataset.padded_batch(batch_size, padded_shapes=dataset.output_shapes)
dataset
eval_dataset = eval_dataset.padded_batch(batch_size, padded_shapes=eval_dataset.output_shapes)
eval_dataset = eval_dataset.take(eval_dataset_size)
eval_dataset
def image_model_fn(images, training):
    net = tf.expand_dims(images, -1)
    net = tf.layers.batch_normalization(net, training=training)
    
    for l in image_conv_layers:
        net = tf.layers.conv2d(net, filters=l[0], kernel_size=l[1], padding="same", activation=None)
        net = tf.layers.batch_normalization(net, training=training)
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, pool_size=l[2], strides=l[2])
        
    net = tf.layers.flatten(net)
    
    for l in image_dense_layers:
        net = tf.layers.dense(net, l, activation=None)
        net = tf.layers.batch_normalization(net, training=training)
        net = tf.nn.relu(net)
    
    return net
def sequence_model_fn(strokes, lengths, training):
    net = tf.layers.batch_normalization(strokes, training=training)
    
    for l in sequence_conv_layers:
        net = tf.layers.conv1d(net, filters=l[0], kernel_size=l[1], activation=None, padding="same")
        net = tf.layers.batch_normalization(net, training=training)
        net = tf.nn.relu(net)
        
    # CudnnLSTM input is time major
    net = tf.transpose(net, [1, 0, 2])
    net, _ = tf.contrib.cudnn_rnn.CudnnLSTM(
        num_layers=sequence_rnn_num_layers,
        num_units=sequence_rnn_state_size,
        direction=sequence_rnn_direction,
        dropout=sequence_rnn_dropout_prob if training else 0.0,
    )(net)
    net = tf.transpose(net, [1, 0, 2])
    
    # mask out-of-length rnn outputs
    mask = tf.tile(
        tf.expand_dims(
            tf.sequence_mask(
                lengths,
                tf.shape(net)[1],
            ), 
            2,
        ), 
        [1, 1, tf.shape(net)[2]],
    )
    
    net = tf.where(mask, net, tf.zeros_like(net))
    
    net = net[:, -1, :]
    
    for l in sequence_dense_layers:
        net = tf.layers.dense(net, l, activation=None)
        net = tf.layers.batch_normalization(net, training=training)
        net = tf.nn.relu(net)
        
    return net
def model_fn(features, labels, mode):
    """
    features: { "strokes": [?, ?, 3], "keys": [?,], "images": [?, 32, 32], lengths": [?,] }
    labels: [?,]
    """
    training = mode == tf.estimator.ModeKeys.TRAIN
    
    image_output = image_model_fn(features["images"], training=training) # [?, 128]
    sequence_output = sequence_model_fn(features["strokes"], features["lengths"], training=training) # [?, 128]
    
    net = tf.concat([image_output, sequence_output], axis=1)
    
    logits = tf.layers.dense(net, n_classes)
    predictions = tf.nn.softmax(logits, axis=1)
    _, indices = tf.nn.top_k(predictions, k=top_k)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode, 
            predictions={
                "predictions": indices,
                "keys": features["keys"],
            },
        )
    
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits),
    )
    
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode, 
            loss=loss,
            eval_metric_ops={
                # https://stackoverflow.com/a/44801217
                "accuracy": tf.metrics.mean(tf.nn.in_top_k(predictions=predictions, targets=labels, k=top_k)),
            },
        )
    
    # https://github.com/tensorflow/models/blob/master/tutorials/rnn/quickdraw/train_model.py#L233
    train_op = tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        learning_rate=learning_rate,
        learning_rate_decay_fn=lambda l, s: tf.train.exponential_decay(l, s, learning_rate_decay_steps, learning_rate_decay_rate, staircase=True),
        optimizer="Adam",
        # some gradient clipping stabilizes training in the beginning.
        clip_gradients=gradient_clipping_norm,
        summaries=["learning_rate", "loss", "gradients", "gradient_norm"],
    )
    
    return tf.estimator.EstimatorSpec(
        mode=mode, 
        loss=loss,
        train_op=train_op,
    )
runConfig = tf.estimator.RunConfig(
    model_dir=model_dir, 
    save_checkpoints_secs=save_checkpoints_secs,
    save_summary_steps=save_summary_steps,
)

estimator = tf.estimator.Estimator(model_fn, config=runConfig)
train_spec = tf.estimator.TrainSpec(lambda: dataset.make_one_shot_iterator().get_next(), max_steps=max_steps)
eval_spec = tf.estimator.EvalSpec(lambda: eval_dataset.make_one_shot_iterator().get_next())

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
def parse_submission_row(k, _, v):
    f = tf.py_func(_parse_sequence, [v], tf.float32, stateful=False)
    f.set_shape((None, 3))
    
    i = tf.py_func(_parse_image, [v], tf.float32, stateful=False)
    i.set_shape((32, 32))
    
    return { "strokes": f, "images": i, "keys": k, "lengths": tf.shape(f)[0] }
submission_dataset = tf.contrib.data.CsvDataset("../input/test_simplified.csv", [tf.string, tf.constant([""], dtype=tf.string), tf.string]).skip(1)
submission_dataset = submission_dataset.map(parse_submission_row)
submission_dataset = submission_dataset.padded_batch(submission_batch_size, padded_shapes=submission_dataset.output_shapes)
submission_dataset
predictions = estimator.predict(lambda: submission_dataset.make_one_shot_iterator().get_next())
rows = []
rows.append(["key_id", "word"])

for p in predictions:
    rows.append([p["keys"].decode("ascii"), " ".join([inv_index[x] for x in p["predictions"]])])

len(rows), rows[:25]
with open("submission.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(rows)
with tarfile.open("task.tar.gz", "w:gz") as tar:
    tar.add("task", arcname=os.path.basename("task"))
shutil.rmtree("task")