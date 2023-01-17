%matplotlib inline

from subprocess import check_output

#print(check_output(["ls", "../working"]).decode("utf8"))

 

#load libraries



#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#from pathlib import Path

#import IPython.display as ipd





#import seaborn as sns



from scipy.io import wavfile

train_audio_path = "../input/train/audio/"

filename = "/yes/012c8314_nohash_0.wav"

sample_rate, samples = wavfile.read(str(train_audio_path)+filename)
from scipy import signal

import numpy as np # linear algebra

def visual_spectogram(audio, sample_rate, window_size=20, step_size=10, eps=1e-10):

    n_per_scnd = int(round(window_size * sample_rate / 1e3))

    n_overlaps = int(round(step_size * sample_rate / 1e3))

    _, _, spec = signal.spectrogram(audio, fs=sample_rate, window='hann',

                                  nperseg = n_per_scnd,

                                  noverlap = n_overlaps,

                                  detrend=False)

    return np.log(spec.T.astype(np.float32) + eps)
import matplotlib.pyplot as plt

spectgrm = visual_spectogram(samples, sample_rate)

figure = plt.figure(figsize=(10,10))

ax1 = figure.add_subplot(211)

ax1.set_title("Wave form for yes")

ax1.set_ylabel("Amplitude")

ax1.plot(samples)

ax2 = figure.add_subplot(212)

ax2.set_title("Spectogram for yes")

ax2.set_ylabel("Features (from 0 to 8000)")

ax2.set_xlabel("Samples")

ax2.imshow(spectgrm.T, aspect="auto",origin="lower")

print(spectgrm)
#normalize values

mean = np.mean(spectgrm, axis = 0)

std = np.std(spectgrm, axis = 0)

spectgrm = (spectgrm - mean) /std

print (spectgrm)
import os

from os.path import isdir, join

directories = [f for f in os.listdir(train_audio_path) if 

              isdir(join(train_audio_path, f))]

directories.sort() # 31 labels

number_of_recs = []

for directory in directories:

    waves = [f for f in os.listdir(join(train_audio_path, directory))]

    number_of_recs.append(len(waves))

plt.figure(figsize=(10,10))

plt.bar(directories, number_of_recs)

plt.title("Number of recs by label")

plt.xticks(rotation="vertical")

plt.ylabel("Y");plt.xlabel("X")

plt.show()

def fastfouriertransform (y, fs):

    T = 1.0 / fs

    N = y.shape[0]

    yf = fft(y)

    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

    vals = 2.0/N * np.abs(yf[0:N//2])

    return xf, vals

    
words = 'yes no up down left right on off stop go silence unknown'.split()

print(directories)
from scipy.fftpack import fft

for directory in directories:

    if directory in words:

        vals_all = []

        spec_all = []

        waves = [f for f in os.listdir(join(train_audio_path, directory))]

        for wav in waves:

            sample_rate,samples = wavfile.read(train_audio_path 

                                               + directory +"/"+wav)

            if samples.shape[0] != 16000:

                continue

            xf, vals = fastfouriertransform(samples, 16000)

            vals_all.append(vals)

            spec_all.append(visual_spectogram(samples, 16000))

            

        plt.figure(figsize=(10,8))

        plt.subplot(121)

        plt.title("Mean foutransf of "+ directory)

        plt.plot(np.mean(np.array(vals_all), axis=0))

        plt.grid()

        plt.subplot(122)

        plt.title("Mean spectgram of "+ directory)

        plt.imshow(np.mean(np.array(spec_all), axis = 0).T, 

                   aspect='auto', origin='lower')

        plt.show()

        
print(waves)
import os

import re

from glob import glob



labels = 'yes no up down left right on off stop go silence unknown'.split()

id2name = {i: name for i, name in enumerate(labels)}

name2id = {name: i for i, name in id2name.items()}

pattern = re.compile("([^_]+)_([^_]+)_.+wav")

data_dir = "../input"

def load_data(data_dir):

    """ 

    Returns 2 lists of tuples: [(class_id, user_id, path), ...] 

    """

    # prefix, label, user_id

    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")

    all_files = glob(os.path.join(data_dir, 'train/audio/*/*wav'))

    with open(os.path.join(data_dir, 'train/validation_list.txt'), 'r') as fin:

        validation_files = fin.readlines()

    validation_set = set()

    for entry in validation_files:

        r = re.match(pattern, entry)

        if r:

            validation_set.add(r.group(3))

    

    possible = set(labels)

    train, val = [], []

    for entry in all_files:

        r = re.match(pattern, entry)

        if r:

            label, uid = r.group(2), r.group(3)

            if label == '_background_noise_':

                label = 'silence'

            if label not in possible:

                label = 'unknown'



            label_id = name2id[label]



            sample = (label_id, uid, entry)

            if uid in validation_set:

                val.append(sample)

            else:

                train.append(sample)



    print('There are {} train and {} val samples'.format(len(train), len(val)))

    return train, val



trainset, valset = load_data(data_dir)

        

import numpy as np

from scipy.io import wavfile

def generate_data(data, params, mode='train'):

    def generator():

        if mode == 'train':

            np.random.shuffle(data)

        for (label_id, uid, fname) in data:

            try:

                _, wav = wavfile.read(fname)

                wav = wav.astype(np.float32) / np.iinfo(np.int16).max

                L = 16000

                if len(wav) < L:

                    continue

                samples_per_file = 1 if label_id != name2id['silence'] else 20

                for _ in range(samples_per_file):

                    if len(wav) > L:

                        beg = np.random.randint(0, len(wav) -L)

                    else:

                        beg = 0

                    yield dict(target=np.int32(label_id),

                              wav = wav[beg: beg + L])

            except Exception as err:

                print(err, label_id, uid, frame)

    return generator
import tensorflow as tf

from tensorflow.contrib import layers



def baseline(x, params, is_training):

    x = layers.batch_norm(x, is_training = is_training)

    for i in range(4):

        x = layers.conv2d(x, 16*(2**i),3,1,activation_fn=tf.nn.elu,

                          normalizer_fn=layers.batch_norm if params.use_batch_norm else None,

                          normalizer_params={'is_training': is_training})

        x = layers.max_pool2d(x,2,2)

    mpool = tf.reduce_max(x, axis=[1, 2], keep_dims=True)

    apool = tf.reduce_mean(x, axis=[1, 2], keep_dims=True)

    

    x = 0.5 * (mpool + apool)

    x = layers.conv2d(x, 128, 1, 1, activation_fn=tf.nn.elu)

    x = tf.nn.dropout(x, keep_prob=params.keep_prob if is_training else 1.0)

    logits = layers.conv2d(x, params.num_classes, 1, 1, activation_fn=None)

    return tf.squeeze(logits, [1,2])

            
from tensorflow.contrib import signal



def model_helper(features, labels, mode, params, config):

    extractor = tf.make_template('extractor', baseline, create_scope_now_=True)

    wav = features['wav']

    specgram = signal.stft(wav, 400, 160)

    phase = tf.angle(specgram) / np.pi

    amp = tf.log1p(tf.abs(specgram))

    x = tf.stack([amp, phase], axis=3)

    x = tf.to_float(x)

    logits = extractor(x, params, mode == tf.estimator.ModeKeys.TRAIN)

    if mode == tf.estimator.ModeKeys.TRAIN:

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,

                                                                            logits=logits))

        def learning_rate_decay_fn(learning_rate, global_step):

            return tf.train.exponential_decay(learning_rate, global_step, 

                                              decay_steps= 10000, decay_rate=0.99)

        

        train_op = tf.contrib.layers.optimize_loss(loss=loss,

                            global_step=tf.contrib.framework.get_global_step(),

                            learning_rate=params.learning_rate,

                            optimizer=lambda lr: tf.train.MomentumOptimizer(lr, 

                                                  0.9, use_nesterov=True),

                            learning_rate_decay_fn =learning_rate_decay_fn,

                            clip_gradients=params.clip_gradients,

                            variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))

        specs = dict(mode=mode, loss=loss, train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:

        prediction = tf.argmax(logits, axis=-1)

        acc, acc_op = tf.metrics.mean_per_class_accuracy(labels, prediction,

                                                        params.num_classes)

        loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(

                            labels=labels, logits=logits))

        specs = dict(mode=mode, loss=loss, eval_metric_ops=dict(acc=(acc,acc_op)))

        if mode == tf.estimator.ModeKeys.PREDICT:

            predictions = {

                'label': tf.argmax(logits, axis=-1),

                'sample': features['sample']

            }

            specs = dict(mode=mode, predictions=predictions)

    if mode == tf.estimator.ModeKeys.PREDICT:

        predictions = {

            'label': tf.argmax(logits, axis=-1),  # for probability just take tf.nn.softmax()

            'sample': features['sample'], # it's a hack for simplicity

        }

        specs = dict(

            mode=mode,

            predictions=predictions,

        )

    return tf.estimator.EstimatorSpec(**specs)



def new_model(config=None, hparams=None):

    return tf.estimator.Estimator(model_fn=model_helper, config=config, params=hdparams)
params = dict(seed=2018, batch_size=64, keep_prob=0.5, learning_rate=1e-3,

             clip_gradients=15.0, use_batch_norm=True, num_classes=len(words))

hparams = tf.contrib.training.HParams(**params)

os.makedirs(os.path.join("../working", 'eval'), exist_ok=True)

model_dir = "../working"

run_config = tf.contrib.learn.RunConfig(model_dir=model_dir)
from tensorflow.contrib.learn.python.learn.learn_io.generator_io import generator_input_fn



train_input_fn = generator_input_fn(x=generate_data(trainset, hparams,

                                                    'train'),

                                    target_key='target',

                                   batch_size=hparams.batch_size,

                                   shuffle=True, num_epochs=None,

                                   queue_capacity=3*hparams.batch_size

                                   + 10, num_threads=1)

val_input_fn = generator_input_fn(x=generate_data(valset,hparams,

                                                  'val'),

                                 target_key='target',

                                 batch_size=hparams.batch_size, 

                                 shuffle=True, num_epochs=None,

                                 queue_capacity=3*hparams.batch_size

                                 + 10, num_threads=1)



def jFlowTest(run_config, hparams):

    exp = tf.contrib.learn.Experiment(

            estimator=new_model(config=run_config, hparams=hparams),

            train_input_fn=train_input_fn,

            eval_input_fn=val_input_fn,

            train_steps=1000,

            eval_steps=20,

            train_steps_per_iteration=100)

    return exp



tf.contrib.learn.learn_runner.run(experiment_fn=jFlowTest,

                                 run_config=run_config,

                                 schedule='continuous_train_and_eval',

                                 hparams=hparams)