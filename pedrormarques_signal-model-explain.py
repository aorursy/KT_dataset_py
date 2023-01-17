# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

DATADIR = "../input/LANL-Earthquake-Prediction"



# Any results you write to the current directory are saved as output.
df = pd.read_csv(os.path.join(DATADIR, 'train.csv'),

                 dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
import linear_signal_py as linear_signal



class SignalFeatures(linear_signal.SignalFeatureGenerator):

    SEQUENCE_LENGHT = 150_000



    S_MEAN = 4

    S_STD = 10



    def __init__(self, normalize=True):

        self.normalize = normalize



    def shape(self):

        return (self.SEQUENCE_LENGHT, 1)



    def generate(self, df: pd.DataFrame, predict=False):

        """ The performance of this function when vectorized is 10x of

        an iterative loop.

        """

        X = df['acoustic_data'].values[:, np.newaxis]

        if self.normalize:

            X = (X - self.S_MEAN) / self.S_STD

        if predict:

            return X

        y = df['time_to_failure'].iloc[df.shape[0] - 1]

        return X, np.array([y])
class StaticSplit(object):

    def __init__(self, n_blocks, test_slice):

        self.n_blocks = n_blocks

        indices = set(range(n_blocks))

        for i in test_slice:

            indices.remove(i)

        self.train_slice = list(indices)

        self.test_slice = test_slice

    def train(self):

        return self.train_slice

    def test(self):

        return self.test_slice



kFolds = [

    StaticSplit(25, [0, 1])

]
SEGMENT_SIZE = 150_000

STRIDES = 4_000



def get_generators(spliter):

    ds_train = linear_signal.LinearDatasetAccessor(df, spliter.n_blocks, spliter.train())

    ds_eval = linear_signal.LinearDatasetAccessor(df, spliter.n_blocks, spliter.test())



    gen_train = linear_signal.LinearSignalGenerator(

        ds_train, SEGMENT_SIZE, SignalFeatures(), strides=STRIDES, batch_size=256)

    gen_eval = linear_signal.LinearSignalGenerator(

        ds_eval, SEGMENT_SIZE, SignalFeatures(), strides=16_000)



    return gen_train, gen_eval
from tensorflow.keras.models import load_model

model = load_model('../input/signal-convolution/model-signal-conv.h5')

model.load_weights('../input/signal-convolution/signal-conv.0.ckpt.hdf5')
gen_train, gen_eval = get_generators(kFolds[0])

model.evaluate_generator(gen_eval)
import shap



background = gen_train[0][0]

e = shap.DeepExplainer(model, background)

test_values = gen_eval[0][0][:4]

y_true = gen_eval[0][1][:4]

y_pred = model.predict(test_values)

shap_values = e.shap_values(test_values)
%matplotlib inline

import matplotlib.pyplot as plt



fig, axes = plt.subplots(8, 1, figsize=(20, 20), sharex=True)

for i, ax in enumerate(axes):

    ex = i // 2

    if i % 2 == 0:

        ax.set_title('y_true: {0}'.format(y_true[ex]))

        ax.set(ylim=(-15.0, 15.0))

        ax.plot(test_values[ex], c='b')

    else:

        s = shap_values[0][ex]

        neg = np.ma.masked_where(s < 0.0, s)

        pos = np.ma.masked_where(s > 0.0, s)

        ax.set(ylim=(-0.05, 0.05))

        ax.set_title('y_pred: {0}'.format(y_pred[ex]))

        x = np.arange(150_000)

        ax.plot(x, neg, x, pos)

plt.show()
from tensorflow.keras.models import Model

layer_outputs = [layer.output for layer in model.layers[1:]]



viz_model = Model(inputs = model.input, outputs = layer_outputs)



background_outs = viz_model.predict(background)

outs = viz_model.predict(test_values)

def show_conv_layers(outs, ex, background_outs=None):

    width = np.max(np.array([outs[i*2].shape[-1] for i in range(4)]))

    fig = plt.figure(figsize=(20, 10))

    gs = fig.add_gridspec(4, width, wspace=0.0)

    for i in range(4):

        ax_left = None

        index = i * 2

        x = outs[index][ex]

        if background_outs is not None:

            b = background_outs[index]

            x -= b.mean()

            x /= b.std()

        else:

            x -= x.mean()

            x /= x.std()



        for j in range(outs[index].shape[-1]):

            step = width // outs[index].shape[-1]

            s = slice(j * step, (j + 1) * step)

            g = gs[i, s]

            ax = fig.add_subplot(g, sharey=ax_left)

            if j == 0:

                ax_left = ax

            else:

                plt.setp(ax.get_yticklabels(), visible=False)

            ax.plot(x[:,j])



    plt.show()
show_conv_layers(outs, 0, background_outs)
show_conv_layers(outs, 1, background_outs)