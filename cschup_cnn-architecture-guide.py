import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.gridspec as pltgs

import torch

import keras

from keras.layers import Conv2D

from keras.layers import MaxPooling2D, AveragePooling2D

from keras.layers import Dense, Flatten

from keras.models import Sequential

from keras import backend as K

from keras.utils import to_categorical
class Plot():

    def __init__(self, width=10, height=4, cmap='magma', size=12, bot=0.1):

        self.width = width

        self.height = height

        self.cmap = cmap

        self.size = size

        self.fig = plt.figure(figsize=(self.width, self.height))

        self.bot = bot



    def _network_dict(self, network_id):

        c = 'Convolution\n+ '

        dictionary = {

            0: [c + 'ReLU', r'Max Pooling $(2 \times 2)$'],

            1: [c + 'Tanh', r'Max Pooling $(2 \times 2)$'],

            2: [c + 'ReLU', r'Average Pooling $(2 \times 2)$'],

            3: [c + 'Tanh', r'Average Pooling $(2 \times 2)$'],

            4: [c + 'Tanh', r'Max Pooling $(2 \times 2)$', c + 'ReLU'],

            5: [c + 'ReLU', r'Max Pooling $(2 \times 2)$', c + 'ReLU',

                c + 'ReLU', r'Max Pooling $(2 \times 2)$']

        }

        return dictionary[network_id]



    def _no_ticks(self, ax):

        ax.set_xticks([])

        ax.set_yticks([])

        return None



    def _get_lims(self, ax):

        return np.array(ax.get_xlim()), np.array(ax.get_ylim())



    def _shape_label(self, x, xlim, ylim):

        _shape = tuple([x.shape[i] for i in [1, 2, 0]])

        plt.text(np.mean(xlim), np.max(ylim)*1.05, _shape,

                 ha='center', va='top', size=self.size)

        return None



    def _level_locs(self, levels):

        x1 = [1 / (levels * 4)]

        x1 += [(i + 0.8) / levels for i in range(1, levels-1)]

        x2 = [(i + 0.7) / levels for i in range(1, levels-1)]

        x2 += [(levels - 0.25) / levels]

        return np.array(x1), np.array(x2)





    def _network_desc(self, levels, network_id):

        x1, x2 = self._level_locs(levels)

        labels = self._network_dict(network_id)

        gs = pltgs.GridSpec(1, 1, left=0, right=1, top=self.bot, bottom=0)

        ax = self.fig.add_subplot(gs[0,0])

        for i in range(len(x1)):

            ax.plot([x1[i], x1[i]], [0.8, 0.7], c='k')

            ax.plot([x1[i], x2[i]], [0.7, 0.7], c='k')

            ax.plot([x2[i], x2[i]], [0.8, 0.7], c='k')

            ax.text((x1[i] + x2[i]) / 2, 0.55, labels[i],

                     ha='center', va='top', size=self.size)

        ax.set_ylim(0, 1)

        ax.set_xlim(0, 1)

        ax.axis('off')

        return None



    def _conv_desc(self, levels, activation):

        x1, x2 = self._level_locs(levels)

        label = 'Convolution'

        if len(activation) > 0:

            label += f'\n+ {activation}'

        gs = pltgs.GridSpec(1, 1, left=0, right=1, top=1, bottom=0)

        ax = self.fig.add_subplot(gs[0,0])

        for i in range(len(x1)):

            ax.plot([0.27, 0.45], [0.50, 0.50], c='k')

            ax.plot([0.45, 0.44], [0.50, 0.52], c='k')

            ax.plot([0.45, 0.44], [0.50, 0.48], c='k')

            ax.text((0.27 + 0.45) / 2, 0.47, label, ha='center', va='top',

                    size=self.size)

        ax.set_ylim(0, 1)

        ax.set_xlim(0, 1)

        ax.axis('off')

        return None



    def _plot_input(self, x_input, levels):

        gs = pltgs.GridSpec(1, 2, left=0, right=1/levels, top=1,

                            bottom=self.bot)

        ax = self.fig.add_subplot(gs[0,0])

        ax.imshow(x_input[0,0], cmap=self.cmap, aspect='equal')

        xlim, ylim = self._get_lims(ax)

        plt.text(np.mean(xlim), -np.max(ylim)*0.05, 'Input',

                 ha='center', va='bottom', size=self.size)

        self._shape_label(x_input[0], xlim, ylim)

        self._no_ticks(ax)

        return None



    def _network_layout(self, x_input, x_list):

        input_size = x_input[0,0].shape[0]

        xrng = np.arange(len(x_list))

        x = [x_list[i][0,:,:,:] for i in xrng]

        layers = [x[i].shape[0] for i in xrng]

        size_ratio = [x[i].shape[1] / input_size for i in xrng]

        ws = [(1 / ((layers[i] - 1) * size_ratio[i])) - 1 for i in xrng]

        hs = [(1 / ((layers[i] - 1) * size_ratio[i])) - 1 for i in xrng]

        return x, xrng, layers, ws, hs



    def _plot_network(self, x_input, x_list, levels):

        x, xrng, layers, ws, hs = self._network_layout(x_input, x_list)

        for i in xrng:

            gs = pltgs.GridSpec(layers[i], layers[i], left=(i+1)/levels,

                                right=(i+2)/levels, top=1, bottom=self.bot,

                                wspace=ws[i], hspace=hs[i])

            for j in range(layers[i]):

                ax = self.fig.add_subplot(gs[j,j])

                ax.imshow(x[i][j], cmap=self.cmap, aspect='equal')

                self._no_ticks(ax)

            xlim, ylim = self._get_lims(ax)

            self._shape_label(x[i], xlim, ylim)

        return None



    def network(self, x_input, x_list, activation='', network_id=0,

                channels='first'):

        if channels == 'last':

            x_input = np.transpose(x_input, [0, 3, 1, 2])

            x_list = [np.transpose(x, [0, 3, 1, 2]) for x in x_list]

        levels = len(x_list) + 1

        if levels > 2:

            self._network_desc(levels, network_id)

        else:

            self.bot = 0

            self._conv_desc(levels, activation)

        self._plot_input(x_input, levels)

        self._plot_network(x_input, x_list, levels)

        return None
def normalize(x):

    return x.astype('float32') / 255





def plot_samples(x_train, y_train):

    class_dict = np.arange(10)

    sample_list = [x_train[y_train[:,i].astype(bool)][:12] for i in range(10)]

    samples = np.concatenate(sample_list)

    gs = pltgs.GridSpec(10, 12, hspace=-0.025, wspace=-0.025)

    fig = plt.figure(figsize=(10, 8.5))

    yloc = np.linspace(0.95, 0.05, 10)

    for i in range(120):

        ax = fig.add_subplot(gs[i//12, i%12])

        ax.imshow(samples[i,:,:,0], cmap='magma')

        ax.set_xticks([])

        ax.set_yticks([])

    return None





def display_conv(x, conv, activation):

    x1 = K.eval(conv(x))

    Plot().network(K.eval(x), [x1], activation=activation, channels='last')

    return None





def display_conv_pool(x, conv_list, network_id, height=5):

    x_list = [conv_list[0](x)]

    for i in range(1, len(conv_list)):

        x_list += [conv_list[i](x_list[i-1])]

    x_list = [K.eval(x_list[i]) for i in range(len(conv_list))]

    Plot(height=height).network(K.eval(x), x_list, network_id=network_id,

                                channels='last')

    return None





def display_fc(model, x):

    label = model.get_layer(index=-1).name

    fc = K.eval(model(x))

    plt.figure(figsize=(10, 0.5))

    plt.imshow(fc, cmap='magma', aspect='auto')

    plt.gca().set_xticks([])

    plt.gca().set_yticks([])

    xloc = np.array(plt.gca().get_xlim())

    plt.text(xloc[0], -0.6, label, ha='left', va='bottom', size=12)

    plt.text(np.mean(xloc), 0.7, fc.shape, ha='center', va='top', size=12)

    return None
train = pd.read_csv('../input/digit-recognizer/train.csv').to_numpy()

test = pd.read_csv('../input/digit-recognizer/test.csv').to_numpy()



x_train = normalize(train[:,1:]).reshape(-1, 28, 28, 1)

x_test = normalize(test).reshape(-1, 28, 28, 1)

y_train = to_categorical(train[:,0])



print(f'  Train data   shape = {x_train.shape}')

print(f'   Test data   shape = {x_test.shape}')

print(f'Train labels   shape = {y_train.shape}')
plot_samples(x_train, y_train)
x = x_train[39:40,:,:,:1]
x = np.transpose(x, [0, 3, 1, 2])

x = torch.Tensor(x)

print(f'x = {x.shape}')
conv1 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3,

                        stride=1, padding=0)
conv2 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3,

                        stride=1, padding=1)
x1 = conv1(x).detach().numpy()

x2 = conv2(x).detach().numpy()
Plot().network(x, [x1], channels='first')
Plot().network(x, [x2], channels='first')
x = x.detach().numpy()

x = np.transpose(x, [0, 2, 3, 1])

x = K.constant(x)
relu1 = Conv2D(filters=3, kernel_size=3, strides=1, padding='valid',

               activation='relu')

relu2 = Conv2D(filters=6, kernel_size=3, strides=1, padding='same',

               activation='relu')
display_conv(x, relu1, activation='ReLU')
display_conv(x, relu2, activation='ReLU')
softmax = Conv2D(filters=3, kernel_size=3, activation='softmax')
display_conv(x, softmax, activation='Softmax')
sigmoid = Conv2D(filters=3, kernel_size=3, activation='sigmoid')
display_conv(x, sigmoid, activation='Sigmoid')
tanh = Conv2D(filters=3, kernel_size=3, activation='tanh')
display_conv(x, tanh, activation='Tanh')
elu = Conv2D(filters=3, kernel_size=3, activation='elu')
display_conv(x, elu, activation='ELU')
softplus = Conv2D(filters=3, kernel_size=3, activation='softplus')
display_conv(x, softplus, activation='Softplus')
softsign = Conv2D(filters=3, kernel_size=3, activation='softsign')
display_conv(x, softsign, activation='Softsign')
relu = Conv2D(filters=6, kernel_size=3, padding='same', activation='relu')

tanh = Conv2D(filters=6, kernel_size=3, padding='same', activation='tanh')
maxpool = MaxPooling2D(pool_size=(2, 2))
display_conv_pool(x, [relu, maxpool], network_id=0)
display_conv_pool(x, [tanh, maxpool], network_id=1)
avgpool = AveragePooling2D(pool_size=(2, 2))
display_conv_pool(x, [relu, avgpool], network_id=2)
display_conv_pool(x, [tanh, avgpool], network_id=3)
relu = Conv2D(filters=6, kernel_size=3, padding='same', activation='relu')

tanh = Conv2D(filters=6, kernel_size=3, padding='same', activation='tanh')
display_conv_pool(x, [tanh, maxpool, relu], network_id=4)
maxpool = MaxPooling2D(pool_size=(2, 2))

relu1 = Conv2D(filters=8, kernel_size=3, padding='same', activation='relu')

relu2 = Conv2D(filters=16, kernel_size=3, padding='same', activation='relu')

relu3 = Conv2D(filters=16, kernel_size=3, activation='relu')
display_conv_pool(x, [relu1, maxpool, relu2, relu3, maxpool],

                  network_id=5)
model = Sequential()

model.add(Conv2D(8, 3, padding='same', activation='relu',

                 input_shape=(28, 28, 1), name='conv_1'))

model.add(Conv2D(8, 3, padding='same', activation='relu', name='conv_2'))

model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool_1'))

model.add(Conv2D(16, 3, activation='relu', name='conv_3'))

model.add(Conv2D(16, 3, activation='relu', name='conv_4'))

model.add(Conv2D(16, 3, activation='relu', name='conv_5'))

model.add(MaxPooling2D(pool_size=(2, 2), name='maxpool_2'))
model.add(Flatten(name='flatten_1'))

display_fc(model, x)
model.add(Dense(128, activation='relu', name='dense_1'))

display_fc(model, x)
model.add(Dense(128, activation='relu', name='dense_2'))

display_fc(model, x)
model.add(Dense(10, activation='softmax', name='dense_3'))

display_fc(model, x)
model.summary()