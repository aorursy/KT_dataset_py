import os

import gc

import cv2

import json

import time



import numpy as np

import pandas as pd

from pathlib import Path

from keras.utils import to_categorical



import seaborn as sns

import plotly.express as px

from matplotlib import colors

import matplotlib.pyplot as plt

import plotly.figure_factory as ff



import torch

T = torch.Tensor

import torch.nn as nn

from torch.optim import Adam

from torch.utils.data import Dataset, DataLoader
SIZE = 800

EPOCHS = 30

CONV_OUT_1 = 50

CONV_OUT_2 = 100

BATCH_SIZE = 32



TEST_PATH = Path('../input/abstraction-and-reasoning-challenge/')

SUBMISSION_PATH = Path('../input/abstraction-and-reasoning-challenge/')



TEST_PATH = TEST_PATH / 'test'

SUBMISSION_PATH = SUBMISSION_PATH / 'sample_submission.csv'
test_task_files = sorted(os.listdir(TEST_PATH))



test_tasks = []

task_ids = []

for task_file in test_task_files:

    with open(str(TEST_PATH / task_file), 'r') as f:

        task = json.load(f)

        test_tasks.append(task)

        task_ids.append(task_file[:task_file.find(".")])

        if "00576224" == task_file[:-5]:

            print(task)
task_ids
Xs_test, Xs_train, ys_train = [], [], []



for task in test_tasks:

    X_test, X_train, y_train = [], [], []



    for pair in task["test"]:

        X_test.append(pair["input"])



    for pair in task["train"]:

        X_train.append(pair["input"])

        y_train.append(pair["output"])

    

    Xs_test.append(X_test)

    Xs_train.append(X_train)

    ys_train.append(y_train)
matrices = []

for X_test in Xs_test:

    for X in X_test:

        matrices.append(X)

        

values = []

for matrix in matrices:

    for row in matrix:

        for value in row:

            values.append(value)

            

df = pd.DataFrame(values)

df.columns = ["values"]
data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')

training_path = data_path / 'training'

training_tasks = sorted(os.listdir(training_path))



def plot_matrix(matrix, ax):

    cmap = colors.ListedColormap(

        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',

         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

    norm = colors.Normalize(vmin=0, vmax=9)

    ax.imshow(matrix, cmap=cmap, norm=norm)

    width = np.shape(matrix)[1]

    height = np.shape(matrix)[0]

    ax.set_xticks(np.arange(0,width))

    ax.set_yticks(np.arange(0,height))

    ax.set_xticklabels([])

    ax.set_yticklabels([])

    ax.tick_params(length=0)

    ax.grid(True)



def plot_task(task, num=0, dupl=True):

    """

    Plots the first train and test pairs of a specified task,

    using same color scheme as the ARC app

    """

    fig, ax = plt.subplots(1, 4, figsize=(15,15))

    

    plot_matrix(task['train'][num]['input'], ax[0])

    ax[0].set_title('Train Input')

    

    plot_matrix(task['train'][num]['output'], ax[1])

    ax[1].set_title('Train Output')

    

    plot_matrix(task['test'][0]['input'], ax[2])

    ax[2].set_title('Test Input')

    

    plot_matrix(task['test'][0]['output'], ax[3])

    ax[3].set_title('Test Output')

    plt.tight_layout()

    plt.show()



for i in range(4):



    task_file = str(training_path / training_tasks[i])



    with open(task_file, 'r') as f:

        task = json.load(f)

    plot_task(task)
different_y = 0

different_xy = 0

k = []

for i in range(len(training_tasks)):

    task_file = str(training_path / training_tasks[i])

    with open(task_file, 'r') as f:

        task = json.load(f)

    outs = [np.array(task['train'][i]['output']) for i in range(len(task['train']))]

    inps = [np.array(task['train'][i]['input']) for i in range(len(task['train']))]

    

    if len(set([x.shape for x in outs])) > 1:

        different_y += 1

        if any([x.shape != y.shape for x, y in zip(outs, inps)]):

            different_xy +=1

            k.append(i)

print(different_y, different_xy)

print(k)
task_file = str(training_path / training_tasks[376])

with open(task_file, 'r') as f:

    task = json.load(f)

plot_task(task)
def replace_values(a, d):

    return np.array([d.get(i, i) for i in range(a.min(), a.max() + 1)])[a - a.min()]



def repeat_matrix(a):

    return np.concatenate([a]*((SIZE // len(a)) + 1))[:SIZE]



def get_new_matrix(X):

    if len(set([np.array(x).shape for x in X])) > 1:

        X = np.array([X[0]])

    return X



def get_outp(outp, dictionary=None, replace=True):

    if replace:

        outp = replace_values(outp, dictionary)



    outp_matrix_dims = outp.shape

    outp_probs_len = outp.shape[0]*outp.shape[1]*10

    outp = to_categorical(outp.flatten(),

                          num_classes=10).flatten()



    return outp, outp_probs_len, outp_matrix_dims
class Mutator:

    def __init__(self):

        pass

    

    def mutate(self, X, y=None):

        return X, y



class OneOfMutator(Mutator):

    def __init__(self, mutators):

        self.mutators = mutators



    def mutate(self, X, y=None):

        return np.random.choice(self.mutators).mutate(X, y)

    

class MutationPipeline(Mutator):

    def __init__(self, mutators):

        self.mutators = mutators

    

    def mutate(self, X, y=None):

        for mutator in self.mutators:

            X, y = mutator.mutate(X, y)

        return X, y
class Shifter(Mutator):

    """ Class will shift whole picture to a random direction

    """

    def __init__(self, sigma):

        self.sigma = sigma

    

    def mutate(self, X, y=None):

        direction = np.random.randint(8)

        shift = np.random.randint(self.sigma)

        

        def do_shift(picture):

            shifted = np.zeros(picture.shape, dtype=int)

            if shift == 0:

                return picture

            if direction == 0:

                shifted[:,:-shift] = picture[:,shift:]

            if direction == 1:

                shifted[:,shift:] = picture[:,:-shift]

            if direction == 2:

                shifted[:-shift,:] = picture[shift:,:]

            if direction == 3:

                shifted[shift:,:] = picture[:-shift,:]

            if direction == 4:

                shifted[:-shift,:-shift] = picture[shift:,shift:]

            if direction == 5:

                shifted[:-shift,shift:] = picture[shift:,:-shift]

            if direction == 6:

                shifted[shift:,shift:] = picture[:-shift,:-shift]

            if direction == 7:

                shifted[shift:,:-shift] = picture[:-shift,shift:]

            return shifted

        

        return do_shift(X), (None if y is None else do_shift(y))

    

mutator = Shifter(2)



fig, ax = plt.subplots(1, 4, figsize=(15,15))

plot_matrix(np.array(Xs_train[0][0]), ax[0])

plot_matrix(np.array(ys_train[0][0]), ax[1])

mutated = mutator.mutate(np.array(Xs_train[0][0]), np.array(ys_train[0][0]))

plot_matrix(mutated[0], ax[2])

plot_matrix(mutated[1], ax[3])
class ColorSwitcher(Mutator):

    """ Class will shift whole picture to a random direction

    """

    def __init__(self, except_=[]):

        self.colors = []

        for i in range(10):

            if i not in except_:

                self.colors.append(i)

        

    

    def mutate(self, X, y=None):

        rep = np.array(self.colors)

        orig = np.array(self.colors.copy())

        np.random.shuffle(rep)

        dictionary = dict(zip(orig, rep))

        return replace_values(X, dictionary), (None if y is None else replace_values(y, dictionary))

    

mutator = ColorSwitcher([7])





fig, ax = plt.subplots(1, 4, figsize=(15,15))

plot_matrix(np.array(Xs_train[0][1]), ax[0])

plot_matrix(np.array(ys_train[0][1]), ax[1])

mutated = mutator.mutate(np.array(Xs_train[0][1]), np.array(ys_train[0][1]))

plot_matrix(mutated[0], ax[2])

plot_matrix(mutated[1], ax[3])
from scipy.ndimage.interpolation import rotate

class Rotator(Mutator):

    """ Class will shift whole picture to a random direction

    """

    def __init__(self, angles):

        self.angles = angles

    

    def mutate(self, X, y=None):

        g = np.random.randint(len(self.angles))

        angle = self.angles[g]

        return np.rot90(X, angle), (None if y is None else np.rot90(y, angle))

    

mutator = Rotator([0,1,2])



fig, ax = plt.subplots(1, 4, figsize=(15,15))

plot_matrix(np.array(X_train[0]), ax[0])

plot_matrix(np.array(y_train[0]), ax[1])

mutated = mutator.mutate(np.array(X_train[0]), np.array(y_train[0]))

plot_matrix(mutated[0], ax[2])

plot_matrix(mutated[1], ax[3])
class Flipper(Mutator):

    """ Class will shift whole picture to a random direction

    """

    def __init__(self, do_t=False):

        self.do_t = do_t

    

    def mutate(self, X, y=None):

        direction = np.random.randint(3-self.do_t)

        def do_mirror(picture):

            if direction == 0:

                return np.flipud(picture)

            if direction == 1:

                return np.fliplr(picture)

            if direction == 2:

                return picture.T

            return picture

        return do_mirror(X), (None if y is None else do_mirror(y))

    

mutator = Flipper(False)



fig, ax = plt.subplots(1, 4, figsize=(15,15))

plot_matrix(np.array(X_train[0]), ax[0])

plot_matrix(np.array(y_train[0]), ax[1])

mutated = mutator.mutate(np.array(X_train[0]), np.array(y_train[0]))

plot_matrix(mutated[0], ax[2])

plot_matrix(mutated[1], ax[3])
def get_borders(X):

    result = []

    max_ = np.max([np.array(x).shape for x in X])

    for x in X:

        x = np.array(x)

        extended = np.zeros((max_, max_))

        diffw = max_ - x.shape[0]

        diffh = max_ - x.shape[1]



        def get_borders_add(diff):

            addl = diff // 2

            if addl != diff / 2.0:

                addr = addl + 1

            else:

                addr = addl

            return addl, addr



        addl, addr = get_borders_add(diffw)

        addt, addb = get_borders_add(diffh)

        result.append((addl, max_ - addr, addt, max_ - addb))

    return result

    

def extend_matrices_to_max(X):

    if len(set([np.array(x).shape for x in X])) == 1:

        return X

    result = []

    borders = get_borders(X)

    max_ = np.max([np.array(x).shape for x in X])

    for i, x in enumerate(X):

        x = np.array(x)

        extended = np.zeros((max_, max_))

        extended[borders[i][0]:borders[i][1], borders[i][2]:borders[i][3]] = x[:, :]

        result.append(extended.astype(int))



    return np.array(result)



def narrow_prediction(prediction, orig_X, orig_y):

    different_y_sizes = False

    different_with_x_y = False

    if len(set([x.shape for x in orig_y])) > 1:

        different_y_sizes = True

        if any([x.shape != y.shape for x, y in zip(orig_y, orig_X)]):

            different_with_x_y = True

    if not different_with_x_y:

        borders = get_borders(orig_y)

        return np.array([y[borders[i][0]:borders[i][1], borders[i][2]:borders[i][3]] for i,y in enumerate(prediction)])



    result = []

    for y in prediction:

        for i in range(y.shape[0]):

            if np.sum(y[i]) > 1e-5:

                left = i

        for i in reversed(range(y.shape[0])):

            if np.sum(y[i]) > 1e-5:

                right = i + 1

        for i in range(y.shape[1]):

            if np.sum(y[:,i]) > 1e-5:

                bot = i

        for i in reversed(range(y.shape[1])):

            if np.sum(y[:,i]) > 1e-5:

                top = i + 1

        result.append(y[left:right, bot:top])

    return np.array(result)

        
class Extender:

    def __init__(self, X_train, X_test, y_train):

        self.X_train = np.array(X_train)

        self.X_test = np.array(X_test)

        self.y_train = np.array(y_train)

        

        def get_max_shape(X):

            shapes1 = [[len(x)] for x in X]

            shapes2 = [[len(x[0])] for x in X]

            return np.max((np.max(shapes1),np.max(shapes2)))

        

        max_train = get_max_shape(X_train)

        max_test = get_max_shape(X_test)



        max_ = np.max([max_train, max_test])

        self.X_train_borders = self.get_borders(self.X_train, max_)

        self.X_test_borders = self.get_borders(self.X_test, max_)



        self.different_y_sizes = False

        self.different_with_x_y = False

        if len(set([np.shape(x) for x in self.y_train])) > 1:

            self.different_y_sizes = True

            if any([np.shape(x) != np.shape(y) for x, y in zip(self.y_train, self.X_train)]):

                self.different_with_x_y = True

        

        

        #self.extended_X_train = np.kron(self.extend_matrices_to_max(self.X_train, max_, self.X_train_borders), np.ones((3,3)))

        #self.extended_X_test = np.kron(self.extend_matrices_to_max(self.X_test, max_, self.X_test_borders), np.ones((3,3)))

        self.extended_X_train = self.extend_matrices_to_max(self.X_train, max_, self.X_train_borders)

        self.extended_X_test = self.extend_matrices_to_max(self.X_test, max_, self.X_test_borders)

        if not self.different_with_x_y and self.different_y_sizes:

            self.y_train_borders = self.get_borders(self.y_train, max_)

            self.extended_y_train = self.extend_matrices_to_max(self.y_train, max_, self.y_train_borders)  

        else:

            max_y_train = get_max_shape(self.y_train)

            self.y_train_borders = self.get_borders(self.y_train, max_y_train)

            self.extended_y_train = self.extend_matrices_to_max(self.y_train, max_y_train, self.y_train_borders)        

        

    def get_borders(self, X, max_):

        result = []



        for x in X:

            x = np.array([np.array(k) for k in x])

            extended = np.zeros((max_, max_))

            diffw = max_ - x.shape[0]

            diffh = max_ - x.shape[1]



            def get_borders_add(diff):

                addl = diff // 2

                if addl != diff / 2.0:

                    addr = addl + 1

                else:

                    addr = addl

                return addl, addr



            addl, addr = get_borders_add(diffw)

            addt, addb = get_borders_add(diffh)

            result.append((addl, max_ - addr, addt, max_ - addb))

        return result

    

    def extend_matrices_to_max(self, X, max_, borders):

        shapes = [np.array(x).shape for x in X]

        result = []

        for i, x in enumerate(X):

            x = np.array([np.array(k,dtype=int) for k in x])

            extended = np.zeros((max_, max_), dtype=int)

            extended[borders[i][0]:borders[i][1], borders[i][2]:borders[i][3]] = x[:, :]



            result.append(extended)

        

        return np.array(result)



    def narrow_prediction(self, prediction, idxs):

        if not self.different_y_sizes:

            return prediction

        if not self.different_with_x_y:

            borders = self.X_test_borders

            return np.array([y[borders[i][0]:borders[i][1], borders[i][2]:borders[i][3]] for y,i in zip(prediction, idxs)])



        result = []

        for y in prediction:

            left, bot=0,0

            right, top = y.shape

            for i in range(y.shape[0]):

                if np.sum(y[i]) > 1e-5:

                    left = i

                    break

            for i in reversed(range(y.shape[0])):

                if np.sum(y[i]) > 1e-5:

                    right = i + 1

                    break

            

            for i in range(y.shape[1]):

                if np.sum(y[:,i]) > 1e-5:

                    bot = i

                    break

            for i in reversed(range(y.shape[1])):

                if np.sum(y[:,i]) > 1e-5:

                    top = i + 1

                    break

            result.append(y[left:right, bot:top])

        return np.array(result)

num = 5

ext = Extender(Xs_train[num], Xs_test[num], ys_train[num])

fig, ax = plt.subplots(1, 4, figsize=(15,15))

plot_matrix(np.array(Xs_train[num][0]), ax[0])

plot_matrix(np.array(ys_train[num][0]), ax[1])

plot_matrix(Xs_test[num][0], ax[2])

plot_matrix(ext.narrow_prediction([ext.extended_y_train[0]], [0])[0], ax[3])
ext.extended_y_train[0]
ext.extended_X_test[0]
class ARCDataset(Dataset):

    def __init__(self, X_train, X_test, y_train, mutation=ColorSwitcher()):

        self.mutation = mutation

        self.extender = Extender(X_train, X_test, y_train)

        self.X_train = repeat_matrix(self.extender.extended_X_train)

        self.y_train = repeat_matrix(self.extender.extended_y_train)

        self.colors = []

        for x in X_train:

            self.colors += np.unique(x).tolist()

        for x in X_test:

            self.colors += np.unique(x).tolist()

        for x in y_train:

            self.colors += np.unique(x).tolist()

        

    def __len__(self):

        return SIZE

    

    def get_input_dimension(self):

        return self.X_train[0].shape

        

    def get_output_dimension(self):

        return self.y_train[0].shape

    

    def get_output_prob_dimension(self):

        return self.y_train[0].shape



    def narrow_prediction(self, prediction, idxs):

        return self.extender.narrow_prediction(prediction, idxs)

        

    def get_extended_test(self):

        return self.extender.extended_X_test

    

    def round_colors(self, prediction):

        @np.vectorize

        def find_nearest(value):

            array = np.array(self.colors)

            idx = (np.abs(array - value)).argmin()

            return array[idx]

        return find_nearest(prediction)

    

    def __getitem__(self, idx):

        in_, out_ = self.mutation.mutate(self.X_train[idx], self.y_train[idx])



        return torch.FloatTensor(in_.copy()), torch.LongTensor(out_.copy()), self.get_output_prob_dimension()
# dataset = ARCDataset(Xs_train[0], Xs_test[0], ys_train[0],MutationPipeline([

#                                            ColorSwitcher(),

#                                            #Shifter(1),

#                                            #Flipper(),

#                                            #Shifter(1),

#                                            #Rotator([0, 2]),

#                                            #Shifter(1)

#                                        ]),)

# for i in range(50):

#     fig, ax = plt.subplots(1, 4, figsize=(15,15))

#     plot_matrix(np.array(Xs_train[0][0]), ax[0])

#     plot_matrix(np.array(ys_train[0][0]), ax[1])

#     mutated = mutator.mutate(np.array(Xs_train[0][0]), np.array(ys_train[0][0]))

#     ten = dataset[i]

#     plot_matrix(ten[0].detach(), ax[2])

#     plot_matrix(ten[1].detach(), ax[3])
import torch

import torch.nn as nn

import torch.nn.functional as F





class DoubleConv(nn.Module):

    """(convolution => [BN] => ReLU) * 2"""



    def __init__(self, in_channels, out_channels, mid_channels=None):

        super().__init__()

        if not mid_channels:

            mid_channels = out_channels

        self.double_conv = nn.Sequential(

            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),

            nn.BatchNorm2d(mid_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)

        )



    def forward(self, x):

        return self.double_conv(x)





class Down(nn.Module):

    """Downscaling with maxpool then double conv"""



    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.maxpool_conv = nn.Sequential(

            nn.MaxPool2d(2),

            DoubleConv(in_channels, out_channels)

        )



    def forward(self, x):

        return self.maxpool_conv(x)





class Up(nn.Module):

    """Upscaling then double conv"""



    def __init__(self, in_channels, out_channels, bilinear=True):

        super().__init__()



        # if bilinear, use the normal convolutions to reduce the number of channels

        if bilinear:

            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

        else:

            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)

            self.conv = DoubleConv(in_channels, out_channels)





    def forward(self, x1, x2):

        x1 = self.up(x1)

        # input is CHW

        diffY = x2.size()[2] - x1.size()[2]

        diffX = x2.size()[3] - x1.size()[3]



        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,

                        diffY // 2, diffY - diffY // 2])

        # if you have padding issues, see

        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a

        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)





class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(OutConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)



    def forward(self, x):

        return self.conv(x)

    

    def shape(self):

        self.conv.shape
""" Full assembly of the parts to form the complete network """



import torch.nn.functional as F





class UNet(nn.Module):

    def __init__(self, n_channels, n_classes, in_d, out_d, bilinear=True):

        super(UNet, self).__init__()

        self.n_channels = n_channels

        self.n_classes = n_classes

        self.bilinear = bilinear

        # self.start = Up(1, 1, bilinear)

        self.inc = DoubleConv(n_channels, 64)

        self.down1 = Down(64, 128)

        #self.down2 = Down(128, 256)

        #self.down3 = Down(256, 512)

        factor = 2 if bilinear else 1

        #self.down4 = Down(512, 1024 // factor)

        #self.up1 = Up(1024, 512 // factor, bilinear)

        #self.up2 = Up(768, 256 // factor, bilinear)

        #self.up3 = Up(384, 128 // factor, bilinear)

        self.up4 = Up(192, 64, bilinear)

        self.outc = OutConv(64, n_classes)

        self.conv = nn.Conv2d(n_classes, n_classes, kernel_size=np.abs(in_d[0]-out_d[0])+1, padding=-np.min((in_d[0]-out_d[0], 0)))



    def forward(self, x):

        # x1 = self.start(x)

        x1 = self.inc(x)

        x2 = self.down1(x1)

        #x3 = self.down2(x2)

        #x4 = self.down3(x3)

        #x = self.up2(x4, x3)

        #x = self.up3(x3, x2)

        x = self.up4(x2, x1)

        logits = self.outc(x)

        logits = self.conv(logits)

        

        return logits
import torch.nn as nn

class BasicCNNModel(nn.Module):

    def __init__(self, inp_dim=(10, 10), outp_dim=(10, 10)):

        super(BasicCNNModel, self).__init__()

        

        #self.begin = nn.Conv2d(1, 3, 1)

        

        self.network = UNet(1, 10, inp_dim, outp_dim)

        #self.network['classifier'][4] = nn.Conv2d(512, 10, kernel_size=(1, 1), stride=(1, 1))



    def forward(self, x):

        #x_ = self.begin.forward(x)

        x_ = self.network.forward(x)

        return x_

a=BasicCNNModel()

a.parameters
for i in [5]:

    task_file = str(training_path / training_tasks[i])

    with open(task_file, 'r') as f:

        task = json.load(f)

    plot_task(task)
def transform_dim(inp_dim, outp_dim, test_dim):

    return (test_dim[0]*outp_dim[0]/inp_dim[0],

            test_dim[1]*outp_dim[1]/inp_dim[1])



def resize(x, test_dim, inp_dim):

    if inp_dim == test_dim:

        return x

    else:

        return cv2.resize(flt(x), inp_dim,

                          interpolation=cv2.INTER_AREA)



def flt(x): return np.float32(x)

def npy(x): return x.cpu().detach().numpy()

def itg(x): return np.int32(np.round(x))
import torchvision.models as models



def train_model(model, train_loader, dataset, loss, optimizer, num_epochs, scheduler=None):    

    loss_history = []

    train_history = []

    val_history = []

    for epoch in range(num_epochs):

        model.train() # Enter train mode

        

        loss_accum = 0

        correct_samples = 0

        total_samples = 0

        for i_step, train_batch in enumerate(train_loader):

            x, y, prob_d = train_batch

            prediction = model(x.unsqueeze(1))

            # print(prediction)

            #_, indices = torch.max(prediction, 1)

            #print(indices)

            #print(y)

#             print(y)

            loss_value = loss(prediction, y)

            optimizer.zero_grad()

            loss_value.backward()

            

            #_, indices = torch.max(prediction, 1)

            #correct_samples += torch.sum(indices == y)

            #total_samples += y.shape[0]

            optimizer.step()

            

            loss_accum += loss_value



        ave_loss = loss_accum / (i_step if i_step else 1)

        #train_accuracy = float(correct_samples) / total_samples

        

        loss_history.append(float(ave_loss))

        #train_history.append(train_accuracy)



        if scheduler is not None:

            scheduler.step()



        #print("Average loss: %f" % (ave_loss))

        

        if len(loss_history) > 5 and np.mean(np.abs(np.array(loss_history[-3:]) - np.array(loss_history[-4:-1]))) < 10**(epoch//10)*1e-6:

            break



        

    return loss_history

        
mutators = [0 for _ in range(100)]

mutators[0] = MutationPipeline([

ColorSwitcher(),

#Shifter(1),

Flipper(False),

#Shifter(1),

Rotator([0, 2]),

#Shifter(1)

])

mutators[1] = MutationPipeline([

#ColorSwitcher(),

Shifter(1),

#Flipper(False),

Shifter(1),

#Rotator([0, 1, 2]),

Shifter(1)

])

mutators[2] = MutationPipeline([

#ColorSwitcher(),

Shifter(1),

Flipper(False),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[3] = MutationPipeline([

])

mutators[4] = MutationPipeline([

#ColorSwitcher(),

#Shifter(1),

Flipper(False),

#Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[5] = MutationPipeline([

ColorSwitcher(),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[6] = MutationPipeline([

ColorSwitcher(),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[7] = MutationPipeline([

])

mutators[8] = MutationPipeline([

])

mutators[9] = MutationPipeline([

ColorSwitcher(),

#Shifter(1),

Flipper(),

#Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[10] = MutationPipeline([

#ColorSwitcher(),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[11] = MutationPipeline([

ColorSwitcher(),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[12] = MutationPipeline([

ColorSwitcher(),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[13] = MutationPipeline([

#ColorSwitcher(),

Shifter(1),

#Flipper(),

#Shifter(1),

#Rotator([0, 1, 2]),

#Shifter(1)

])

mutators[14] = MutationPipeline([

ColorSwitcher(),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[15] = MutationPipeline([

])

mutators[16] = MutationPipeline([

ColorSwitcher(),

#Shifter(1),

Flipper(),

#Shifter(1),

Rotator([0, 1, 2]),

#Shifter(1)

])

mutators[17] = MutationPipeline([

])

mutators[18] = MutationPipeline([

])

mutators[19] = MutationPipeline([

])

mutators[20] = MutationPipeline([

])

mutators[21] = MutationPipeline([

ColorSwitcher(),

Shifter(1),

#Flipper(),

Shifter(1),

#Rotator([0, 1, 2]),

Shifter(1)

])

mutators[22] = MutationPipeline([

ColorSwitcher(),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[23] = MutationPipeline([

])

mutators[24] = MutationPipeline([

])

mutators[25] = MutationPipeline([

ColorSwitcher(),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[26] = MutationPipeline([

])

mutators[27] = MutationPipeline([

ColorSwitcher([5]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[28] = MutationPipeline([

#ColorSwitcher(),

#Shifter(1),

Flipper(),

#Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[29] = MutationPipeline([

ColorSwitcher([1,2,3]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[30] = MutationPipeline([

#ColorSwitcher(),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[31] = MutationPipeline([

ColorSwitcher([0,1,4]),

Flipper(),

Rotator([0, 1, 2]),

])

mutators[32] = MutationPipeline([

ColorSwitcher([2, 4]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[33] = MutationPipeline([

])

mutators[34] = MutationPipeline([

])

mutators[35] = MutationPipeline([

])

mutators[36] = MutationPipeline([

])

mutators[37] = MutationPipeline([

ColorSwitcher([0, 2, 4]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[38] = MutationPipeline([

ColorSwitcher([0]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[39] = MutationPipeline([

])

mutators[40] = MutationPipeline([

#ColorSwitcher([0]),

#Shifter(1),

Flipper(),

#Shifter(1),

Rotator([0, 1, 2]),

#Shifter(1)

])

mutators[41] = MutationPipeline([

ColorSwitcher([0, 8]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[42] = MutationPipeline([

])

mutators[43] = MutationPipeline([

])

mutators[44] = MutationPipeline([

ColorSwitcher([1, 0, 2, 5]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[45] = MutationPipeline([

ColorSwitcher([0, 5]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[46] = MutationPipeline([

])

mutators[47] = MutationPipeline([

ColorSwitcher([0]),

Shifter(1),

#Flipper(),

Shifter(1),

Rotator([0, 2]),

Shifter(1)

])

mutators[48] = MutationPipeline([

ColorSwitcher([0]),

#Shifter(1),

Flipper(),

#Shifter(1),

Rotator([0, 1, 2]),

#Shifter(1)

])

mutators[49] = MutationPipeline([

])

mutators[50] = MutationPipeline([

])

mutators[51] = MutationPipeline([

ColorSwitcher([0, 5]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[52] = MutationPipeline([

ColorSwitcher([0]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[53] = MutationPipeline([

ColorSwitcher([0]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 2]),

Shifter(1)

])

mutators[54] = MutationPipeline([

#ColorSwitcher([0, 5, 1,]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[55] = MutationPipeline([

])

mutators[56] = MutationPipeline([

#ColorSwitcher([0]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[57] = MutationPipeline([

])

mutators[58] = MutationPipeline([

#ColorSwitcher([0]),

Shifter(1),

Flipper(),

Shifter(2),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[59] = MutationPipeline([

#ColorSwitcher([0]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 2]),

Shifter(1)

])

mutators[60] = MutationPipeline([

ColorSwitcher([4]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[61] = MutationPipeline([

ColorSwitcher([0]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[62] = MutationPipeline([

])

mutators[63] = MutationPipeline([

])

mutators[64] = MutationPipeline([

])

mutators[65] = MutationPipeline([

])

mutators[66] = MutationPipeline([

ColorSwitcher([0,5]),

Shifter(1),

#Flipper(),

Shifter(1),

#Rotator([0, 1, 2]),

Shifter(1)

])

mutators[67] = MutationPipeline([

])

mutators[68] = MutationPipeline([

ColorSwitcher([0,3]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[69] = MutationPipeline([

ColorSwitcher([1,2]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[70] = MutationPipeline([

])

mutators[71] = MutationPipeline([

ColorSwitcher([0, 1]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[72] = MutationPipeline([

])

mutators[73] = MutationPipeline([

])

mutators[74] = MutationPipeline([

ColorSwitcher([0, 5]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[75] = MutationPipeline([])

mutators[76] = MutationPipeline([])

mutators[77] = MutationPipeline([

ColorSwitcher([0]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[78] = MutationPipeline([

ColorSwitcher([0, 2]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[79] = MutationPipeline([

ColorSwitcher([0, 5, 1]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[80] = MutationPipeline([

ColorSwitcher([0, 9, 4]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[81] = MutationPipeline([

ColorSwitcher([0, 5]),

Shifter(1),

#Flipper(),

Shifter(1),

#Rotator([0, 1, 2]),

Shifter(1)

])

mutators[82] = MutationPipeline([

#ColorSwitcher([0, 5]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[83] = MutationPipeline([

])

mutators[84] = MutationPipeline([

ColorSwitcher([0, 5]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 1, 2]),

Shifter(1)

])

mutators[85] = MutationPipeline([

])

mutators[86] = MutationPipeline([

])

mutators[87] = MutationPipeline([

ColorSwitcher([0, 5]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 2]),

Shifter(1)

])

mutators[88] = MutationPipeline([

ColorSwitcher([0]),

Flipper(),

Shifter(1),

Rotator([0, 2]),

Shifter(4)

])

mutators[89] = MutationPipeline([

])

mutators[90] = MutationPipeline([

ColorSwitcher()

])

mutators[91] = MutationPipeline([

ColorSwitcher([0]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 2]),

Shifter(1)

])

mutators[92] = MutationPipeline([

#ColorSwitcher([0]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 2]),

Shifter(1)

])

mutators[93] = MutationPipeline([

])

mutators[94] = MutationPipeline([

ColorSwitcher([0,3]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 2]),

Shifter(1)

])

mutators[95] = MutationPipeline([

ColorSwitcher([0]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 2]),

Shifter(1)

])

mutators[96] = MutationPipeline([

])

mutators[97] = MutationPipeline([

ColorSwitcher([0]),

Shifter(1),

Flipper(),

Shifter(1),

Rotator([0, 2]),

Shifter(1)

])

mutators[98] = MutationPipeline([

])

mutators[99] = MutationPipeline([

])
idx = 0

start = time.time()

test_predictions = []

class Flattener(nn.Module):

    def forward(self, x):

        batch_size, *_ = x.shape

        return x.view(batch_size, -1)



RESULT = {}

    

def do_train(dataset, do_test=False):

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)



    in_d = dataset.get_input_dimension()

    prob_d = dataset.get_output_prob_dimension()

    d = dataset.get_output_dimension()

    network = BasicCNNModel(in_d, d)

    # network = nn.Sequential(

    #   nn.Conv2d(1, 256, np.min((FL, in_d[0])), padding=1),

    #   nn.ReLU(inplace=True),

    #   nn.MaxPool2d(4),

    #   nn.BatchNorm2d(256),

    #   nn.Conv2d(64, 10, 3, padding=1)

    # )

    loss = nn.CrossEntropyLoss()



    optimizer = Adam(network.parameters(), lr=1e-2, weight_decay=1e-3)

    step_lr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.05)

    lh = train_model(network, train_loader, dataset, loss, optimizer, EPOCHS)

    preds = []

    if do_test:

        test_loader = DataLoader(np.array(dataset.extender.extended_X_test, dtype=np.single), batch_size=1, shuffle=False)

        network.eval()

        for i, test_batch in enumerate(test_loader):

            test_batch = test_batch

            test_preds = network(torch.FloatTensor(test_batch).unsqueeze(1))

            _, indices = torch.max(test_preds, 1)

            test_preds = dataset.narrow_prediction(indices.detach().numpy(), [i])

            preds.append(dataset.round_colors(test_preds)[0])

    return lh[-1], preds



for idx, (X_train, y_train) in enumerate(zip(Xs_train, ys_train)):

    X_test = Xs_test[idx]

    print("TASK " + str(idx + 1))

    mutator_losses = {}

    losses = []

    EPOCHS = 65

    if len(mutators[idx].mutators) != 0:

        dataset = ARCDataset(X_train, X_test, y_train, mutators[idx])

        train_loss, predictions = do_train(dataset, do_test=True)

    else:

        predictions = X_test

    for test_num, pred in enumerate(predictions):

        RESULT["{}_{}".format(task_ids[idx], test_num)] = np.array(pred).astype(int).tolist()

    end = time.time()

    print("Train loss: " + str(np.round(train_loss, 3)) + "   " +\

          "Total time: " + str(np.round(end - start, 1)) + " s" + "\n")

    fig, ax = plt.subplots(1, 4, figsize=(15,15))

    plot_matrix(np.array(X_train[0]), ax[0])

    plot_matrix(np.array(y_train[0]), ax[1])

    plot_matrix(X_test[0], ax[2])

    plot_matrix(RESULT["{}_{}".format(task_ids[idx], test_num)], ax[3])
RESULT["{}_{}".format(task_ids[idx], test_num)]
def flattener(pred):

    str_pred = str([row for row in pred])

    str_pred = str_pred.replace(', ', '')

    str_pred = str_pred.replace('[[', '|')

    str_pred = str_pred.replace('][', '|')

    str_pred = str_pred.replace(']]', '|')

    return str_pred
RESULT["00576224_0"]
flattener(RESULT["00576224_0"])
#test_predictions = [[list(pred) for pred in test_pred] for test_pred in test_predictions]

rr = {}

for id_ in RESULT:

    rr[id_] = flattener(RESULT[id_])

    

#submission = pd.read_csv(SUBMISSION_PATH)

#submission["output"] = test_predictions
rr
submissions = pd.Series(rr, name='output')

submissions.index.name = 'output_id'

submissions.reset_index()

submissions = pd.DataFrame(submissions)

submissions
submissions.to_csv("submission.csv")