import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pickle

from matplotlib import pyplot as plt

import seaborn as sns

sns.set()
with open(r"../input/ich-training/DENSE_BSB_100_10.history","rb") as f:

    i_bsb = pd.DataFrame(pickle.load(f))

with open(r"../input/ich-training/DENSE_GRADIENT_100_10.history","rb") as f:

    i_grad = pd.DataFrame(pickle.load(f))

with open(r"../input/ich-training/DENSE_SIGMOID_BSB_100_10.history","rb") as f:

    i_sigmoid = pd.DataFrame(pickle.load(f))

with open(r"../input/ich-training/DENSE_WINDOW_100_10.history","rb") as f:

    i_window = pd.DataFrame(pickle.load(f))

with open(r"../input/dense-sigmoid-history/DENSE_SIGMOID_100_10.history","rb") as f:

    i_sig = pd.DataFrame(pickle.load(f))

df = pd.DataFrame()

df[['bsb_loss','bsb_val_loss']] = i_bsb[['loss','val_loss']]

df[['grad_loss','grad_val_loss']] = i_grad[['loss','val_loss']]

df[['sigmoid_bsb_loss','sigmoid_bsb_val_loss']] = i_sigmoid[['loss','val_loss']]

df[['win_loss','win_val_loss']] = i_window[['loss','val_loss']]

df[['sig_loss','sig_val_loss']] = i_sig[['loss','val_loss']]
df
f, ax = plt.subplots(figsize=(10,10))

#colours = ['red','red','blue','blue','green','green','purple','purple']

#customPalette = sns.set_palette(sns.color_palette(colours))

sns.lineplot(data=df[['bsb_loss','win_loss','grad_loss','sigmoid_bsb_loss','sig_loss']],ax=ax,dashes=None,palette='bright')

#ax.lines[1].set_linestyle("--")

#ax.lines[3].set_linestyle("--")

#ax.lines[5].set_linestyle("--")

#ax.lines[7].set_linestyle("--")

ax.set_xlabel("Epoch #")

ax.set_ylabel("Loss")

ax.set_title("Preprocessing functions loss")

ax.legend(['BSB','Window','Gradient','Sigmoid BSB','Sigmoid'])
f, ax = plt.subplots(figsize=(10,10))

#colours = ['red','red','blue','blue','green','green','purple','purple']

#customPalette = sns.set_palette(sns.color_palette(colours))

sns.lineplot(data=df[['bsb_val_loss','win_val_loss','grad_val_loss','sigmoid_bsb_val_loss','sig_val_loss']],ax=ax,dashes=None,palette='bright')

ax.lines[0].set_linestyle("--")

ax.lines[1].set_linestyle("--")

ax.lines[2].set_linestyle("--")

ax.lines[3].set_linestyle("--")

ax.lines[4].set_linestyle("--")

ax.set_xlabel("Epoch #")

ax.set_ylabel("Validation Loss")

ax.set_title("Preprocessing functions validation loss")

ax.legend(['BSB','Window','Gradient','Sigmoid BSB','Sigmoid'])
with open(r"../input/ich-training-backbones/DENSE_SIGMOID_BSB_200_15.history","rb") as f:

    dense = pd.DataFrame(pickle.load(f))

with open(r"../input/ich-training-backbones/INCEPT_SIGMOID_BSB_200_15.history","rb") as f:

    incept = pd.DataFrame(pickle.load(f))

with open(r"../input/ich-training-backbones/RESNET_SIGMOID_BSB_200_15.history","rb") as f:

    resnet = pd.DataFrame(pickle.load(f))
losses = pd.DataFrame()

losses['dense_loss'] = dense['loss']

losses['resnet_loss'] = resnet['loss']

losses['incept_loss'] = incept['loss']

losses['dense_val_loss'] = dense['val_loss']

losses['resnet_val_loss'] = resnet['val_loss']

losses['incept_val_loss'] = incept['val_loss']
from matplotlib.ticker import FormatStrFormatter

yticks = np.arange(130,185,5)/100

f,ax = plt.subplots(1,3,figsize=(16,5),constrained_layout=True,sharey=True)

#plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=None)

#f.tight_layout()

for a in ax:

    a.set_xlabel('Epoch #')

    a.set_ylabel('Loss')

    a.set_xticks(range(0,16,2))

    a.set_xticklabels(range(1,16,2))

    a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))





sns.lineplot(data=losses[['dense_loss','dense_val_loss']],ax=ax[0],palette="bright")

ax[0].legend(['Loss','Validation Loss'])

ax[0].set_title("DenseNet")

sns.lineplot(data=losses[['resnet_loss','resnet_val_loss']],ax=ax[1],palette="bright")

ax[1].legend(['Loss','Validation Loss'])

ax[1].set_title("ResNet")

sns.lineplot(data=losses[['incept_loss','incept_val_loss']],ax=ax[2],palette="bright")

ax[2].legend(['Loss','Validation Loss'])

ax[2].set_title("InceptionNet")

fig, ax = plt.subplots(figsize=(15,15))

colours = ['red','blue','purple','red','blue','purple']

customPalette = sns.set_palette(sns.color_palette(colours))

sns.lineplot(ax=ax,data=losses,palette=customPalette,dashes=None,markers=True)

ax.lines[3].set_linestyle("--")

ax.lines[4].set_linestyle("--")

ax.lines[5].set_linestyle("--")
res = [237,244,239,237,207,205,230,204,231,205,205,231,206,231,231]

res_m = [np.mean(res)]*len(res)

den = [269,269,209,268,272,206,208,276,217,275,210,268,273,206,265]

den_m = [np.mean(den)]*len(den)

inc = [241,245,242,243,243,198,243,242,198,241,197,241,197,239,241]

inc_m = [np.mean(inc)]*len(inc)

x = range(1,16)

times = pd.DataFrame({

    'x':x,

    'res':res,

    'den':den,

    'inc':inc,

    'r_m':res_m,

    'd_m':den_m,

    'i_m':inc_m

})

f,ax = plt.subplots(figsize=(15,5))

sns.lineplot(data=times,dashes=None,palette="bright",markers=False)

#ax.set_yticks(range(150,300,10))

ax.legend(['ResNet','DenseNet','InceptionNet'])

ax.lines[3].set_color(ax.lines[0].get_color())

ax.lines[3].set_linestyle("--")



ax.lines[4].set_color(ax.lines[1].get_color())

ax.lines[4].set_linestyle("--")



ax.lines[5].set_color(ax.lines[2].get_color())

ax.lines[5].set_linestyle("--")

ax.set_xlabel("Epoch #")

ax.set_ylabel("Time (ms)")

ax.set_title("Training time per epoch with mean lines")