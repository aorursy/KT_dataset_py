%matplotlib inline
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import ast
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
def f2cat(filename: str) -> str:
    return filename.split('.')[0]

class Simplified():
    def __init__(self, input_path='./input'):
        self.input_path = input_path

    def list_all_categories(self):
        files = os.listdir(os.path.join(self.input_path, 'train_simplified'))
        return [f2cat(f) for f in files]

    def read_training_csv(self, category, nrows=None, usecols=None, drawing_transform=False):
        df = pd.read_csv(os.path.join(self.input_path, 'train_simplified', category + '.csv'),
                         nrows=nrows, parse_dates=['timestamp'], usecols=usecols)
        if drawing_transform:
            df['drawing'] = df['drawing'].apply(ast.literal_eval)
        return df
s = Simplified('../input/')
categories = s.list_all_categories()
len(categories)
df = s.read_training_csv('owl', nrows=100, drawing_transform=True)
df.head()
df.shape
n = 10
fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(16, 10))
for i, row in df[: n * n].iterrows():
    ax = axs[i // n, i % n]
    for x, y in row.drawing:
        color = 'green' if row.recognized else 'red'
        ax.plot(x, -np.array(y), lw=3, color=color)
    ax.axis('off')
plt.suptitle('Recognized and unrecognized owls')
plt.show();
def plot_category_samples(df, category, n=10):
    fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(16, 10))
    for i, row in df[: n * n].iterrows():
        ax = axs[i // n, i % n]
        for x, y in row.drawing:
            color = 'green' if row.recognized else 'red'
            ax.plot(x, -np.array(y), lw=3, color=color)
        ax.axis('off')
    plt.suptitle(category)
    fig.savefig('{}.png'.format(category), dpi=100)
    plt.close('all')
    plt.gcf()
animals = [
    'ant', 'bat', 'bear', 'bee', 'bird', 'butterfly', 'camel', 'cat', 'cow', 'crab', 'crocodile', 'dog',
    'dolphin', 'dragon', 'duck', 'elephant', 'fish', 'flamingo', 'frog', 'giraffe', 'hedgehog', 'horse',
    'kangaroo', 'lion', 'lobster', 'monkey', 'mosquito', 'mouse', 'octopus', 'owl', 'panda', 'parrot', 'penguin',
    'pig', 'rabbit', 'raccoon', 'rhinoceros', 'scorpion', 'sea turtle', 'shark', 'sheep', 'snail', 'snake',
    'spider', 'squirrel', 'swan', 'teddy-bear', 'tiger', 'whale', 'zebra'
]
for animal in tqdm(animals):
    df = s.read_training_csv(animal, nrows=100, drawing_transform=True)
    plot_category_samples(df, animal, n=10)