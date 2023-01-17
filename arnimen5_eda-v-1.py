import pandas as pd

import numpy as np

from collections import Counter

import plotly.express as px

from plotly import graph_objs as go

import seaborn as sns



import matplotlib.pyplot as plt

from tqdm.notebook import tqdm 

import cv2

import pydicom as dicom

import tqdm

import plotly.express as px

from colorama import Fore, Back, Style

y_ = Fore.GREEN

r_ = Fore.YELLOW

g_ = Fore.CYAN

b_ = Fore.BLUE

m_ = Fore.MAGENTA

sr_ = Style.RESET_ALL
PATH = '../input/lish-moa/'
train_df = pd.read_csv('{}train_features.csv'.format(PATH))

train_targets_nonscored = pd.read_csv('{}train_targets_nonscored.csv'.format(PATH))

train_targets_scored = pd.read_csv('{}train_targets_scored.csv'.format(PATH))

test_df = pd.read_csv('{}test_features.csv'.format(PATH))

sample = pd.read_csv('{}sample_submission.csv'.format(PATH))
print(f"{y_}Number of rows in train data: {r_}{train_df.shape[0]}\n{y_}Number of columns in train data: {r_}{train_df.shape[1]}")

print(f"{g_}Number of rows in test data: {r_}{test_df.shape[0]}\n{g_}Number of columns in test data: {r_}{test_df.shape[1]}")

print(f"{b_}Number of rows in submission data: {r_}{sample.shape[0]}\n{b_}Number of columns in submission data:{r_}{sample.shape[1]}")
sample.head().style.applymap(lambda x: 'background-color:khaki')
test_df.head().style.applymap(lambda x: 'background-color:wheat')
train_df.head().style.applymap(lambda x: 'background-color:LightSalmon')
train_targets_nonscored.head().style.applymap(lambda x: 'background-color:Tomato')
train_targets_scored.head().style.applymap(lambda x: 'background-color:Plum')
top = Counter([ i for i in train_df['cp_dose']])

temp = pd.DataFrame(top.most_common(10))

temp.columns = ['cp_dose','count']

temp.style.background_gradient(cmap='Reds')
top = Counter([ i for i in train_df['cp_type']])

temp = pd.DataFrame(top.most_common(10))

temp.columns = ['cp_type','count']

temp.style.background_gradient(cmap='Greens')
train_df['dataset'] = 'train'

test_df['dataset'] = 'test'

df = pd.concat([train_df, test_df])

ds = df.groupby(['cp_type', 'dataset'])['sig_id'].count().reset_index()

ds.columns = ['cp_type', 'dataset', 'count']

fig = px.bar(

    ds, 

    x='cp_type', 

    y="count", 

    color = 'dataset',

    barmode='group',

    orientation='v', 

    title='cp_type in train/test counts', 

    width=600,

    height=500

)

fig.show()
train_df['dataset'] = 'train'

test_df['dataset'] = 'test'

df = pd.concat([train_df, test_df])

ds = df.groupby(['cp_dose', 'dataset'])['sig_id'].count().reset_index()

ds.columns = ['cp_dose', 'dataset', 'count']

fig = px.bar(

    ds, 

    x='cp_dose', 

    y="count", 

    color = 'dataset',

    barmode='group',

    orientation='v', 

    title='cp_dose in train/test counts', 

    width=600,

    height=500

)

fig.show()
ds = df.groupby(['cp_time', 'dataset'])['sig_id'].count().reset_index()

ds.columns = ['cp_time', 'dataset', 'count']

fig = px.bar(

    ds, 

    x='cp_time', 

    y="count", 

    color = 'dataset',

    barmode='group',

    orientation='v', 

    title='cp_time in train/test counts', 

    width=600,

    height=500

)

fig.show()