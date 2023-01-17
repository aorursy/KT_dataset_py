import numpy as np

import pandas as pd

from IPython.display import display

import random

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

import time



pd.options.display.max_columns = None
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

test = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')



train['dataset'] = 'train'

test['dataset'] = 'test'



df = pd.concat([train, test])
train.head(10)
test.head(10)
print('Training Set - Number of rows {} and Number of columns {}: '.format(train.shape[0], train.shape[1]-1))

print('Teseting Set - Number of row {} and Number of columns {}'.format(test.shape[0], test.shape[1]-1))
ds = df[df['dataset'] == 'train']

ds = ds.groupby(['cp_dose', 'cp_type', 'cp_time'])['sig_id'].count().reset_index()

ds.columns = ['cp_dose', 'cp_type', 'cp_time', 'count']



fig = px.sunburst(ds, path = ['cp_dose', 'cp_type', 'cp_time'], values = 'count',

                 title = "Sunburst chart for cp_dose, cp_type, cp_time", width = 600, height = 600

                 )



fig.show()
train_columns = train.columns.to_list()

g_list = [i for i in train_columns if i.startswith('g-')]

c_list = [i for i in train_columns if i.startswith('c-')]



columns = g_list + c_list



correlation_var = random.choices(columns, k=40)

data = df[correlation_var]



f = plt.figure(figsize=(20,18))

plt.matshow(data.corr(), fignum = f.number)

plt.xticks(range(data.shape[1]), data.columns, fontsize=15, rotation=50)

plt.yticks(range(data.shape[1]), data.columns, fontsize = 15)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=15)

cols = ['cp_time'] + columns

all_columns = []



for i in range(0, len(cols)):

    for j in range(i + 1, len(cols)):

        if abs(train[cols[i]].corr(train[cols[j]])) > 0.9:

            all_columns.append(cols[i])

            all_columns.append(cols[j])

all_columns = list(set(all_columns))

data = df[all_columns]



f = plt.figure(figsize = (20,18))

plt.matshow(data.corr(), fignum= f.number)

plt.xticks(range(data.shape[1]), data.columns, fontsize = 15, rotation = 50)

plt.yticks(range(data.shape[1]), data.columns, fontsize=15)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=15)
import seaborn as sns

from colorama import Fore, Back, Style

y_ = Fore.YELLOW

r_ = Fore.RED

g_ = Fore.GREEN

b_ = Fore.BLUE

m_ = Fore.MAGENTA

sr_ = Style.RESET_ALL



def distribution(feature, color):

    plt.figure(figsize = (16,8))

    plt.subplot(121)

    sns.distplot(train[feature], color = color)

    plt.subplot(122)

    sns.violinplot(train[feature])

    print("{}Max value of {} is: {} {:.2f} \n{}Min value of {} is: {} {:.2f}\n{}Mean of {} is: {}{:.2f}\n{}Standard Deviation of {} is:{}{:.2f}"\

      .format(y_,feature,r_,train[feature].max(),g_,feature,r_,train[feature].min(),b_,feature,r_,train[feature].mean(),m_,feature,r_,train[feature].std()))
distribution("g-1","blue")
train['g_mean'] = train[[x for x in train.columns if x.startswith("g-")]].mean(axis = 1)

test['g_mean'] = test[[x for x in test.columns if x.startswith("g-")]].mean(axis = 1)

distribution("g_mean", "yellow")