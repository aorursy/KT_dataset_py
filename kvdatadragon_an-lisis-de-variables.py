#calculo

import numpy as np

import pandas as pd



#grafico

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import display

%matplotlib inline

sns.set(style="whitegrid")



#warning ignore future

import warnings

# warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.filterwarnings("ignore")



import os

print(os.listdir("../input"))
set_parameter_csv = {

    'sep': ',',

    'encoding': 'ISO-8859-1',

    'low_memory': False

}

train = pd.read_csv('../input/Train_AdquisicionAhorro.csv', **set_parameter_csv)

display(train.head())

test = pd.read_csv('../input/Test_AdquisicionAhorro.csv', **set_parameter_csv)

display(test.head())

sub = pd.read_csv('../input/Submmit_AdquisicionAhorro.csv', **set_parameter_csv)

display(sub.head())
train.isnull().sum()
test.isnull().sum()
percentiles = [.1, .25, .5, .75, .9]

train.describe(percentiles=percentiles).T
test.describe(percentiles=percentiles).T
train.describe(include=['object', 'bool']).T
test.describe(include=['object', 'bool']).T
## VISUALIZANDO
def view_cat(data, col_init, col_out, **kwargs):

    color_label = kwargs.get('color_label', 'black')

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

    

    cross = pd.crosstab(data[col_out], data[col_init])

    sum_total = sum([cross[col].sum() for col in cross.columns])

    sns.heatmap(

        cross/sum_total, 

        annot=True, ax=axes[0], center=0, cmap="YlGnBu", fmt='.2%'

    )

    sns.barplot(

        x=col_init, y=col_out, data=data, ax=axes[1]

    )



def view_numeric(data, col_init, col_out, **kwargs):

    color_label = kwargs.get('color_label', 'black')

    bins = kwargs.get('bins', 3)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))



    sns.violinplot(x=col_out, y=col_init, data=data, ax=axes[0])

    sns.distplot(data[col_init], ax=axes[1])
cols_str = list(test.describe(include=['object', 'bool']).columns[1:])

cols_str
for col in cols_str:

    view_cat(data=train, col_init=col, col_out='Adq_Ahorro')
cols_num = list(test.describe().columns)

cols_num
for col in cols_num:

    view_numeric(data=train, col_init=col, col_out='Adq_Ahorro')