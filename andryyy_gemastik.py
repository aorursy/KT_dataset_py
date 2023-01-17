# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import json

import zipfile

import urllib.request

import itertools

from sklearn import metrics

from sklearn import linear_model

import warnings

warnings.filterwarnings('ignore')

import networkx as nx

from pandas import *

import seaborn as sns

from matplotlib import rcParams

import datetime as dt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_sample = pd.read_csv("/kaggle/input/GEMASTIK.csv")

##print(data_sample)

data_sample.head()
try1 = data_sample.sort_values(['Tahun','Kota'], ascending=[True,True])[(data_sample.JenisBencana == "BANJIR")&(data_sample.Tahun >= 2015)][['Kota','JenisBencana','Tahun','Jumlah']].groupby(['Tahun'])

##try1.to_csv("out.csv", index=False)

try1.plot(

    x='Kota',

    y='Jumlah',

    kind='barh',

    figsize=(20, 7),

    title='Grafik Data Banjir 2015-2019')
try1 = data_sample.sort_values(['Tahun','Kota'], ascending=[True,True])[(data_sample.JenisBencana == "PUTING BELIUNG")&(data_sample.Tahun >= 2015)][['Kota','JenisBencana','Tahun','Jumlah']].groupby(['Tahun'])

##try1.to_csv("out.csv", index=False)

try1.plot(

    x='Kota',

    y='Jumlah',

    kind='barh',

    figsize=(20, 7),

    title='Grafik Data Puting Beliung 2015-2019')
try1 = data_sample.sort_values(['Tahun','Kota'], ascending=[True,True])[(data_sample.JenisBencana == "TANAH LONGSOR")&(data_sample.Tahun >= 2015)][['Kota','JenisBencana','Tahun','Jumlah']].groupby(['Tahun'])

##try1.to_csv("out.csv", index=False)

try1.plot(

    x='Kota',

    y='Jumlah',

    kind='barh',

    figsize=(20, 7),

    title='Grafik Data Tanah Longsor 2015-2019')
try1 = data_sample.sort_values(['Tahun','Kota'], ascending=[True,True])[(data_sample.JenisBencana == "TANAH LONGSOR")&(data_sample.Tahun >= 2015)][['Kota','JenisBencana','Tahun','Jumlah']].groupby(['Tahun'])

##try1.to_csv("out.csv", index=False)

try1.plot(

    x='Kota',

    y='Jumlah',

    kind='barh',

    figsize=(20, 7),

    title='Grafik Data Tanah Longsor 2015-2019')
try1 = data_sample.sort_values(['Tahun','Kota'], ascending=[True,True])[(data_sample.Tahun == 2018)][['Kota','JenisBencana','Tahun','Jumlah']].groupby(['Kota'])

##try1.to_csv("out.csv", index=False)

try1.plot(

    x='JenisBencana',

    y='Jumlah',

    kind='barh',

    figsize=(20, 7),

    title='Grafik Data Bencana Seluruh Kabupaten 2018')