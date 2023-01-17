# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_time=pd.read_csv('../input/2004-2019.tsv',sep='\t')

cols=list(data_time.columns)

print(data_time.shape)

for col in cols:

    print(data_time[col].count())



    
for col in cols:

    print(col)

print(data_time.head())
!pip install googletrans
import googletrans

from googletrans import Translator

translator=Translator()
cols_en=[]

for col in cols:

    translation=translator.translate(col)

    text_col=translation.text

    cols_en.append(text_col)

print(cols_en)    
data_time.columns=cols_en

data_time=data_time.drop('Unnamed: 0',axis=1)

print(data_time['INITIAL DATE'].unique().tolist())

print(data_time['DATA FINAL'].unique().tolist())

print(data_time['UNIT OF MEASUREMENT'].unique().tolist())
regions=data_time['REGION'].unique().tolist()

states=data_time['STATE'].unique().tolist()

print(len(regions))

print(len(states))
def columns_describer(column,data=data_time):

    col_list=data[column].unique().tolist()

    print('for column',column,'lengths of uniques are:',len(col_list))
for col in list(data_time.columns):

    columns_describer(col)
cols=list(data_time.columns)

plt.hist(data_time[cols[1]].tolist())