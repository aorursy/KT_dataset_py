# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/preferred-ticket-payment-method-in-buses/dataset.csv')
data.head(5)
list_of_columns = []

for column in data.columns:

    list_of_columns.append(column)
list_of_columns
f, axes = plt.subplots(1,5,figsize=(18,5))

x = ['Ethnicity','Gender','AgeGroup','Employment','ModeOfPayment']

    

ax = 0

for x in x:

    sns.countplot(x=x,data=data,ax = axes[ax])

    ax = ax + 1
f, axes = plt.subplots(1,4,figsize=(18,5))

x = ['Ethnicity','Gender','AgeGroup','Employment']

hue = 'ModeOfPayment'    

ax = 0

for x in x:

    sns.countplot(x=x,hue=hue,data=data,ax = axes[ax])

    ax = ax + 1