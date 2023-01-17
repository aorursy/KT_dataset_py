# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
a = pd.read_csv("../input/kiva-augmented-data/kivaData_augmented.csv")
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

a.sample(5)
a.shape
plt.style.use('seaborn')

sns.set_style('whitegrid')

plt.figure(figsize=(15, 3))

a.isnull().mean().sort_values(ascending=False).plot.bar(color='black', alpha=0.5)

plt.axhline(y=0.2, color='r', linestyle='-')

plt.title('Missing values average per column', fontsize=25, weight='bold' )

plt.show()
b=a.dropna(thresh=len(a)*0.8, axis=1)

print('We dropped ',a.shape[1]-b.shape[1], ' features with missing values')
#Categorical missing values

NAcols=b.columns

for col in NAcols:

    if b[col].dtype == "object":

        b[col] = b[col].fillna("None")
#Numerical missing values

for col in NAcols:

    if b[col].dtype != "object":

        b[col]= b[col].fillna(0)
def label_rice (row):

    if row['Temperature'] < 20 and row['precipitation'] < 175 :

        return 'low conditions'

    if row['Temperature'] >= 20 and row['Temperature'] < 27 :

        return 'Moderate conditions'

    if row['Temperature'] >= 20 and row['Temperature'] < 27 and row['precipitation'] >= 175 and row['precipitation'] < 300 :

        return 'Good conditions'

    if row['Temperature'] < 20 and row['precipitation'] < 175 :

        return 'Bad conditions'

   

    return 'Other'



# we add those new columns to the existing  dataset:

b['label_rice'] = b.apply (lambda row: label_rice(row), axis=1)


def label_wheat (row):

    if row['Temperature'] < 21 and row['precipitation'] < 31 :

        return 'low conditions'

    if row['Temperature'] >= 21 and row['Temperature'] < 24 :

        return 'Moderate conditions'

    if row['Temperature'] >= 21 and row['Temperature'] < 24 and row['precipitation'] >= 31 and row['precipitation'] < 38 :

        return 'Good conditions'



    if row['Temperature'] > 24 and row['precipitation'] < 38 :

        return 'High conditions'

   

    return 'Other'



# we add those new columns to the existing  dataset:

b['label_wheat'] = b.apply (lambda row: label_wheat(row), axis=1)


def label_corn (row):

    if row['Temperature'] < 15 and row['precipitation'] < 60 :

        return 'low T+P conditions'

    if row['Temperature'] >= 15 and row['Temperature'] < 18 :

        return 'Moderate conditions'

    if row['Temperature'] >= 15 and row['Temperature'] < 18 and row['precipitation'] >= 60 and row['precipitation'] < 110 :

        return 'Good conditions'

    if row['Temperature'] > 18 and row['precipitation'] < 110 :

        return 'High T+P conditions'

    

    return 'Other'



# we add those new columns to the existing dataset:

b['label_corn'] = b.apply (lambda row: label_corn(row), axis=1)


def label_corn2 (row):

    if row['Temperature'] < 15 and row['precipitation'] < 60 :

        return '1'

    if row['Temperature'] >= 15 and row['Temperature'] < 18 :

        return '2'

    if row['Temperature'] >= 15 and row['Temperature'] < 18 and row['precipitation'] >= 60 and row['precipitation'] < 110 :

        return '3'

    if row['Temperature'] > 18 and row['precipitation'] < 110 :

        return '4'

    

    return '5'



# we add those new columns to the existing dataset:

b['label_corn2'] = b.apply (lambda row: label_corn2(row), axis=1)
b.head()
def plot_label(label):

    plot= sns.catplot(y=label, kind='count', data=b)

    return plot
plot_label("label_rice")
plot_label("label_wheat")
plot_label('label_corn')