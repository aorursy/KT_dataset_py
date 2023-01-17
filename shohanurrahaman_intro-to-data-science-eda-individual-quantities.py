# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')

display(df.head(5))



#summarize stats

print(df[['EDUCATION','AGE']].describe())
import matplotlib.pyplot as plt 



print('value counts of gender :', df['EDUCATION'].value_counts())

print('value counts of gender :', df['SEX'].value_counts())

print('value counts of gender :', df['MARRIAGE'].value_counts())



fig, plots = plt.subplots(1,3, figsize=(13,10))



df['EDUCATION'].value_counts().plot(kind='bar',ax=plots[0], title='EDUCATION')

df['SEX'].value_counts().plot(kind='bar',ax=plots[1], title='SEX')

df['MARRIAGE'].value_counts().plot(kind='bar',ax=plots[2], title='MARRIAGE')

plt.show()
cbins = [20,25,30,35,40,45,50,55,60,65,70,75,80]

df['AGE'].plot(kind='hist', bins=cbins, rwidth=0.8)

plt.show()