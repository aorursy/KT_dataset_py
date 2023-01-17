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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from pandas import Series,DataFrame

import warnings

warnings.filterwarnings('ignore')

print ('Setup Complete')
#Reading data set

df = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv',encoding = "ISO-8859-1")
df.head()
df.info()
df.describe()
#Number of fires per year in Brazil from 1998 to 2017

plt.figure(figsize=(12,4))

sns.barplot(x='year',y='number',data=df)

plt.xticks(fontsize=14, rotation=90)

plt.yticks(fontsize=14)

plt.title('Fires by Year', fontsize = 18)

plt.ylabel('Number of fires', fontsize=14)

plt.xlabel('Year', fontsize=14)
plt.figure(figsize=(12,4))

sns.barplot(x=df.state,y=df.number,data=df)

plt.xticks(fontsize=14, rotation=90)

plt.yticks(fontsize=14)

plt.title('States wise Fires' , fontsize=15)

plt.ylabel('Number of fires', fontsize=14)

plt.xlabel('States', fontsize=14)
dfg=df

df1=dfg.groupby('year')['number'].sum().reset_index()

df1
plt.figure(figsize=(18,6))

gr = sns.lineplot( x = 'year', y = 'number',data = df1, color = 'blue', lw = 2, err_style = None)

gr.xaxis.set_major_locator(plt.MaxNLocator(19))

gr.set_xlim(1998, 2017)

sns.set()

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.title('Number of Fires per Year',fontsize=20)

plt.xlabel('Year', fontsize = 20)

plt.ylabel('Number of Fires', fontsize = 20)
plt.figure(figsize=(12,10))



sns.boxplot(x = 'month', order = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro','Outubro', 'Novembro', 'Dezembro'], 

            y = 'number', data =df)



plt.title('Fires by Month', fontsize = 18)

plt.xlabel('Month', fontsize = 14)

plt.ylabel('Number of Fires', fontsize = 14)