import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from PIL import Image

from matplotlib.pyplot import MaxNLocator, FuncFormatter
data = pd.read_csv("/kaggle/input/forest-fires-in-brazil/amazon.csv",encoding="ISO-8859-1")

data.head(10)
#dropping date column

data.drop(['date'],axis=1,inplace=True)

data.head()
#shape fo the data

print('Shape of the Data:')

data.shape
#checking for data types.

data.dtypes
data['year'].unique()
#checking for number of states

data['state'].unique()
data['month'].unique()
month_map={'Janeiro':'January', 'Fevereiro':'February', 'Março':'March', 'Abril':'April', 'Maio':'May', 'Junho':'June', 'Julho':'July',

       'Agosto':'August', 'Setembro':'September', 'Outubro':'October', 'Novembro':'November', 'Dezembro':'December'}

data['month']=data['month'].map(month_map)
#checking for number of months

data['month'].unique()
#information about the data

print('Information of the Data:- ')

data.info()
#checking for null values

data.isnull().sum()
#checking how many  fires were reported in 19 years

print('Number of fires  were reported in 19 years:',data['number'].sum())

#Each year how many fires were reported

table = pd.pivot_table(data,values="number",index=["year"],aggfunc=np.sum)

table

#checking yearwise Trend





plt.figure(figsize = (15,10))

plot = sns.lineplot(data = data, x = 'year', y = 'number', markers = True)

plot.xaxis.set_major_locator(plt.MaxNLocator(19))

plt.title('Yearwise Trend')

plot.set_xlim(1998, 2017)

#checking monthwise trend

plt.figure(figsize=(15,7))

sns.boxplot(x='month',y='number',data=data[['month','number']])

plt.title('Month Wise Trend')

plt.show()
#checking statewise trend

plt.figure(figsize=(20,10))

st=sns.boxenplot(x='state',y='number',data=data[['state','number']])

st.set_xticklabels(st.get_xticklabels(), rotation=45)

plt.title('State Wise Trend')

plt.show()

