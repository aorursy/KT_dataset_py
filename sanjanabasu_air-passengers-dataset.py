import seaborn as sns

import pandas as pd

import numpy as np



import matplotlib.pyplot as plt  

import seaborn as sns
data = pd.read_csv('../input/air-passengers/AirPassengers.csv')
def clean_date(x):

    l=[]

    if isinstance(x,str):

        l=x.split("-")

        x=l[0]

    return(x)

data['Year'] = data['Month'].apply(clean_date).astype('int')
print("Train: rows:{} columns:{}".format(data.shape[0], data.shape[1]))
data.describe()
data['#Passengers'].describe()
data.columns
data.info()
pd.qcut(data['Year'], q=4)
data1=data.groupby(by='Year').sum()
sns.barplot(x=data1.index, y='#Passengers',data=data1)

plt.show()
#Parse strings to datetime type

data['Month'] = pd.to_datetime(data['Month'],infer_datetime_format=True) 

#convert from string to datetime

data = data.set_index(['Month'])
def plot_df(data, x, y, title="", xlabel='Passengers', ylabel='Value', dpi=100):

    plt.figure(figsize=(16,5), dpi=dpi)

    plt.plot(x, y, color='tab:red')

    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)

    plt.show()



plot_df(data, x=data.index, y=data['#Passengers'], title='Monthly AirPassengers from 1949-01 to 1960-12.')    