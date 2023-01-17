



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



data=pd.read_csv("/kaggle/input/top50spotify2019/top50.csv",encoding='ISO-8859-1')
data.head(5)
data.describe()
#Show all the columns names

data.columns
#Renaming the Columns

data=data.rename(columns={'Track.Name':'Track Name','Artist.Name':'Artist Name','Beats.Per.Minute':'Beats per Minute','Loudness..dB..':'Loudness',

                    'Valence.':'Valence','Length.':'Length','Acousticness..':'Acousticness','Speechiness.':'Speechiness'})

#Checking for the null entries

print(data.isna().sum())
#Print the data types

print(data.dtypes)


#Defining the Integer variables

integer=['int64']

integer_variables = data.select_dtypes(include=integer).columns.tolist()

#print(integer_variables)



#Defining the Categorical variables

categorical=['object']

categorical_variables = data.select_dtypes(include=categorical).columns.tolist()

#print(categorical_variables)
#Sorting

data.sort_values(by='Popularity',ascending=False)
#Show different types of Genre and their frequencies

sd=sns.countplot(x='Genre',data=data)

plt.xticks(rotation='vertical')

plt.show()
#Show the co-relation between all the variables

plt.figure(figsize=(9,9))

aa=sns.heatmap(data.corr(),annot=True)

plt.show()
#Boxplot for Popularity by Genre

ac=sns.boxplot(y='Popularity',x='Genre',data=data)

plt.xticks(rotation='vertical')

plt.show()
#Loudness boxplot

ad=sns.boxplot(y='Loudness',data=data)

plt.show()
#Violin plot

plt.figure(figsize=(8,8))

zx=sns.violinplot(x='Loudness',y='Popularity',data=data)

plt.show()
#Show all the plots

xd=sns.pairplot(data)

plt.plot()

plt.show()
#Show dependencies between Acousticness and Popularity

df=sns.regplot(x='Acousticness',y='Popularity',data=data,ci=None)

sns.kdeplot(data.Acousticness,data.Popularity)

plt.show()

#Show dependencies between Loudness and Popularity

df=sns.regplot(x='Loudness',y='Popularity',data=data,ci=None)

sns.kdeplot(data.Loudness,data.Popularity)

plt.show()