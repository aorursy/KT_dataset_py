#import libries

import pandas as pd

import numpy as np

from pandas import DataFrame, Series

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



filename='/kaggle/input/top50spotify2019/top50.csv'

df=pd.read_csv(filename,encoding='ISO-8859-1')

df.head()
df.tail()
df.info()
df.describe()
#check null values

df.isnull().sum()
#checking every  feature contains

column = df.columns



for col  in column:

    print('-'*20)

    print(col ,'column contents')

    print(df[col].value_counts().sort_values())

    print('\n')
# artist name and there no. of songs

df['Unnamed: 0'].groupby(df['Artist.Name']).count()
df[df['Artist.Name']=='J Balvin']
df[df['Artist.Name']=='Ed Sheeran']
# numbers of genre and there count

df['Unnamed: 0'].groupby(df['Genre']).count()
df[df['Genre']=='dance pop']
plt.figure(figsize=(30,8))

sns.countplot(df['Artist.Name'])

plt.show()
plt.figure(figsize=(8,8))

plt.pie(df['Genre'].value_counts().values, labels=df['Genre'].value_counts().index, wedgeprops = { 'linewidth' : 7, 'edgecolor' : 'white' })

p=plt.gcf()

my_circle=plt.Circle( (0,0), 0.7, color='white')

p.gca().add_artist(my_circle)

plt.show()
df['Popularity'].plot(kind='hist')

plt.show()
df['Energy'].plot(kind='hist')
sns.distplot( df['Energy'] , color="skyblue")
plt.figure(figsize=(10,7))

sns.scatterplot(x = df['Popularity'] ,y = df['Energy'], hue = df['Genre'])

plt.show()
plt.figure(figsize=(10,8))

sns.scatterplot(y = df['Danceability'] ,x = df['Popularity'], hue = df['Genre'])

plt.show()
sns.jointplot(x=df['Beats.Per.Minute'], y=df['Popularity'], kind='kde', color="skyblue")

plt.show()
corr_matrix = df.corr()

plt.figure(figsize=(10,8))

sns.heatmap(corr_matrix, annot =True)
# try to predict populrity on features

data = df.copy()

data.head()
data =data.drop(['Unnamed: 0','Track.Name','Artist.Name'],axis=1)

x = data.drop(['Popularity'],axis=1)

y = data['Popularity']
x = pd.get_dummies(x)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y ,test_size=0.3)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
from sklearn.metrics import mean_squared_error

print('Mean Squared error :',mean_squared_error(y_test,y_pred))

print('Root mean squared error :', np.sqrt(mean_squared_error(y_test,y_pred)))
result = DataFrame({'Actual': y_test, 'Predicted': y_pred})

result