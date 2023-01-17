# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import linear_model

import matplotlib as lib

from sklearn.model_selection import train_test_split 

from sklearn import metrics

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv(r'''/kaggle/input/forest-fires-in-brazil/amazon.csv''', engine='python')
print(df.head())

print(df.shape)

print(df.columns)

print(df.dtypes)

print (df.describe())
df['Month'] = df['month'].map( {'Janeiro': 'January', 'Fevereiro': 'Febuary', 'Março': 'March', 'Abril': 'April','Maio': 'May', 'Junho': 'June', 'Julho': 'July', 'Agosto': 'August', 'Setembro': 'September', 'Outubro': 'October', 'Novembro': 'November', 'Dezembro':'December' } ).astype(str)

df=df.drop(['month'], axis = 1)

print(df.head())
print(df.state.unique())

print(df.isnull().sum())
df1=df.groupby('state').number.mean().reset_index()

df1=df1.sort_values('number', ascending=False)

print(df1)

df1.plot(x='state', y='number', kind = 'bar')

plt.show()
df1=df.groupby('year').number.mean().reset_index()

df1=df1.sort_values('year', ascending=False)

print(df1)

df1.plot(x='year', y='number', kind = 'line')

plt.show()
df1=df.groupby('Month').number.mean().reset_index()

df1=df1.sort_values('number', ascending=False)

print(df1)

df1.plot(x='Month', y='number', kind = 'bar')

plt.show()

df1=df.groupby(['state', 'Month']).number.mean().reset_index()

print(df1)
li_state = ['Acre','Alagoas','Amapa','Amazonas', 'Bahia' ,'Ceara' ,'Distrito Federal',

 'Espirito Santo', 'Goias' ,'Maranhao', 'Mato Grosso', 'Minas Gerais' ,'Pará',

 'Paraiba' ,'Pernambuco', 'Piau', 'Rio', 'Rondonia' ,'Roraima' ,'Santa Catarina',

 'Sao Paulo' ,'Sergipe', 'Tocantins']

df_f=pd.DataFrame(columns=['state', 'number'])

for i in range (len(li_state)):

    df2 = pd.DataFrame(columns=['state', 'number'])

    df3=df1.loc[df1.state==li_state[i],:].number.nlargest(3)

    df2.number=df3

    df2['state']=li_state[i]

    #print(df2)

    df_f=df_f.append(df2)

    del(df2)

print(df_f)

df_f.plot(x='state', y='number', kind='bar')

plt.show()
state_map = {'Acre':1,'Alagoas':2,'Amapa':3,'Amazonas':3,'Bahia':4 ,'Ceara':5 ,'Distrito Federal':6,

 'Espirito Santo':7, 'Goias':8 ,'Maranhao':9, 'Mato Grosso':10, 'Minas Gerais':11 ,'Pará':12,

 'Paraiba':13 ,'Pernambuco':14, 'Piau':15, 'Rio':16, 'Rondonia':17 ,'Roraima':18 ,'Santa Catarina':19,

 'Sao Paulo':21 ,'Sergipe':22, 'Tocantins':23}



df['State'] = df['state'].map(state_map).astype(str)

df=df.drop(['state'], axis = 1)

print(df.head())
df['month'] = df['Month'].map( {'January': 1, 'Febuary': 2, 'nan': 3, 'April': 4,'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December':12}).astype(int)



print(df.head())

df=df.drop(['Month'], axis = 1)

print(df.head())

ax=sns.heatmap(df.corr())

print(ax)


X = df['month'].values.reshape(-1,1)

y = df['number'].values.reshape(-1,1)



#Splitting the dataset in Test and Training set



X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

reg= linear_model.LinearRegression()



#Fitting the model



reg.fit(X_train,y_train)



#To retrieve the intercept:

print(reg.intercept_)

#For retrieving the slope:

print(reg.coef_)

y_pred = reg.predict(X_test)

df3 = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})



# We will see 25 values for Predicted value as actual dataset is pretty big.

df1 = df3.head(25)

df1.plot(kind='bar',figsize=(16,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))