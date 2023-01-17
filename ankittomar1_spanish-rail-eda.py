import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import seaborn as sns
# load the dataset

df = pd.read_csv('../input/renfe.csv')

df.head(2)
# let us clean the dataset 

# del unnamed it looks like a noise to the data

del df['Unnamed: 0']
# validate the df 

df.head(2)
# validate the summary of the dataset 

df.info()
# types of the train type 



plt.figure(figsize= (5,10))

plt.subplot(311)

plt.title('Train Class')

plt.tight_layout()

sns.countplot(y =df['train_class'])



plt.subplot(312)

plt.title('Train type')

plt.tight_layout()

sns.countplot(y = df['train_type'])



plt.subplot(313)

plt.title('fare type')

plt.tight_layout()

sns.countplot(y =df['fare'])



# craete a dataset for uptime starting from MADRID	

df_M = df[df['origin'] =='MADRID']

df_S = df[df['origin'] =='SEVILLA']

df_P = df[df['origin'] =='PONFERRADA']

df_B = df[df['origin'] =='BARCELONA']

df_V = df[df['origin'] =='VALENCIA']
df_M.info()
# very high nan values 

# 13%  NaN - 183258

df_M.price.isna().sum()

plt.figure(figsize=(5,5))

plt.subplot(211)

df_M['price'].hist()

plt.subplot(212)

sns.boxplot(df_M['price'])



# fillna with mean

df_M['price'].fillna(df_M['price'].mean(), inplace=True)



#  Na is filled

df_M.price.isna().sum()
# very high nan values 

# 13%  NaN - 21941

df_P.price.isna().sum()

plt.figure(figsize=(5,5))

plt.subplot(211)

df_P['price'].hist()

plt.subplot(212)

sns.boxplot(df_P['price'])



# fillna with mean

df_P['price'].fillna(df_P['price'].mean(), inplace=True)



#  Na is filled

df_P.price.isna().sum()
# very high nan values 

# 13%  NaN - 88904

df_S.price.isna().sum()



plt.figure(figsize=(5,5))

plt.subplot(211)

df_S['price'].hist()

plt.subplot(212)

sns.boxplot(df_S['price'])





# fillna with mean

df_S['price'].fillna(df_S['price'].mean(), inplace=True)



#  Na is filled

df_S.price.isna().sum()
# very high nan values 

# 13%  NaN - 88904

df_B.price.isna().sum()



plt.figure(figsize=(5,5))

plt.subplot(211)

df_B['price'].hist()

plt.subplot(212)

sns.boxplot(df_B['price'])





# fillna with mean

df_B['price'].fillna(df_B['price'].mean(), inplace=True)



#  Na is filled

df_B.price.isna().sum()
# very high nan values 

# 13%  NaN - 88904

df_V.price.isna().sum()



plt.figure(figsize=(5,5))

plt.subplot(211)

df_V['price'].hist()

plt.subplot(212)

sns.boxplot(df_V['price'])





# fillna with mean

df_V['price'].fillna(df_V['price'].mean(), inplace=True)



#  Na is filled

df_V.price.isna().sum()
# look on the first glance on the origin from Madrid

df_M.head()
# types of the train type 



plt.figure(figsize= (5,12))

plt.subplot(511)

plt.title('Train Class')

plt.tight_layout()

sns.countplot(y =df_M['train_class'])



plt.subplot(512)

plt.title('Train type')

plt.tight_layout()

sns.countplot(y = df_M['train_type'])



plt.subplot(513)

plt.title('fare type')

plt.tight_layout()

sns.countplot(y =df_M['fare'])



plt.subplot(514)

plt.title('price ')

plt.tight_layout()

sns.boxplot(df_M['price'])





plt.subplot(515)

plt.title('destination ')

plt.tight_layout()

sns.countplot(y = df_M['destination'])

# check out of there are null values 

df_M.info()



# yes there are you use mode to fill that 

df_M.dropna(axis=1, inplace = True)



# check out of there are null values 

df_M.info()

# assign the values

X_df_M = df_M.drop(columns= ['price', 'insert_date', 'start_date', 'end_date'])

y_df_M = df_M['price'].values
X_df_M.isna().sum()
# implementing the encoding

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

Xm = encoder.fit_transform(X_df_M.values)

Xm
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    Xm, y_df_M, test_size=0.1, random_state=2019

)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
# we are using LR 

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
#  predict 

y_pred = model.predict(X_test)
# values 

y_pred
# looking for accuracy matrix 

from sklearn.metrics import mean_squared_error

print(np.sqrt(mean_squared_error(y_test, y_pred)))
# types of the train type 



plt.figure(figsize= (5,12))

plt.subplot(511)

plt.title('Train Class')

plt.tight_layout()

sns.countplot(y =df_S['train_class'])



plt.subplot(512)

plt.title('Train type')

plt.tight_layout()

sns.countplot(y = df_S['train_type'])



plt.subplot(513)

plt.title('fare type')

plt.tight_layout()

sns.countplot(y =df_S['fare'])



plt.subplot(514)

plt.title('price ')

plt.tight_layout()

sns.boxplot(df_S['price'])





plt.subplot(515)

plt.title('destination ')

plt.tight_layout()

sns.countplot(y = df_S['destination'])

# look for how many null values are there 

df_S.info()
# yes there are you use mode to fill that 

df_S.dropna(axis=1, inplace = True)



# check out of there are null values 

df_S.info()

# assign the values

X_df_S = df_S.drop(columns= ['price', 'insert_date', 'start_date', 'end_date'])

y_df_S = df_S['price'].values
# implementing the encoding

from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

Xs = encoder.fit_transform(X_df_S.values)

Xs
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    Xs, y_df_S, test_size=0.2, random_state=2019

)



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
# we are using LR 

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
#  predict 

y_pred = model.predict(X_test)
y_pred
# looking for accuracy matrix 

from sklearn.metrics import mean_squared_error

print(np.sqrt(mean_squared_error(y_test, y_pred)))