import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import scipy.stats as sc
data=pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv",encoding='utf-8')
data
# checking nan values.....

data.isnull().sum()

# missing value percentage....

(data.isnull().sum()/data.shape[0])*100
# hist plot for rating.......

xt=data['Rating'].hist(bins=15, density=True, stacked=True, color='teal', alpha=0.6)

data["Rating"].plot(kind='density', color='teal')



plt.grid()

plt.xlim(-10.5,10.5)

plt.show()
# cleaning of rating data

t_data=data[pd.notnull(data['Rating'])]



# mean,median,mode

mean=np.mean(t_data['Rating'])

median=np.median(t_data['Rating'])

mode=sc.mode(t_data['Rating'])



print(mean,median,mode)
# rating is right skewed so take median in place of nan

data['Rating'].fillna(median,inplace=True)



# In other features missing value % is not considrable so drop nan

data.dropna(inplace=True)

data.info()
# Remove dublicate values



(data.duplicated().value_counts()/data.shape[0])*100
data.drop_duplicates(inplace=True) 
data.info()
# converting last date

data['Last Updated']=pd.to_datetime(data['Last Updated'])

data['before update']=data['Last Updated'].max()-data['Last Updated']

data['Installs']=data['Installs'].str.replace(',','').str.replace('+','').astype('int')

 
# converting review to int

data['Reviews']=data['Reviews'].astype('int')
# converting size

data['Size']=data['Size'].str.replace('M','e+6').str.replace('k','e+3').str.replace('Varies with device','0').astype('float')
data['Price']=data['Price'].str.replace('$','').astype('float')
data.describe()
# most Most popular category

plt.figure(figsize=(40,10))

data['Category'].value_counts().plot(kind='pie')

plt.show()

plt.figure(figsize=(40,10))

data['Category'].value_counts().plot(kind='bar')

plt.xlabel('Category')

plt.ylabel('freq.')

plt.grid()

plt.show()
#  Content Rating 

plt.figure(figsize=(40,10))

explode=[0.01,0.1,0.1,0.1,0.1,0.5]

data['Content Rating'].value_counts().plot(kind='pie',autopct="%2i%%",explode=explode)

plt.legend()

plt.show()



plt.figure(figsize=(5,5))

data['Content Rating'].value_counts().plot(kind='bar')

plt.xlabel('Content Rating')

plt.ylabel('freq.')



plt.grid()

plt.show()
data['Size'].value_counts()
plt.figure(figsize=(40,10))

data['Genres'].value_counts().plot(kind='bar')

plt.xlabel('Genres')

plt.ylabel('freq.')

plt.show()
plt.figure(figsize=(10,10))

explode=[0.1,0]

data['Type'].value_counts().plot(kind='pie',autopct="%2i%%",explode=explode)

plt.legend()

plt.show()

plt.figure(figsize=(10,10))

data['Type'].value_counts().plot(kind='bar')

plt.xlabel('Type')

plt.ylabel('freq.')

plt.show()
# max size app

data[data['Size']==data['Size'].max()]
# max size install app

data[data['Installs']==data['Installs'].max()]
# App which hasn't been updated

data[data['before update']==data['before update'].max()]
# App with largest number of reviews

data[data['Reviews']==data['Reviews'].max()]
##B most reviewed apps

import seaborn as sns

sorte = data.sort_values(['Reviews'],ascending = 0 )[:20]

ax = sns.barplot(x = 'Reviews' , y = 'App' , data = sorte )

ax.set_xlabel('Reviews')

ax.set_ylabel('')

ax.set_title("Most Popular Categories in Play Store", size = 20)
# most populer catogry by Family

data_cat=data[data['Category']=='FAMILY'].sort_values(['Installs'],ascending=0)[:20]



ax = sns.barplot(x = 'Installs' , y = 'App' , data = data_cat )

ax.set_xlabel('APPs')

ax.set_ylabel('Most INSTALLED APP IN FAMILY CATO.')

ax.set_title("Most Popular Categories in Play Store", size = 20)

data_cat=data[data['Category']=='GAME'].sort_values(['Installs'],ascending=0)[:20]



ax = sns.barplot(x = 'Installs' , y = 'App' , data = data_cat )

ax.set_xlabel('APPs')

ax.set_ylabel('Most INSTALLED APP IN GAME CATO.')

ax.set_title("Most Popular Categories in Play Store", size = 20)

data_cat=data[data['Category']=='TOOLS'].sort_values(['Installs'],ascending=0)[:20]



ax = sns.barplot(x = 'Installs' , y = 'App' , data = data_cat )

ax.set_xlabel('APPs')

ax.set_ylabel('Most INSTALLED APP IN TOOLS CATO.')

ax.set_title("Most Popular Categories in Play Store", size = 20)
data_cat=data[data['Category']=='BUSINESS'].sort_values(['Installs'],ascending=0)[:10]



ax = sns.barplot(x = 'Installs' , y = 'App' , data = data_cat )

ax.set_xlabel('APPs')

ax.set_ylabel('Most INSTALLED APP IN BUSINESS CATO.')

ax.set_title("Most Popular Categories in Play Store", size = 20)
# most use version of android in android phones

plt.figure(figsize=(25,25))

data['Android Ver'].value_counts().plot(kind='pie')

plt.legend()

plt.show()

plt.figure(figsize=(20,10))

data['Android Ver'].value_counts().plot(kind='bar')

plt.xlabel('android version')

plt.ylabel('apps most persuing..')



plt.show()
# Data Modeling

data_model_x=data[['Category','Reviews','Size','Installs','Type','Price','Content Rating','Genres','before update']]

data_model_y=data[['Rating']]

from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()

scaler.fit(data_model_x[['before update']])

data_model_x[['before update']]=scaler.transform(data_model_x[['before update']])



data_model_x
## one-hot encoding of Category,Content Rating,Type,Genres

encoded_x=pd.get_dummies(data_model_x, columns=['Category',"Content Rating","Type","Genres"])
# train and test split

from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(encoded_x,data_model_y,random_state=0)
x_train
y_train.shape,y_test.shape
# feature scaling

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train=sc.fit_transform(x_train)

x_test=sc.transform(x_test)
x_train.shape,x_test.shape
# Appling linear_regression



from sklearn.linear_model import LinearRegression

cls=LinearRegression()



cls.fit(x_train,y_train)

cls.predict(x_test)
cls.score(x_train,y_train)

cls.score(x_test,y_test)
# using support vector regressor



from sklearn import svm
clf=svm.SVR(C=2.0,epsilon=0.3)

clf.fit(x_train,y_train)

clf.predict(x_test)
clf.score(x_train,y_train)
clf.score(x_test,y_test)
# from neural network



from keras.models import Sequential

from keras.layers import Dense



model=Sequential()
layer1=Dense(units=500,activation='relu',input_dim=165)

model.add(layer1)

model.add(Dense(units=50,activation='relu'))

model.add(Dense(units=50,activation='relu'))

model.add(Dense(units=50,activation='relu'))

model.add(Dense(units=50,activation='relu'))

model.add(Dense(units=1,activation='relu'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=25,batch_size=50,validation_data=(x_test,y_test))
score=model.evaluate(x_test,y_test)

score
# from randomforest

from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(max_depth=8,random_state=0)

rfr.fit(x_train, y_train)

y_pred=rfr.predict(x_test)

print(rfr.score(x_train, y_train))

rfr.score(x_test, y_test)