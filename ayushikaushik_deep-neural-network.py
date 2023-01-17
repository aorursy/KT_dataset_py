import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
raw_data = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')

df = raw_data.copy()
df.info()
df.drop('id',axis=1,inplace=True)
plt.figure(figsize=(10,6))

sns.distplot(df['price'],bins=30)
new_df = df[df['price']<2e6].copy()
plt.figure(figsize=(10,6))

sns.distplot(new_df['price'],bins=30)
(len(df)-len(new_df))*100/len(df)
new_df.corr()['price'].sort_values()
%matplotlib inline
plt.figure(figsize=(10,5))

sns.countplot(x='bathrooms',data=new_df)
plt.figure(figsize=(8,24))

plt.subplot(3,1,1)

plt.scatter(y='sqft_living15',x='price',data=new_df,s=1)

plt.ylabel('sqft_living15')

plt.xlabel('price')

plt.subplot(3,1,2)

plt.scatter(y='sqft_living',x='price',data=new_df,s=1)

plt.ylabel('sqft_living')

plt.xlabel('price')

plt.subplot(3,1,3)

plt.scatter(y='sqft_above',x='price',data=new_df,s=1)

plt.ylabel('sqft_above')

plt.xlabel('price')
plt.figure(figsize=(10,5))

sns.countplot(x='grade',data=new_df)
new_df.head()
new_df['date'].apply(lambda x:x[:-7])
new_df['date'] = pd.to_datetime(new_df['date'])
new_df['month'] = new_df['date'].apply(lambda date:date.month)

new_df['year'] = new_df['date'].apply(lambda date:date.year)
new_df.corr()['price'].sort_values()
new_df.drop('zipcode',axis=1,inplace=True)
new_df.groupby('month').mean()['price'].plot()
new_df.groupby('year').mean()['price'].plot()
new_df.drop('date',axis=1,inplace=True)
new_df['sqft_basement'].value_counts()
def convert_to_dummy(value):

    if value == 0:

        return 0

    else:

        return 1

new_df['basement']=new_df['sqft_basement'].apply(convert_to_dummy)

new_df['basement'].value_counts()
new_df.drop('sqft_basement',axis=1,inplace=True)
new_df['yr_renovated'].value_counts()
def convert_to_dummy(value):

    if value == 0:

        return 0

    else:

        return 1

new_df['renovated']=new_df['yr_renovated'].apply(convert_to_dummy)

new_df['renovated'].value_counts()
new_df.drop('yr_renovated',axis=1,inplace=True)
new_df.columns
from sklearn.model_selection import train_test_split
X = new_df.drop('price',axis=1)

y = new_df['price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
import tensorflow as tf
model = tf.keras.Sequential([

    tf.keras.layers.Dense(19,activation='relu'),

    tf.keras.layers.Dense(50,activation='relu'),

    tf.keras.layers.Dense(50,activation='relu'),

    tf.keras.layers.Dense(50,activation='relu'),

    tf.keras.layers.Dense(50,activation='relu'),

    tf.keras.layers.Dense(50,activation='relu'),

    tf.keras.layers.Dense(1)

])
model.compile(optimizer='adam',loss='mse')
num_epochs=100

model.fit(X_train,y_train.values,epochs=num_epochs,batch_size=128,verbose=2)
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score
predictions = model.predict(X_test)
# Our predictions

plt.scatter(y_test,predictions)



# Perfect predictions

plt.plot(y_test,y_test,'r')
Deep_Net =['Deep Net',mean_absolute_error(y_test,predictions),np.sqrt(mean_squared_error(y_test,predictions)),explained_variance_score(y_test,predictions)]
error_metrics=pd.DataFrame({'model':[],'mean absolute error':[],'root mean squared error':[],'Explained variance score':[]})
error_metrics.loc[0]= Deep_Net
error_metrics
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)
pred = dtree.predict(X_test)
# Our predictions

plt.scatter(y_test,pred)



# Perfect predictions

plt.plot(y_test,y_test,'r')
Decision_Tree =['Decision Tree',mean_absolute_error(y_test,pred),np.sqrt(mean_squared_error(y_test,pred)),explained_variance_score(y_test,pred)]
error_metrics.loc[1] = Decision_Tree
error_metrics
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)
pred_lm = lm.predict(X_test)
# Our predictions

plt.scatter(y_test,pred_lm)



# Perfect predictions

plt.plot(y_test,y_test,'r')
Linear_model =['Linear Regression',mean_absolute_error(y_test,pred_lm),np.sqrt(mean_squared_error(y_test,pred_lm)),explained_variance_score(y_test,pred_lm)]



error_metrics.loc[2]= Linear_model



error_metrics