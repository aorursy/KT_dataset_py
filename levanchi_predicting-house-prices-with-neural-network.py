import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
df.head()
#df.describe()
#df.info()
df1 = df[['price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long', 'sqft_living15', 'sqft_lot15']]
h = df1.hist(bins = 25,figsize = (16,16),xlabelsize = 10,ylabelsize = 10,xrot=-15)
sns.despine(left = True,bottom=True)
[x.title.set_size(12) for x in h.ravel()];
[x.yaxis.tick_left() for x in h.ravel()];
sns.set(style="whitegrid", font_scale=1)
f, axes = plt.subplots(1,2,figsize = (15,5))
sns.boxplot(x = 'bedrooms',y = 'price',data = df,ax = axes[0]);
sns.boxplot(x = 'floors', y = 'price',data = df,ax = axes[1]);
sns.despine(left=True, bottom=True)
axes[0].set(xlabel = 'Bedrooms',ylabel = 'Price')
axes[0].yaxis.tick_left()

axes[1].set(xlabel = 'Floors',ylabel = 'Price')
axes[1].yaxis.set_label_position('right')
axes[1].yaxis.tick_right()

f, axe = plt.subplots(1,1,figsize = (15,5))
sns.boxplot(x = 'bathrooms' , y = 'price',data = df,ax = axe);
axe.set(xlabel = 'Bathrooms', ylabel = 'Price');
f, axes = plt.subplots(1, 2,figsize=(15,5))
sns.boxplot(x=df['waterfront'],y=df['price'], ax=axes[0])
sns.boxplot(x=df['view'],y=df['price'], ax=axes[1])
sns.despine(left=True, bottom=True)
axes[0].set(xlabel='Waterfront', ylabel='Price')
axes[0].yaxis.tick_left()
axes[1].yaxis.set_label_position("right")
axes[1].yaxis.tick_right()
axes[1].set(xlabel='View', ylabel='Price')

f, axe = plt.subplots(1, 1,figsize=(12.18,5))
sns.boxplot(x=df['grade'],y=df['price'], ax=axe)
sns.despine(left=True, bottom=True)
axe.yaxis.tick_left()
axe.set(xlabel='Grade', ylabel='Price');
df.corr()['price'].sort_values()
plt.figure(figsize = (10,6))
sns.scatterplot(x = 'sqft_living',y = 'price', data = df);
df.head()
df = df.drop('id',axis = 1)
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].apply(lambda date: date.year)
df['month'] = df['date'].apply(lambda date: date.month)
df.groupby('month').mean()['price'].plot();
df = df.drop('date',axis = 1)
#df = df.drop('zipcode',axis =1)
df['yr_renovated'].value_counts()
df['sqft_basement'].value_counts()
X = df.drop('price', axis = 1).values
y = df['price'].values
X.shape,y.shape
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.8, random_state = 42)
mm_scaler = MinMaxScaler()
X_train = mm_scaler.fit_transform(X_train)
X_test = mm_scaler.transform(X_test)
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
model = Sequential()

model.add(Dense(units = 6,activation = 'relu',input_dim = X.shape[1]))
model.add(Dense(units = 6,activation = 'relu'))
model.add(Dense(units = 6,activation = 'relu'))
model.add(Dense(units = 6,activation = 'relu'))

model.add(Dense(units = 1,activation = 'linear'))

adam = optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(optimizer = adam,loss = 'mse')
model.fit(x = X_train, y = y_train,epochs = 1000,validation_data = (X_test,y_test), batch_size = 128,verbose = 1)
losses = pd.DataFrame(model.history.history)
losses.plot()
y_pred = model.predict(X_test)
np.sqrt(metrics.mean_squared_error(y_test,y_pred))
metrics.mean_absolute_error(y_test,y_pred)
df['price'].describe()
plt.figure(figsize = (12,8))
plt.scatter(y_test, y_pred)
plt.plot(y_test, y_test,'r')
