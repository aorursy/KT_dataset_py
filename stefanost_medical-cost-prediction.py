import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



warnings.filterwarnings('ignore')
df=pd.read_csv('../input/insurance/insurance.csv')

df.head()
df.info()
#distribution of charges

sns.distplot(df['charges']);
#pairplot of numerical variables

sns.pairplot(df[['age','bmi','children','charges']]);
#boxplots and violin plots for categorical variable distributions of charges

def dist(feature):

    plt.figure(figsize=(12,4));

    plt.subplot(1,2,1);

    sns.boxplot( x=feature, y='charges', data=df);

    plt.title('%s distribution_boxplot' %feature);

    plt.subplot(1,2,2);

    sns.violinplot(x=feature, y='charges', data=df);

    plt.title('%s distribution_violinplot' %feature);



dist('sex')

dist('children')

dist('smoker')

dist('region')
#correlation heatmap of numerical variables

sns.heatmap(df.corr(), annot=True, fmt='.2f');
# one-hot encoding of categoricals

df['male']=pd.get_dummies(df['sex'],drop_first=True)

df['smoker_yes']=pd.get_dummies(df['smoker'],drop_first=True)

regions=pd.get_dummies(df['region'], prefix='region', prefix_sep='_')

df=pd.concat([df,regions],axis=1)

df.drop(columns=['sex','smoker','region'],inplace=True)
#new correlation heatmap, all variables

plt.figure(figsize=(8,5));

sns.heatmap(df.corr(), annot=True, fmt='.1f');
from sklearn.linear_model import Lasso,Ridge,ElasticNet,LinearRegression, RANSACRegressor

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error as mse
#preprocessing: standardization, train_test split



y=df['charges']

x=df.drop(columns='charges')



x_tr, x_ts, y_tr, y_ts= train_test_split(x, y, test_size=0.15, random_state=42)



sc_x=StandardScaler()

sc_y=StandardScaler()

x_tr=sc_x.fit_transform(x_tr)

x_ts=sc_x.transform(x_ts)

y_tr=sc_y.fit_transform(y_tr[:,np.newaxis]).flatten()

y_ts=sc_y.transform(y_ts[:,np.newaxis]).flatten()
#Linear Regression

lr=LinearRegression()

lr.fit(x_tr,y_tr)

y_tr_pred=lr.predict(x_tr)

y_ts_pred_lr=lr.predict(x_ts)

print('mse train:', mse(y_tr,y_tr_pred))

print('mse test:', mse(y_ts,y_ts_pred_lr))
#RANSAC Regression

rnsc=RANSACRegressor(LinearRegression(),max_trials=100,min_samples=50,loss='absolute_loss',

                    residual_threshold=5.0,random_state=42)

rnsc.fit(x_tr,y_tr)

y_tr_pred=rnsc.predict(x_tr)

y_ts_pred_ransac=rnsc.predict(x_ts)

print('mse train:', mse(y_tr,y_tr_pred))

print('mse test:', mse(y_ts,y_ts_pred_ransac))
#Ridge Regression

rdg=Ridge(alpha=0.00001)

rdg.fit(x_tr,y_tr)

y_tr_pred=rdg.predict(x_tr)

y_ts_pred_rdg=rdg.predict(x_ts)

print('mse train:', mse(y_tr,y_tr_pred))

print('mse test:', mse(y_ts,y_ts_pred_rdg))
#Lasso Regression

ls=Lasso(alpha=0.00000000001)

ls.fit(x_tr,y_tr)

y_tr_pred=ls.predict(x_tr)

y_ts_pred_ls=ls.predict(x_ts)

print('mse train:', mse(y_tr,y_tr_pred))

print('mse test:', mse(y_ts,y_ts_pred_ls))
#ElasticNet Regression

lnt=ElasticNet(alpha=0.00000000000000001, l1_ratio=0.5)

lnt.fit(x_tr,y_tr)

y_tr_pred=lnt.predict(x_tr)

y_ts_pred_el=lnt.predict(x_ts)

print('mse train:', mse(y_tr,y_tr_pred))

print('mse test:', mse(y_ts,y_ts_pred_el))
#stacked regressions



y_ts_stacked=(y_ts_pred_lr + y_ts_pred_ransac + y_ts_pred_rdg + y_ts_pred_el +y_ts_pred_ls)/5

print("mse stacked:", mse(y_ts,y_ts_stacked))
import tensorflow as tf

from keras import models

from keras import layers

from keras.callbacks import EarlyStopping, ModelCheckpoint



model=models.Sequential()

model.add(layers.Dense(64, activation='relu', input_shape=(x_tr.shape[1],)))

model.add(layers.Dense(64,activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1))

model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])

model.summary()
stop=EarlyStopping(patience=5,verbose=1)

check=ModelCheckpoint('DNN_linear_regression.h5', verbose=1, save_best_only=True)

fit=model.fit(x_tr,y_tr, validation_split=0.1, batch_size=16, epochs=50,

                 callbacks=[stop,check])
test_mse,test_mae=model.evaluate(x_ts,y_ts)

print('mse test:', test_mse)
#second DNN



model2=models.Sequential()

model2.add(layers.Dense(32, activation='relu', input_shape=(x_tr.shape[1],)))

model2.add(layers.Dense(32,activation='relu'))

model2.add(layers.Dropout(0.5))

model2.add(layers.Dense(1))

model2.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])

model2.summary()
stop2=EarlyStopping(patience=5,verbose=1)

check2=ModelCheckpoint('DNN2_linear_regression.h5', verbose=1, save_best_only=True)

fit2=model2.fit(x_tr,y_tr, validation_split=0.1, batch_size=16, epochs=30,

                 callbacks=[stop,check])
test_mse2,test_mae2=model2.evaluate(x_ts,y_ts)

print('mse test:', test_mse2)
from sklearn.preprocessing import PolynomialFeatures as poly



lreg=LinearRegression()

sq=poly(degree=2)

x_tr_sq=sq.fit_transform(x_tr)

x_ts_sq=sq.transform(x_ts)

lreg.fit(x_tr_sq,y_tr)

print('polynomial 2nd degree mse train:' ,mse(y_tr, lreg.predict(x_tr_sq)))

print('polynomial 2nd degree mse test:' ,mse(y_ts, lreg.predict(x_ts_sq)))

print("")

print('-----------')

print("")

cube=poly(degree=3)

x_tr_cb=cube.fit_transform(x_tr)

x_ts_cb=cube.transform(x_ts)

lreg.fit(x_tr_cb,y_tr)

print('polynomial 3nd degree mse train:' ,mse(y_tr, lreg.predict(x_tr_cb)))

print('polynomial 3nd degree mse test:' ,mse(y_ts, lreg.predict(x_ts_cb)))
from sklearn.ensemble import RandomForestRegressor as rfr



forest=rfr(n_estimators=1000, criterion='mse', random_state=7)

forest.fit(x_tr,y_tr)

y_tr_pred=forest.predict(x_tr)

y_ts_pred=forest.predict(x_ts)



print('mse train:', mse(y_tr,y_tr_pred))

print('mse test:', mse(y_ts,y_ts_pred))
model3=models.Sequential()

model3.add(layers.Dense(64, activation='relu', input_shape=(x_tr_sq.shape[1],)))

model3.add(layers.Dense(64,activation='relu'))

model3.add(layers.Dropout(0.5))

model3.add(layers.Dense(1))

model3.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])

model3.summary()
stop3=EarlyStopping(patience=5,verbose=1)

check3=ModelCheckpoint('DNN3_linear_regression.h5', verbose=1, save_best_only=True)

fit3=model3.fit(x_tr_sq,y_tr, validation_split=0.1, batch_size=16, epochs=30,

                 callbacks=[stop,check])
test_mse3,test_mae3=model3.evaluate(x_ts_sq,y_ts)

print('mse test:', test_mse3)