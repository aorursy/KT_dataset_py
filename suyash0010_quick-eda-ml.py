# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/property-prices-in-tunisia/Property Prices in Tunisia.csv')

df.head(20)
df.dtypes
df.isnull().sum()
df_sold_rent = df.groupby(['type','category']).count()

df_sold_rent = df_sold_rent.unstack('category')['room_count']

df_sold_rent
plt.style.use('ggplot')



fig, ax = plt.subplots(1,2, figsize=(20,5))



ax[0].bar(df_sold_rent.columns,df_sold_rent.iloc[0,:])

ax[0].set_title(df_sold_rent.index[0])

ax[0].set_xticklabels(labels = df_sold_rent.columns,rotation=60)



ax[1].bar(df_sold_rent.columns,df_sold_rent.iloc[1,:])

ax[1].set_xticklabels(labels = df_sold_rent.columns,rotation=60)

ax[1].set_title(df_sold_rent.index[1])

# plt.xticks(rotation=60)

# plt.show()
df_city_category = df.groupby(['category','city'])['price'].count()

df_city_category=df_city_category.unstack('category')

df_city_category.fillna(0,inplace=True)

df_city_category


fig, ax = plt.subplots(7,1,figsize=(15,55))

for i in range(len(df_city_category.columns)):

    plt.tight_layout()

    ax[i].bar(df_city_category.index,df_city_category[df_city_category.columns[i]])

    ax[i].set_title(df_city_category.columns[i])

    ax[i].set_xticklabels(labels=df_city_category.index,rotation=45)
for col in df_city_category.columns:

    df_bool = df_city_category.loc[df_city_category[col]==df_city_category[col].max(),:]

    print('{} are in high demand at {}'.format(col,df_bool.index[0]))

df_city_category = df.groupby(['city','category'])['price'].mean()

df_city_category =df_city_category.unstack('category')

df_city_category.fillna(0,inplace=True)

df_city_category

fig, ax = plt.subplots(7,1,figsize=(15,55))

for i in range(len(df_city_category.columns)):

    plt.tight_layout()

    ax[i].bar(df_city_category.index,df_city_category[df_city_category.columns[i]])

    ax[i].set_title(df_city_category.columns[i])

    ax[i].set_xticklabels(labels=df_city_category.index,rotation=45)
important_cities = []

for category in df_city_category.columns:

    df_price_sorted = df_city_category[category].sort_values(ascending=False)

    important_cities.append(df_price_sorted.head(5).index)



city_features=[]

for each in important_cities:

    for city in each:

        if city not in city_features:

            city_features.append(city)

                      

print(len(city_features))

df['city'].value_counts().count()
unimportant_cities = [city for city in df['city'].value_counts().index if city not in city_features]

len(unimportant_cities)

for city in unimportant_cities:

    df.loc[df['city']==city,'city']='Other'

df['city'].value_counts()


df_region_category = df.groupby(['region','category'])['price'].mean()

df_region_category =df_region_category.unstack('category')

df_region_category.fillna(0,inplace=True)

df_region_category

important_regions = []

for category in df_region_category.columns:

    df_price_sorted = df_region_category[category].sort_values(ascending=False)

    important_regions.append(df_price_sorted.head(3).index)

region_features=[]

for each in important_regions:

    for region in each:

        if region not in region_features:

            region_features.append(region)

                      

print(len(region_features))

df['region'].value_counts().count()
unimportant_regions = [region for region in df['region'].value_counts().index if region not in region_features]

len(unimportant_regions)

for region in unimportant_regions:

    df.loc[df['region']==region,'region']='Other'

df['region'].value_counts()
df.drop(['price'],axis=1,inplace=True)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df[['room_count','size','bathroom_count']])

df[['room_count','size','bathroom_count']] = scaler.transform(df[['room_count','size','bathroom_count']])

df.head()
dummy_df = pd.get_dummies(df,drop_first=True)

len(dummy_df.columns)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, cross_val_score

x = dummy_df.drop('log_price',axis=1).values

y = dummy_df['log_price'].values

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=123)

lr = LinearRegression()

lr.fit(x_train,y_train)

lr.score(x_test,y_test)

cv_results = cross_val_score(lr,x,y,cv=10)

cv_results.mean()

lr.coef_
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()

cv_results = cross_val_score(dt,x,y,cv=10)

cv_results.mean()
from sklearn.model_selection import GridSearchCV

params = {'max_depth':range(1,11)}

grid_cv = GridSearchCV(dt,param_grid=params)

grid_cv.fit(x_train,y_train)

grid_cv.best_params_
dt = DecisionTreeRegressor(max_depth=6)

cv_results = cross_val_score(dt,x,y,cv=10)

cv_results.mean()
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

cv_results = cross_val_score(rf,x,y,cv=10)

cv_results.mean()
from sklearn.model_selection import RandomizedSearchCV

params = {'n_estimators':range(50,200)}

grid_cv = RandomizedSearchCV(rf,param_distributions=params)

grid_cv.fit(x_train,y_train)

best =grid_cv.best_params_
rf = RandomForestRegressor(n_estimators=best)

cv_results = cross_val_score(dt,x,y,cv=10)

cv_results.mean()
import numpy as np

from keras.layers import Dense

from keras.models import Sequential

from keras.callbacks import EarlyStopping

model = Sequential()

n_cols = x.shape[1]

model.add(Dense(100,activation='relu',input_shape=(n_cols,)))

model.add(Dense(120,activation='relu'))

model.add(Dense(120,activation='relu'))

model.add(Dense(120,activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
m_fit = model.fit(x,y,batch_size=50,epochs=30,validation_split=0.3,callbacks=[EarlyStopping(patience=2)])

m_fit.history['loss']

# m_fit.history['accuracy']
pred=[]

for i in range(len(y)):

    myx= x[i].reshape((-1,47))

    pred.append(float(model.predict(myx)))

def r_squared(y_true, y_pred):



    SS_res =  np.sum(np.square( y_true-y_pred ))

    SS_tot = np.sum(np.square( y_true - np.mean(y_true) ) )

    return ( 1 - SS_res/(SS_tot) )
r_squared(y,pred)