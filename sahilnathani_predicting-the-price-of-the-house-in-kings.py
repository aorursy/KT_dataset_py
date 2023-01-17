import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/kc_house_data.csv", encoding='ISO-8859-1')
df.head()
df.info()
df.isnull().sum()
df.corr()['price']
df = df.drop(['id', 'date', 'zipcode'], axis=1)
df['how_old'] = [(2015-each) for each in df['yr_built']]
df.corr()['how_old']
df['log_price'] = [np.log(each) for each in df['price']]
df.corr()['log_price']
from sklearn.preprocessing import MinMaxScaler

#df['total_sqft'] = df['sqft_living'] + df['sqft_lot'] + df['sqft_above'] + df['sqft_basement']

df = df.drop(['sqft_lot', 'condition', 'long', 'log_price', 'view', 'yr_built', 'yr_renovated'], axis=1)
df.columns
df.corr()['price']
x = np.array(df.drop(['price'], 1))

y = np.array(df['price'])

y = y.reshape(-1, 1)



scaler = MinMaxScaler()

scaler.fit(x)

scaler.fit(y)

x = scaler.transform(x)

y = scaler.transform(y)





from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=47)
from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression



classifiers = [SVR(kernel='linear', C=1000, gamma=30), RandomForestRegressor(n_estimators=40, max_depth=None),

               BaggingRegressor(n_estimators=40), LinearRegression(normalize=True, n_jobs=-1)]

names = ['svm', 'rfr', 'br','lr']



for n, c in zip(names, classifiers):

    c.fit(x_train, y_train)

    score = c.score(x_test, y_test)

    print('The accuracy achieved by', n, 'is', score*100)
rfr = RandomForestRegressor(n_estimators=40, max_depth=None, min_samples_split=2, min_samples_leaf=1)

rfr.fit(x_train, y_train)

acc = rfr.score(x_test, y_test)



print('Accuracy achieved using Random Forest is: ', acc*100)
import os

from sklearn.tree import export_graphviz



x = 0

for tree_in_forest in rfr.estimators_:

    if(x<1):

        export_graphviz(tree_in_forest)

        os.system('dot -Tpng tree.dot -o tree.png')

    x+=1

    
plt.plot(history.history['mean_squared_error'])

plt.plot(history.history['mean_absolute_error'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['mean_squared', 'mean_absolute'], loc='upper left')

plt.show()
from sklearn.metrics import mean_squared_error



y_pred = model.predict(x_test)

y_pred = scaler.inverse_transform(y_pred)

y_pred = rfr.predict(x_test)

score = mean_squared_error(y_test, y_pred)