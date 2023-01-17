# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data= pd.read_csv("../input/kc_house_data.csv")

data.head()
data.info()
data.drop(['date','id'],axis=1,inplace=True)
data.columns
a4_dims= (12,8)

fig, ax = plt.subplots(figsize=a4_dims)

sns.heatmap(data.corr())

corr= data.corr()

columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] >= 0.9:

            if columns[j]:

                columns[j] = False

selected_columns = data.columns[columns]

data = data[selected_columns]
data.info()
sns.distplot(data['price'])
from sklearn.preprocessing import StandardScaler

scaler= StandardScaler()

scaler.fit(data.drop(['price'],axis=1))

scaled_features= scaler.transform(data.drop('price',axis=1))

df= pd.DataFrame(scaled_features,columns=data.columns[1:])

df.head()
sns.heatmap(data.corr())
corr= data.corr()

columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] >= 0.9:

            if columns[j]:

                columns[j] = False

selected_columns = data.columns[columns]

data = data[selected_columns]
df.info()
df.columns
sns.distplot(df['yr_built'],kde=False,color='darkred',bins=30)
X= df

y= data[ 'price']
from sklearn.linear_model import LinearRegression

from sklearn.neural_network import MLPRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics 

from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, 

                                                    random_state=101)
models=[]

models.append(('DTC',DecisionTreeRegressor()))

models.append(('KNC',KNeighborsRegressor()))

models.append(('LR',LinearRegression()))

models.append(('RFC',RandomForestRegressor()))

models.append(("MLP",MLPRegressor()))

models.append(("GBC",GradientBoostingRegressor()))
names=[]

for name,algo in models:

    algo.fit(X_train,y_train)

    prediction= algo.predict(X_test)

    a= metrics.mean_squared_error(y_test,prediction) 

    print("%s: %f "%(name, a))
rm= GradientBoostingRegressor(random_state=21, n_estimators=400)
rm.fit(X_train,y_train)
prediction = rm.predict(X_test)
print(prediction)
compare = pd.DataFrame({'Prediction': prediction, 'Test Data' : y_test})

compare.head(10)
sns.distplot((y_test-prediction),bins=50);
scores = cross_val_score(rm, X, y, cv=10)

print("Cross-validation scores: {}".format(scores))

print("Average cross-validation score: {:.2f}".format(scores.mean()))