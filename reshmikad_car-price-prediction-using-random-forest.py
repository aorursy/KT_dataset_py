# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import classification_report,confusion_matrix

from sklearn import metrics

from sklearn.ensemble import ExtraTreesRegressor

import matplotlib.pyplot as plt
data =pd.read_csv('/kaggle/input/vehicle-dataset-from-cardekho/car data.csv')

data.shape
data.head()
data.isnull().sum()
data.describe()
data.columns
print(data.Fuel_Type.value_counts(),"\n")

print(data.Seller_Type.value_counts(),"\n")

print(data.Transmission.value_counts())
sns.barplot(data['Fuel_Type'],data['Selling_Price'],data=data,palette='summer')

sns.barplot(data['Seller_Type'],data['Selling_Price'],data=data,palette='twilight')
sns.barplot(data['Transmission'],data['Selling_Price'],data=data,palette='spring')
data=data.iloc[:,1:]

data.head()
data['This Year'] = 2020

data['no_year']=data['This Year']- data['Year']

data.drop(['Year'],axis=1,inplace=True)

data.drop(['This Year'],axis=1,inplace=True)
data.head()
plt.figure(figsize=(10,5))

sns.barplot('no_year','Selling_Price',data=data)
sns.heatmap(data.corr(),annot=True,cmap='summer')
data=pd.get_dummies(data,drop_first=True)
data.head()
sns.pairplot(data,diag_kind="kde", diag_kws=dict(shade=True, bw=.05, vertical=False))

plt.show()
X=data.iloc[:,1:]

y=data.iloc[:,0]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print("x train: ",X_train.shape)

print("x test: ",X_test.shape)

print("y train: ",y_train.shape)

print("y test: ",y_test.shape)
model = ExtraTreesRegressor()

model.fit(X,y)

print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.plot(kind='barh')

plt.show()
#Randomized Search CV



n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]

min_samples_split = [2, 5, 10, 15, 100]

min_samples_leaf = [1, 2, 5, 10]



r_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf}



print(r_grid)
regressor=RandomForestRegressor()



rf_random = RandomizedSearchCV(estimator = regressor, param_distributions = r_grid,scoring='neg_mean_squared_error', 

                               n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)



rf_random.fit(X_train,y_train)
predictions=rf_random.predict(X_test)
print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
fig = plt.figure()

sns.distplot((y_test - predictions), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)               

plt.xlabel('Errors', fontsize = 18)  
fig = plt.figure()

plt.scatter(y_test,predictions)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16) 
df = pd.DataFrame({'Actual':y_test,"Predicted":predictions})

df.head()
from sklearn.metrics import r2_score

R2 = r2_score(y_test,predictions)

R2