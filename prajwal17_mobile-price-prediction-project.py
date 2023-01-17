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
import matplotlib.pyplot as plt
%matplotlib inline

data_train = pd.read_csv('../input/mobile-price-classification/train.csv')
data_test = pd.read_csv('../input/mobile-price-classification/train.csv')
data_train.info()
import seaborn as sns
data_train.isnull().sum()
data_train.describe()
data_train.corr()
#sns.pairplot(data_train,hue='price_range')

data_train.info()
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(data_train.corr(),cbar=True,annot=True,fmt='.1g',ax=ax)
### Shows a good positive corelation
sns.jointplot(data=data_train,y='ram',x='price_range',kind='hex')  ##"reg" | "resid" | "kde" | "hex"
from sklearn.ensemble import ExtraTreesRegressor
X= data_train.iloc[:,:-1]
y= data_train.iloc[:,-1]
model= ExtraTreesRegressor()
model.fit(X,y)
print(model.feature_importances_)
impcolumn =  pd.Series(model.feature_importances_,index = X.columns)
impcolumn.nlargest(7).plot(kind='barh')
plt.show()
#### we will gona consider all the Feature for building model
## Required Packages

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
### Hyperparameter selection
## No of trees in randomforest
n_estimator = [int(x) for x in np.linspace(100,1200,12)]

## max features---- no of features should consider in every split 
max_features = ['auto','sqrt']

##Max depth maximun number of depth in tree

max_depth = [int(x) for x in np.linspace(5,30,6)]

### min_sample leaf minimun no of sample required to split a node
min_samples_split = [2,5,10,15,100]

## min sample spli minimum no of sample required to split a leaf

min_samples_leaf = [1,2,5,10]
random_grid = {'n_estimators': n_estimator,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf}
rand_fst = RandomForestRegressor()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =.2,random_state =42)
rcv = RandomizedSearchCV(estimator = rand_fst,param_distributions=random_grid,scoring='neg_mean_squared_error',n_iter=10,cv=5
                        ,random_state=42,verbose=-1)
from sklearn.linear_model import LinearRegression
lr =LinearRegression()
lr.fit(X_train,y_train)
lr.score(X_test,y_test)
rand_fst.score(X_test,y_test)
one_hot_encoded_training_predictors = pd.get_dummies(data_train)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def get_mae(X, y):
    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
    return -1 * cross_val_score(RandomForestRegressor(50), 
                                X, y, 
                                scoring = 'neg_mean_absolute_error').mean()

predictors_without_categoricals = data_train.select_dtypes(exclude=['object'])

mae_without_categoricals = get_mae(predictors_without_categoricals, y)

mae_one_hot_encoded = get_mae(one_hot_encoded_training_predictors, y)

print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals)))
print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))
y_pred = rand_fst.predict(X_test)
plt.scatter(y_test,y_pred)
### our predicted distribution is Almost normal
sns.distplot(y_test-y_pred)
plt.plot(y_test,y_pred)
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
rcf =RandomForestClassifier()
rcf.fit(X_train,y_train)
rcf.score(X_test,y_test)
y_pred = rcf.predict(X_test)
print(classification_report(y_test,y_pred))
matrix=confusion_matrix(y_test,y_pred)
print(matrix)
plt.figure(figsize = (10,7))
sns.heatmap(matrix,annot=True)
