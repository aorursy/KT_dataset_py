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
import math

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.metrics import accuracy_score

from sklearn.impute import SimpleImputer
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv").set_index('Id')

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv").set_index('Id')



y = train['SalePrice']

train = train.drop('SalePrice',axis=1)



display(train.head())

display(test.head())
X = pd.concat([train,test])

X.head()
X.dtypes
X.describe()
num_col = X.select_dtypes(include=['float64','int64']).columns

cat_col = X.select_dtypes(include=['object']).columns
plt.figure(figsize=(8,6))

correlation = X[num_col].corr()

sns.heatmap(correlation, mask = correlation <0.4, cmap='Blues')
X[num_col].isnull().sum()
sns.distplot(X.LotFrontage).set_title("LotFrontage Before Imputing")
imputer = SimpleImputer(strategy='median')

imputed = imputer.fit_transform(X[['LotFrontage']])



sns.distplot(imputed).set_title("LotFrontage After Median Imputing")
def replace_with_random(a):

    """

    a: Value or NaN to be replaced

    

    Cannot set a random state as it would generate the same value each time this function

    is called. This is unlikely to be the derired behaviour

    """    

    

    from random import randint

        

    if pd.isnull(a):

        return randint(20,100)

    else:

        return a
randimpute = X['LotFrontage'].apply(lambda a: replace_with_random(a))



sns.distplot(randimpute).set_title("LotFrontage After Random Imputing")
X['LotFrontage'] = randimpute
sns.distplot(X.MasVnrArea).set_title("MasVnrArea Before Imputing")
sns.distplot(X.GarageYrBlt).set_title("GarageYrBlt Before Imputing")
imputer = SimpleImputer(strategy='median')

X['BsmtFinSF1'] = imputer.fit_transform(X[['BsmtFinSF1']])
imputer = SimpleImputer(strategy='median')

X['BsmtFinSF1'] = imputer.fit_transform(X[['BsmtFinSF1']])
imputer = SimpleImputer(strategy='median')

X['BsmtFinSF2'] = imputer.fit_transform(X[['BsmtFinSF2']])
imputer = SimpleImputer(strategy='median')

X['BsmtUnfSF'] = imputer.fit_transform(X[['BsmtUnfSF']])
imputer = SimpleImputer(strategy='median')

X['TotalBsmtSF'] = imputer.fit_transform(X[['TotalBsmtSF']])
imputer = SimpleImputer(strategy='median')

X['BsmtFullBath'] = imputer.fit_transform(X[['BsmtFullBath']])
imputer = SimpleImputer(strategy='median')

X['BsmtHalfBath'] = imputer.fit_transform(X[['BsmtHalfBath']])
imputer = SimpleImputer(strategy='median')

X['GarageYrBlt'] = imputer.fit_transform(X[['GarageYrBlt']])
imputer = SimpleImputer(strategy='median')

X['GarageCars'] = imputer.fit_transform(X[['GarageCars']])
imputer = SimpleImputer(strategy='median')

X['GarageArea'] = imputer.fit_transform(X[['GarageArea']])
imputer = SimpleImputer(strategy='median')

X['MasVnrArea'] = imputer.fit_transform(X[['MasVnrArea']])
X[num_col].isnull().sum()
X.describe()
cat_col = X.select_dtypes(include=['object']).columns

X[cat_col]
X[cat_col].isnull().sum()
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer



# Preprocessing for numerical data

numerical_transformer = Pipeline(steps=[

    ('imputer',SimpleImputer(strategy='constant')),

    ('scaler',StandardScaler())

    ])



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

    ])



preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, num_col),

        ('cat', categorical_transformer, cat_col)

    ])
test = X.loc[test.index]

X = X.loc[train.index]

y = y.loc[train.index]



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.75,random_state=81)
# Import models

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegressionCV

from sklearn.linear_model import Perceptron

from sklearn.ensemble import AdaBoostClassifier
"""

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer



parameters = {'model__n_estimators':[100,200,500],

              'model__min_samples_split':[2],

              'model__min_samples_leaf':[1]}



scorer = make_scorer(accuracy_score,greater_is_better=True)



grid = GridSearchCV(pipeline,parameters,scoring=scorer)



grid.fit(X_train,y_train)



y_pred = grid.predict(X_test)



accuracy = accuracy_score(y_test,y_pred)



print("Accuracy:",accuracy)



#final_params = grid.best_params_



"""
#Train RF model model, I did a Grid Search CV on this, and it yielded the following setup of parameters:

RandomForest = RandomForestClassifier(n_estimators=500,

                                      min_samples_split=2,

                                      min_samples_leaf=1,

                                      random_state=81)



RF_pipeline = Pipeline(steps=[('preprocessor', preprocessor),('model', RandomForest)])



RF_pipeline.fit(X_train, y_train)



y_pred = RF_pipeline.predict(X_test)



RF_accuracy = accuracy_score(y_test,y_pred)



print("Accuracy:",RF_accuracy)
Perceptron = Perceptron()



Perc_pipeline = Pipeline(steps=[('preprocessor',preprocessor),('model',Perceptron)])



Perc_pipeline.fit(X_train,y_train)



y_pred = Perc_pipeline.predict(X_test)



Perceptron_accuracy = accuracy_score(y_test,y_pred)



print("Accuracy:",Perceptron_accuracy)
ADA = AdaBoostClassifier()



ADA_pipeline = Pipeline(steps=[('preprocessor',preprocessor),('model',ADA)])



ADA_pipeline.fit(X_train,y_train)



y_pred = ADA_pipeline.predict(X_test)



ADA_accuracy = accuracy_score(y_test,y_pred)



print("Accuracy:",ADA_accuracy)

results = pd.DataFrame({'Model':['Random Forest','Perceptron','ADA Boost'],

                        'Accuracy':[RF_accuracy, Perceptron_accuracy,ADA_accuracy]}).set_index('Model')
results.sort_values('Accuracy',ascending=False)
#Choosing RF pipeline, seems best

test_pred = RF_pipeline.predict(test)



submission = pd.DataFrame(test_pred,index=test.index,columns=['houseprices'])



submission.to_csv("./submission.csv")