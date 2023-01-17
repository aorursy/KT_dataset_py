# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
## Importing Raw Data

df=pd.read_csv('../input/vehicle-dataset-from-cardekho/car data.csv')

df.head()
## Checking Shape of Data

df.shape
## Getting more info on Data

df.describe()
# Check missing / null values

df.isnull().sum()
## Removing Car Name

final_dataset=df[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',

       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]

final_dataset.head()
## Finding the age of the cars

final_dataset['Current Year']=2020

final_dataset['Ageing']=final_dataset['Current Year']-final_dataset['Year']

final_dataset.head()
## Removing Current Year and Selling Year

final_dataset.drop(['Year','Current Year'],axis=1,inplace=True)

final_dataset.head()
## One hot encoding

final_dataset=pd.get_dummies(final_dataset,drop_first=True)

final_dataset.head()
## Finding correlation

final_dataset.corr()
## Visualizing correlation

corrmat=final_dataset.corr()

top_corr_features=corrmat.index

plt.figure(figsize=(20,20))

#Plot Heatmap

g=sns.heatmap(final_dataset[top_corr_features].corr(),annot=True,cmap='RdYlGn')
## Splitting into Dependent and Independent features

X=final_dataset.iloc[:,1:]

y=final_dataset.iloc[:,0]
## Finding feature importance

model=ExtraTreesRegressor()

model.fit(X,y)

feature_importance=pd.Series(model.feature_importances_,index=X.columns)

feature_importance.sort_values().plot(kind='barh')
## Splitting into Test and Train



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
## Setting hyperparameters



# Number of trees in random forest

n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)]

# Number of features to consider at every split

max_features=['auto','sqrt']

# Maximum number of levels in tree

max_depth=[int(x) for x in np.linspace(5,30,num=6)]

# Minimum number of samples required to split a node

min_samples_split=[2,5,10,15,100]

# Minimum number of samples required at each leaf node

min_samples_leaf=[1,2,5,10]
## Create Random grid



random_grid={'n_estimators':n_estimators,

            'max_features':max_features,

            'max_depth': max_depth,

            'min_samples_split': min_samples_split,

            'min_samples_leaf':min_samples_leaf}

print(random_grid)
## Use Random grid to search for best hyperparameters

## First create the base model to tune

rf=RandomForestRegressor()

rf_random=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,scoring='neg_mean_squared_error',n_iter=10,cv=5,random_state=42,n_jobs=1)
rf_random.fit(X_train,y_train)
predictions=rf_random.predict(X_test)

predictions
sns.distplot(y_test-predictions)
plt.scatter(y_test,predictions)