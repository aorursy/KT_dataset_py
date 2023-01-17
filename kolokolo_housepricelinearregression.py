# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import *
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew #for some statistics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score




# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
#Which numeric Parameters are relevant ?
corrmat = train.corr()
f, ax = plt.subplots(figsize=(30, 15))
sns.heatmap(corrmat, vmax=.8, annot=True);
#Convert Categorial to numercical Data
train = pd.get_dummies(train, drop_first=True,dummy_na=True)
#Drop all too corellated numerical Features and all to less corellated numerical features

# Create correlation matrix
corr_matrix = train.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
to_drop_1 = [column for column in upper.columns if any(upper[column] > 0.9)]

# Drop features
train.drop(to_drop_1, axis=1, inplace=True)

# Find features with correlation less then than 0.02 to SalePrice
to_drop_2 = [column for column in upper.columns if (upper['SalePrice'][column] < 0.01)]

# Drop features
train.drop(to_drop_2, axis=1, inplace=True)

#Drop some outliars
train = train[train.LotFrontage < 250]
train = train[train.LotArea < 100000]
#tranform skewed features
skewness = train.apply(lambda x: skew(x))
skewness.sort_values(ascending=False)
skewness = skewness[abs(skewness)>0.5]
train[skewness.index] = np.log1p(train[skewness.index])
#Remove NaN Values
train = train.fillna(train.mean())

#Scale Features

#Remove Y
y = train[['SalePrice']]
train.drop(['SalePrice'], axis=1)
#Remove IdColumn
idColumn = train.iloc[:,0].to_numpy()
idColumn = np.expand_dims(idColumn, axis=1)
#Scale Features
scaler = StandardScaler()
train_toScale = train.iloc[:, 1:]
train_toScale = scaler.fit_transform(train_toScale)
train_scaled= np.hstack((idColumn,train_toScale))
#Replace Dataframe through scaled Dataframe
train = pd.DataFrame(data=train_scaled,columns = train.columns)

#Linear Regression

#Set Training and Test Set
X_train,X_test,y_train,y_test = train_test_split(train,y,test_size = 0.3,random_state= 0)
#Do Linear Regression
lr = LinearRegression()
lr.fit(X_train,y_train)

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

#Transform Test Data

#Convert Categorial to numercical Data
test = pd.get_dummies(test, drop_first=True,dummy_na=True)

#Drop Features
test.drop(to_drop_1, axis=1, inplace=True)
test.drop(to_drop_2, axis=1, inplace=True)
#Skeewness
test[skewness.index] = np.log1p(test[skewness.index])
#FillNull
test = test.fillna(test.mean())
#ScaleFeatures

#Remove IdColumn
idColumn = test.iloc[:,0].to_numpy()
idColumn = np.expand_dims(idColumn, axis=1)
#Scale Features
scaler = StandardScaler()
test_toScale = test.iloc[:, 1:]
test_toScale = scaler.fit_transform(test_toScale)
test_scaled= np.hstack((idColumn,test_toScale))
test = pd.DataFrame(data=train_scaled,columns = train.columns)
#Predict
test_prediction = lr.predict(test)
print(test_prediction)

