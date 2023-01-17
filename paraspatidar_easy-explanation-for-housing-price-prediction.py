# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#For Plotting

import matplotlib.pyplot as plt

import seaborn as sns

#For DataProcessing

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler, LabelEncoder

#For Machine Learning

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/train.csv")
data.head()
data.info()
total_missing_data = data.isnull().sum().sort_values(ascending=False)

percent_missing_data = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total_missing_data,percent_missing_data],axis=1,keys=['Total','Percent'])

missing_data.head(10)
data = data.drop(["Alley","FireplaceQu","PoolQC","Fence","MiscFeature","LotFrontage"], axis=1)

data = data.drop(["GarageType","GarageYrBlt","GarageFinish","GarageQual","GarageCond"],axis = 1)

data = data.drop(["BsmtFinType2","BsmtExposure","BsmtFinType1","BsmtCond","BsmtQual"], axis = 1)

data = data.drop(["MasVnrType"], axis = 1)
data['MasVnrArea'].fillna(data['MasVnrArea'].median(), inplace = True)

data = data.fillna(data['Electrical'].value_counts().index[0])
total_missing_data = data.isnull().sum().sort_values(ascending = False)

percent_missing_data = (data.isnull().sum()/data.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total_missing_data, percent_missing_data], axis=1, keys=['Total', 'Percent'])

missing_data.head(3)
correlations = data.corr()

plt.figure(figsize=(15,15))

g = sns.heatmap(correlations,cbar = True, square = True, fmt= '.2f', annot_kws={'size': 15})
k = 10 

cols = correlations.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(data[cols].values.T)

sns.set(font_scale = 1)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 

                 yticklabels=cols.values, xticklabels=cols.values)

plt.show()
data.info()
categorical_data = data.select_dtypes(include=['object']).copy()

categorical_data_columns=categorical_data.columns

categorical_data_columns
label_encoder = LabelEncoder()



for i in categorical_data_columns:

    data[i] = label_encoder.fit_transform(data[i])
y = data['SalePrice']

X = data.drop(['SalePrice'],axis=1)
Scaler = StandardScaler()



X = pd.DataFrame(Scaler.fit_transform(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
from sklearn.ensemble import RandomForestRegressor

RFR = RandomForestRegressor(max_depth = 20, random_state = 0, n_estimators = 100)

RFR.fit(X_train,y_train)
predicted_prices = RFR.predict(X_test)
print ("Training score:",RFR.score(X_train,y_train),"Test Score:",RFR.score(X_test,y_test))
predicted_prices
my_submission = pd.DataFrame({'Id': X_test.index, 'SalePrice': predicted_prices})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)