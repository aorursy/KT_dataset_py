# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import sklearn.preprocessing as Prep

import sklearn.model_selection as mod

import sklearn.metrics as metrics
training_data = pd.read_csv('../input/train.csv')

testing_data = pd.read_csv('../input/test.csv')
print("training data shape is : "+ str(training_data.shape))

print("testing data shape is : "+ str(testing_data.shape))
training_data.info()

print('='*50)

testing_data.info()
y = training_data['SalePrice'].values
training_data.head()
# data  = pd.concat([training_data, testing_data])

data = pd.concat([training_data.drop(['SalePrice'], axis=1), testing_data], axis=0)
data.shape

# data.head()
data.drop('Id', axis=1,inplace=True)

# testing_data.drop(['Id'], axis=1)

data.shape
data.head()
# # from sklearn.preprocessing import Imputer

# # imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

# # imp.fit(training_data)

# # train = imp.transform(training_data)

for col in data:

    if data[col].dtype == 'object':

        data[col] = data[col].fillna(data[col].mode()[0])

    else:

        data[col] = data[col].fillna(data[col].median())
# numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']



# newdf = list(training_data.select_dtypes(include=numerics).columns.values)

# print(len(newdf))

data.shape
# from sklearn.preprocessing import  Imputer

# imputer = Imputer(strategy = 'median')



# imputer.fit(data)



# train_sample = imputer.transform(data)
print(data.isnull().sum().sum())
# data["Alley"]
X = pd.get_dummies(data)
X.head()
X.shape
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

result = scaler.fit_transform(X)
type(result)
brb2 = result[:training_data.shape[0]]

test_values = result[training_data.shape[0]:]
print(brb2.shape)

test_values.shape
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



features = X.iloc[:,0:289]  #independent columns

target = X.iloc[:,-1]    #target column i.e price range



bestfeatures = SelectKBest(score_func=chi2, k=10)

fit = bestfeatures.fit(features,target)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(features.columns)



featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(10,'Score'))
# import seaborn as sns

# import matplotlib.pyplot as plt

# corrmat = X.corr()

# top_corr_features = corrmat.index

# plt.figure(figsize=(80,80))

# g=sns.heatmap(X[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# train_label = X["SalePrice"]

# train_label.shape

# X
# from sklearn.model_selection import train_test_split



# X_train, X_test , y_train, y_test = train_test_split(X,train_label,train_size = 0.8)
# X_train.shape
# from sklearn.linear_model import LinearRegression



# model = LinearRegression()

# model.fit(X_train,y_train)
# y_preds = model.predict(X_test)
# from sklearn import metrics

# from sklearn.metrics import r2_score





# print("Root Mean square error: " , np.sqrt(metrics.mean_squared_error(y_test,y_preds)))

# print("Test acc: ", r2_score(y_test, y_preds))
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso





X_train, X_test, y_train, y_test = train_test_split(brb2, y, random_state=42)



# clf = LinearRegression()

clf = Lasso()



clf.fit(X_train, y_train)



y_pred = clf.predict(X_test)

y_train_pred = clf.predict(X_train)
from sklearn.metrics import r2_score



print("Train acc: " , r2_score(y_train, y_train_pred))

print("Test acc: ", r2_score(y_test, y_pred))
final_labels = clf.predict(test_values)
final_result = pd.DataFrame({'Id': testing_data['Id'], 'SalePrice': final_labels})

final_result.to_csv('house_price.csv', index=False)
print(os.listdir("../input"))
