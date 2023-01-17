# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import skew

from sklearn.preprocessing import LabelBinarizer

from sklearn.impute import SimpleImputer

from pandas.plotting import scatter_matrix

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import f1_score , precision_score , recall_score , mean_squared_error 

from sklearn.model_selection import GridSearchCV , KFold , cross_val_score

from xgboost import XGBClassifier

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test= pd.read_csv("../input/test.csv")
train.head()
train.info()
train.drop(['Alley' , 'MiscFeature', 'Fence' , 'PoolQC'] , axis=1 , inplace=True)
train_labels = train['SalePrice']

#train.drop(['SalePrice'] , axis=1 , inplace=True)

All_data = pd.concat([train.loc[: , 'MSSubClass':'SaleCondition'] , test.loc[:,'MSSubClass':'SaleCondition']] , sort=False)



cat_data = list(train.select_dtypes(include='object').columns.values)

num_data = train._get_numeric_data().columns.drop(['Id','SalePrice'])
All_data.hist(bins=50 , figsize=(20,20))

plt.show()
fig = plt.figure(figsize=(12,6))

prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})

prices.hist()

plt.show()
skewed = All_data[num_data].apply(lambda x : skew(x.dropna()))

skewed = skewed[skewed>0.75]

#skewed = skewed.index

    

train_labels = np.log1p(train_labels)
#Filling NAN values with median values

# imputer = SimpleImputer(strategy="median")

# imputer.fit_transform(All_data[num_data])

All_data.fillna(All_data.mean() , inplace=True)

All_data[num_data] = np.log1p(All_data[num_data])



#Filling up NAN values in Categorical Columns

All_data[cat_data] = All_data[cat_data].bfill()
All_data.head()
corr = train.corr()

corr_mat = corr['SalePrice'].sort_values(ascending=False)

corr_mat.index[1:11]
fig = plt.figure(figsize=(18,36))

for i , cols in enumerate(num_data,1):

    plt.subplot(9,4 ,i)

    plt.scatter(x=train[cols] , y=train_labels)

    plt.xlabel(cols)

plt.show()

fig = plt.figure(figsize=(12,10))

sns.heatmap(corr , cmap=plt.cm.Reds)
import statsmodels.api as sm



train.fillna(train.mean() , inplace=True)

pmax = 1

cols = list(num_data)



while (len(cols)>0):

    

    p = []

    X_1 =  train[cols]

    X_1 = sm.add_constant(X_1)

    model = sm.OLS(train_labels , X_1).fit()

    p = pd.Series(model.pvalues.values[1:] , index=cols)

    pmax = max(p)

    feature_with_max_p = p.idxmax()

    if (pmax > 0.5):

        cols.remove(feature_with_max_p)

    else:

        break



print(cols )
training_data = All_data[cols][:train.shape[0]]

testing_data = All_data[cols][train.shape[0]:]
lin = LinearRegression(normalize=False)

lin.fit(train[cols] , train_labels)
np.expm1(lin.predict(testing_data[cols]))
dec = DecisionTreeRegressor()

dec.fit(training_data[cols] , train_labels)
np.expm1(dec.predict(testing_data[cols]))
scores = cross_val_score(dec , np.expm1(training_data[cols]) , np.expm1(train_labels) , cv=10 , scoring='neg_mean_squared_error')
rmse_score = np.sqrt(-scores)

rmse_score
from sklearn.ensemble import RandomForestRegressor



ran = RandomForestRegressor(n_estimators=100)

ran.fit(training_data[cols] , train_labels)
ran_scores = cross_val_score(ran , np.expm1(training_data[cols]) , np.expm1(train_labels) , cv=10 , scoring='neg_mean_squared_error')
ran_scores_rmse = np.sqrt(-ran_scores)

ran_scores_rmse
predict = np.expm1(ran.predict(testing_data[cols]))
xgb =  XGBClassifier(n_estimators=100)

xgb.fit(training_data[cols] , train_labels)
xgb_scores = cross_val_score(xgb , np.expm1(training_data[cols]) , np.expm1(train_labels) , cv=10 , scoring='neg_mean_squared_error')
predict = np.expm1(xgb.predict((testing_data[cols])))
data = { 'Id': test['Id'] , 'SalePrice':predict}

my_model = pd.DataFrame(data= data )
my_model.to_csv('submission.csv' , index=False)