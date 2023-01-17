# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train=pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test=pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print(len(df_train.columns))

print(len(df_test.columns))
navalues= df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([navalues, percent], axis=1, keys=['navalues_train', 'Percent'])

missing_data.head(20)
#dropping values of train

df_train=df_train.drop(missing_data[missing_data['navalues_train']>1].index,1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

print(df_train.isnull().sum().max())

df_train.head(5)
df_train1=df_train[['MSSubClass','OverallQual','LotArea','OverallCond','YearBuilt','TotalBsmtSF','TotRmsAbvGrd','YrSold','SalePrice']]
corrmat = df_train1.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(df_train1[top_corr_features].corr(),annot=True,cmap="RdYlGn")
train=df_train1[['OverallQual','LotArea','YearBuilt','TotalBsmtSF','TotRmsAbvGrd','SalePrice']]

X = train.iloc[:, :-1].values

y = train.iloc[:, -1].values
testfeat=df_test[['OverallQual','LotArea','YearBuilt','TotalBsmtSF','TotRmsAbvGrd']]

testfeat.isnull().sum().sum()
navalues= testfeat.isnull().sum().sort_values(ascending=False)

print(navalues)
testfeat['TotalBsmtSF'].fillna((testfeat['TotalBsmtSF'].mean()), inplace=True)

testfeat.isnull().sum().sum()
Xtestfeat=testfeat.iloc[:, :].values
y=y.reshape(len(y),1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=1)
from sklearn.linear_model import LinearRegression

lm= LinearRegression()

lm.fit(X_train,y_train)
ypred=lm.predict(X_test)

np.set_printoptions(precision=2)

print(np.concatenate((ypred,y_test),1))
from sklearn.metrics import r2_score

r2_score(y_test, ypred)
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, ypred))

print('MSE:', metrics.mean_squared_error(y_test, ypred))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, ypred)))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=1)
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

sc_y = StandardScaler()

X_train = sc_X.fit_transform(X_train)

y_train = sc_y.fit_transform(y_train)
from sklearn.svm import SVR

SVregressor = SVR(kernel = 'rbf')

SVregressor.fit(X_train, y_train)
y_predSVM =sc_y.inverse_transform(SVregressor.predict(sc_X.transform(X_test)))

np.set_printoptions(precision=2)

print(np.concatenate((y_predSVM.reshape(len(y_predSVM),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import r2_score

r2_score(y_test, y_predSVM)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
from sklearn.tree import DecisionTreeRegressor

DTregressor = DecisionTreeRegressor(random_state = 1)

DTregressor.fit(X_train, y_train)
y_predDT = DTregressor.predict(X_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_predDT.reshape(len(y_predDT),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import r2_score

r2_score(y_test, y_predDT)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
from sklearn.ensemble import RandomForestRegressor

RFregressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

RFregressor.fit(X_train, y_train)
y_predRF = RFregressor.predict(X_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_predRF.reshape(len(y_predRF),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import r2_score

r2_score(y_test, y_predRF)
testpredSVM=sc_y.inverse_transform(SVregressor.predict(sc_X.transform(Xtestfeat)))

testpredSVM
testpred=pd.DataFrame(testpredSVM)

sub_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

datasets=pd.concat([sub_df['Id'],testpred],axis=1)

datasets.columns=['Id','SalePrice']

datasets.to_csv('sample_submission.csv',index=False)