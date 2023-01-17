# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



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
df_train.head()
#Converting categorical as all are ordinal:

Score_map = {"Ex":5, "Gd":4, "TA":3, "Fa":2,"Po":1,"NA":0}

BsmtFin_map = {"GLQ":5, "ALQ":4, "BLQ":3, "Rec":2, "LwQ":1, "Unf":1, "NA":0}

BsmtEx_map= {"Gd":3, "Av":2, "Mn":1, "No":0, "NA":0}

df_train["BsmtFinType1"] = df_train.BsmtFinType1.replace(BsmtFin_map)

df_train["BsmtFinType2"] = df_train.BsmtFinType2.replace(BsmtFin_map)

df_train["BsmtExposure"]=df_train.BsmtExposure.replace(BsmtEx_map)

df_train["BsmtQual"]=df_train.BsmtQual.replace(Score_map)

df_train["BsmtCond"]=df_train.BsmtCond.replace(Score_map)

df_test["BsmtFinType1"]=df_test.BsmtFinType1.replace(BsmtFin_map)

df_test["BsmtFinType2"]=df_test.BsmtFinType2.replace(BsmtFin_map)

df_test["BsmtExposure"]=df_test.BsmtExposure.replace(BsmtEx_map)

df_test["BsmtQual"]=df_test.BsmtQual.replace(Score_map)

df_test["BsmtCond"]=df_test.BsmtCond.replace(Score_map)
basement_new= df_train[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1'

            , 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF','BsmtFinType2'

            , 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath','SalePrice']]
basement_new.head()
corrmat = basement_new.corr()

top_corr_features = corrmat.index

plt.figure(figsize=(20,20))

#plot heat map

g=sns.heatmap(basement_new[top_corr_features].corr(),annot=True,cmap="RdYlGn")
Score_mapex = {"Ex":5, "Gd":4, "TA":3, "Fa":2,"Po":1}

df_train["ExterQual"] = df_train.ExterQual.replace(Score_mapex)

df_train["ExterCond"] = df_train.ExterCond.replace(Score_mapex)

df_train["HeatingQC"] = df_train.HeatingQC.replace(Score_mapex)

df_train["KitchenQual"] = df_train.KitchenQual.replace(Score_mapex)

df_test["ExterQual"] = df_test.ExterQual.replace(Score_mapex)

df_test["ExterCond"] = df_test.ExterCond.replace(Score_mapex)

df_test["HeatingQC"] = df_test.HeatingQC.replace(Score_mapex)

df_test["KitchenQual"] = df_test.KitchenQual.replace(Score_mapex)
cor2=df_train[['ExterQual','ExterCond','BsmtFullBath','BsmtHalfBath','FullBath'

               ,'HalfBath','HeatingQC','KitchenQual','SalePrice']]

corrmat2 = cor2.corr()

top_corr_features = corrmat2.index

plt.figure(figsize=(18,18))

#plot heat map

g=sns.heatmap(cor2[top_corr_features].corr(),annot=True,cmap="RdYlGn")
Score_map = {"Ex":4, "Gd":3, "TA":2, "Fa":1,"NaN":0}

Score_mappool = {"Ex":5, "Gd":4, "TA":3, "Fa":2,"Po":1,"NaN":0}

df_train["FireplaceQu"] = df_train.FireplaceQu.replace(Score_map)

df_train["GarageQual"] = df_train.GarageQual.replace(Score_map)

df_train["GarageCond"] = df_train.GarageCond.replace(Score_map)

df_train["PoolQC"] = df_train.GarageCond.replace(Score_mappool)

df_test["FireplaceQu"] = df_test.FireplaceQu.replace(Score_map)

df_test["GarageQual"] = df_test.GarageQual.replace(Score_map)

df_test["GarageCond"] = df_test.GarageCond.replace(Score_map)

df_test["PoolQC"] = df_test.GarageCond.replace(Score_mappool)
cor3=df_train[['FireplaceQu','GarageQual','GarageCond','PoolQC','MiscVal'

               ,'WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','SalePrice']]
cor3=cor3.fillna(0)

cor3.head()
corrmat3 = cor3.corr()

top_corr_features = corrmat3.index

plt.figure(figsize=(18,18))

#plot heat map

g=sns.heatmap(cor3[top_corr_features].corr(),annot=True,cmap="RdYlGn")
train=df_train[['OverallQual','BsmtQual','BsmtExposure','BsmtFinType1','BsmtFinSF1'

                 ,'TotalBsmtSF','ExterQual','FullBath','HeatingQC','KitchenQual','WoodDeckSF','OpenPorchSF'

                 ,'LotArea','YearBuilt','TotalBsmtSF','TotRmsAbvGrd','SalePrice']]

train=train.fillna(0)

train.isnull().sum().sort_values(ascending=False)
X = train.iloc[:, :-1].values

y = train.iloc[:, -1].values
testfeat=df_test[['OverallQual','BsmtQual','BsmtExposure','BsmtFinType1','BsmtFinSF1'

                 ,'TotalBsmtSF','ExterQual','FullBath','HeatingQC','KitchenQual','WoodDeckSF','OpenPorchSF'

                 ,'LotArea','YearBuilt','TotalBsmtSF','TotRmsAbvGrd']]

testfeat=testfeat.fillna(0)
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
RandomForestRegressor(n_estimators=10, random_state=0)

y_predRF = RFregressor.predict(X_test)

np.set_printoptions(precision=2)

print(np.concatenate((y_predRF.reshape(len(y_predRF),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import r2_score

r2_score(y_test, y_predRF)
testpredRFM = RFregressor.predict(Xtestfeat)

testpredRFM
testpred=pd.DataFrame(testpredRFM)

sub_df=pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

datasets=pd.concat([sub_df['Id'],testpred],axis=1)

datasets.columns=['Id','SalePrice']

datasets.to_csv('NewSubmission.csv',index=False)