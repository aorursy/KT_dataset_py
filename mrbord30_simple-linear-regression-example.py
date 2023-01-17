import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import xgboost as xgb

import sklearn.linear_model as linear_model

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_predict

from IPython.display import HTML, display

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

import scipy.stats as st

import os

%matplotlib inline



pd.options.display.max_rows = 1000

pd.options.display.max_columns = 20

#load train and test 

print(os.listdir("../input"))

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
#check sample data 

train.head(10)
#получить размеры набора данных обучения

print("train data shape", train.shape)

#получить размеры набора данных теста

print("test data shape", test.shape)



#колонки

print("\ncolumn in training data set\n\n",train.columns.values)

print("\ncolumn in testing data set\n\n",test.columns.values)



#Количественные и качественные переменные

quantitative = [f for f in train.columns if train.dtypes[f] != 'object']

quantitative.remove('SalePrice')

quantitative.remove('Id')

qualitative = [f for f in train.columns if train.dtypes[f] == 'object']



testID = test['Id']





#тренды 

for number in quantitative:

    g = sns.lmplot(x=number, y="SalePrice", data=train, line_kws={'color': 'yellow'})
plt.style.use(style='ggplot')

plt.rcParams['figure.figsize']=(10,6)



#проверка распределения целевой переменной

print("Распределение : ", train.SalePrice.skew())

plt.hist(train.SalePrice, color='blue')

plt.show()

#распределение немного искажено влево



#изменяем цену продажи на логарифм

print("Распределение после log: ", np.log(train.SalePrice).skew())

plt.hist(np.log(train.SalePrice), color='blue')

plt.show()

target= np.log(train.SalePrice)

#логарифмируем переменную целефую и получаем ее нормальное распределение
#вытаскиваем все числовые значения

num_features = train.select_dtypes(include=[np.number])
#корреляция числовых перемнных

corr = num_features.corr()



imp_coef = pd.concat([corr['SalePrice'].sort_values()])



plt.rcParams['figure.figsize'] = (6.0, 15.0)

imp_coef.plot(kind = "barh")
#проверить уникальные значения функции OverallQual

train.OverallQual.unique()
#проверка первой общей переменной с помощью SalePirce

qual_pivot = train.pivot_table(index='OverallQual', 

                               values='SalePrice', 

                               aggfunc=np.mean)

display(qual_pivot)



##график OverallQual и SalePrice

qual_pivot.plot(kind='bar', color='blue')

plt.xlabel('Overall Quality')

plt.ylabel('Sale Price')

plt.xticks(rotation=0)

plt.show()
#график Gr Living area и SalePrice

plt.scatter(x=train['GrLivArea'], y=target, color='green')

plt.ylabel('Sale Price')

plt.xlabel('GrLivArea')

plt.show()
plt.scatter(x=train['GarageArea'], y=target, color='green')

plt.ylabel('Sale Price')

plt.xlabel('Garage Area')

plt.show()
plt.scatter(x=train['TotalBsmtSF'], y=target, color='green')

plt.ylabel('Sale Price')

plt.xlabel('TotalBsmtSF')

plt.show()
nuls = pd.DataFrame(train.isnull().sum().sort_values(ascending =True)[:25])

nuls.columns = ['Null Count']

nuls.index.name = 'Feature'

nuls
print ("Unique values:", train.MiscFeature.unique())
catgr  = train.select_dtypes(exclude=[np.number])

catgr.describe()
print('originals')

print(train.Street.value_counts(),"\n")
#f = train.copy()

#ftest = test.copy()

#f = f.drop(["LotShape","LandSlope","MasVnr","GarageYrBlt", "Condition2", "RoofStyle"])

#ftest = ftest.drop(["LotShape","LandSlope","MasVnr","GarageYrBlt", "Condition2", "RoofStyle"])

#train = train.drop(columns =["LotShape","LandSlope","MasVnrType","GarageYrBlt","Condition2","RoofStyle"])

#test = test.drop(columns = ["LotShape","LandSlope","MasVnrType","GarageYrBlt","Condition2","RoofStyle"])
pd.options.display.max_rows = 1000

pd.options.display.max_columns = 20

pd.set_option('display.max_columns', 5000)



train = pd.get_dummies(train, drop_first=True)

test = pd.get_dummies(test, drop_first=True)

train.head(1)
print("Encoded:")

print(train.Street_Pave.value_counts())

#Garage car 

train.GarageCars.value_counts().plot(kind='bar', color='green')

plt.xlabel('Garage Car value')

plt.ylabel('Counts')

plt.xticks(rotation=0)

plt.show()
#update missing values 

train = train.fillna(train.mean())

test = test.fillna(test.mean())
#interpolate missing values 

dt = train.select_dtypes(include=[np.number]).interpolate().dropna()

#check if all cols have zero null values 

sum(dt.isnull().sum()!=0)
#change y to natural log 

y = np.log(train.SalePrice)

#drop original dependent var and id 

X = dt.drop(['Id','SalePrice'], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
#linear regression 

from sklearn import linear_model

lr = linear_model.LinearRegression(normalize=False)

model = lr.fit(X_train, y_train)

#r square 

print("R-Square : " ,model.score(X_test,y_test))

#rmse 

preds = model.predict(X_test)



from sklearn.metrics import mean_squared_error

print ('RMSE: ', mean_squared_error(y_test, preds))



plt.scatter(preds, y_test, alpha=.75, color='g')

plt.xlabel('predicted price')

plt.ylabel('actual sale price ')

plt.title('Linear regression ')

plt.show()
X_train_scale = pd.DataFrame(StandardScaler().fit_transform(X_train), columns = X_train.columns)

X_train_scale.set_index(X_train.index, inplace = True)

X_test_scale = pd.DataFrame(StandardScaler().fit_transform(X_test), columns = X_test.columns)

X_test_scale.set_index(X_test.index, inplace = True)
from sklearn import linear_model

lr = linear_model.LinearRegression()

model = lr.fit(X_train, y_train)

#r square 

#print("R-Square : " ,model.score(X_train_scale,y_test))

#rmse 

preds = model.predict(X_test_scale)



#print ('RMSE: ', mean_squared_error(y_test, preds))



plt.scatter(preds, y_test, alpha=.75, color='g')

plt.xlabel('predicted price')

plt.ylabel('actual sale price ')

plt.title('Linear regression ')

plt.show()
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score



rm = linear_model.Ridge(alpha=0.1)

ridge_model = rm.fit(X_train, y_train)

preds_ridge = ridge_model.predict(X_test)

plt.scatter(preds_ridge, y_test, alpha=.75, color='g')

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')

plt.title('Ridge Regularization with alpha = {}'.format(0.1))

overlay = 'R^2 is: {}\nRMSE is: {}'.format(ridge_model.score(X_test, y_test),

                                               mean_squared_error(y_test, preds_ridge))

plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')

plt.show()
l = linear_model.Lasso(alpha=0.1)

lasso_model = l.fit(X_train, y_train)

preds_lasso = lasso_model.predict(X_test)

plt.scatter(preds_lasso, y_test, alpha=.75, color='g')

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')

plt.title('Lasso Regularization with alpha = {}'.format(0.1))

overlay = 'R^2 is: {}\nRMSE is: {}'.format(lasso_model.score(X_test, y_test),

                                               mean_squared_error(y_test, preds_ridge))

plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')

plt.show()
f = X_train.copy()

ftest = X_test.copy()

f = f.drop(columns = ["LotShape_IR2","LotShape_IR3","LotShape_Reg","LandSlope_Mod","LandSlope_Sev", "GarageYrBlt"])

ftest = ftest.drop(columns = ["LotShape_IR2","LotShape_IR3","LotShape_Reg","LandSlope_Mod","LandSlope_Sev", "GarageYrBlt"])



l = linear_model.Lasso(alpha=0.1)

lasso_model = l.fit(f, y_train)

preds_lasso = lasso_model.predict(ftest)

plt.scatter(preds_lasso, y_test, alpha=.75, color='g')

plt.xlabel('Predicted Price')

plt.ylabel('Actual Price')

plt.title('Lasso Regularization with alpha = {}'.format(0.1))

overlay = 'R^2 is: {}\nRMSE is: {}'.format(lasso_model.score(ftest, y_test),

                                               mean_squared_error(y_test, preds_ridge))

plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')

plt.show()