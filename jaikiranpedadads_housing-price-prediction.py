import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.shape,test.shape
numerical_features = train.select_dtypes(exclude = 'object')
num_corr = numerical_features.corr()
sns.heatmap(num_corr.sort_values(by = ['SalePrice'], ascending = False).head(1),cmap = 'Reds')

plt.title('Correlation with Saleprice')

plt.figure(figsize = (15,10))
num_corr['SalePrice'].sort_values(ascending = False).head(10).to_frame()
plt.figure(figsize= (15,6))

plt.scatter(train['GrLivArea'],train['SalePrice'],color = 'green', alpha = 0.5)

plt.xticks(weight = 'bold')

plt.yticks(weight = 'bold')

plt.title('Saleprice Vs OverallQual', weight = 'bold', color = 'red', fontsize = 15)
sns.heatmap(train.isnull())
import missingno as msno
msno.matrix(all_data)
msno.bar(all_data)
all_data = pd.concat((train.loc[:,:'SaleCondition'],test.loc[:,:'SaleCondition']))
all_data.drop(['Id'], axis=1, inplace=True)
all_data.dropna(thresh = len(all_data)*0.9, inplace = True, axis = 1)
all_na =all_na.drop(all_na[all_na == 0].index).sort_values(ascending = False)
plt.figure(figsize= (15,10))

all_na.plot.barh(color = 'blue')
all_data_na = all_data[all_na.index]
all_data_na.shape
cat_na = all_data_na.select_dtypes(include = 'object')

num_na = all_data_na.select_dtypes(exclude = 'object')
num_na
all_data.GarageYrBlt=all_data.GarageYrBlt.fillna(1980)
num_na.drop(['GarageYrBlt'], axis = 1, inplace = True)
num_na.columns
for nf in num_na.columns:

    all_data[nf] = all_data[nf].fillna(0)
small_cat_na.index
for sf in small_cat_na.index:

    all_data[sf] = all_data[sf].fillna(method = 'ffill')
cat_na1 = all_data.select_dtypes(include = 'object')
for cn in cat_na1.columns:

    all_data[cn] = all_data[cn].fillna('None')
msno.bar(all_data)
all_data.head(3)
all_data['TotalArea'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF'] + all_data['GrLivArea'] + all_data['GarageArea']
all_data['Bathrooms'] = all_data['FullBath'] + all_data['HalfBath']*0.5
#all_data['Age'] = 2016 -all_data['YearBuilt']
all_data.select_dtypes(exclude = 'object').columns
all_data['MoSold'] = all_data['MoSold'].astype(str)
all_data= pd.get_dummies(all_data)
print("The shape of combinde data",all_data.shape)
from scipy.stats import skew
numeric_features = all_data.select_dtypes(exclude = 'object').columns
sk_f = all_data[numeric_features].apply(lambda x : skew(x.dropna()))
sk_f = sk_f[sk_f > 0.75]
sk_features = sk_f.index
all_data[sk_features] = np.log1p(all_data[sk_features])
plt.figure(figsize = (15,10))

plt.subplot(221)

sns.distplot(train['GrLivArea'])

plt.title("befor trnasformation")



plt.subplot(222)

sns.distplot(all_data['GrLivArea'], color = 'Red')

plt.title("after trnasformation")



plt.subplot(223)

sns.distplot(train['1stFlrSF'])

plt.title("before trnasformation")



plt.subplot(224)

sns.distplot(all_data['1stFlrSF'], color = 'Red')

plt.title("after trnasformation")
sns.boxplot(all_data['GrLivArea'],orient = 'v')
#it's skewed let's apply log transformation

train['SalePrice'] = np.log1p(train['SalePrice'])
sns.set_style('whitegrid')

plt.figure(figsize = (15,10))

plt.subplot(121)

plt.hist(train['SalePrice'],color = 'red')

plt.title("Before transformation")



plt.subplot(122)

plt.hist(np.log1p(train['SalePrice']),color = 'green')

plt.title("after transformation")
Train = all_data[:1460]

Test = all_data[1460:]
#pos = [1298,523, 297, 581, 1190, 1061, 635, 197,1328, 495, 583, 313, 335, 249, 706]

pos = [1298,523, 297]

#y.drop(y.index[pos], inplace=True)

Train.drop(Train.index[pos], inplace = True)
#pos = [1298,523, 297, 581, 1190, 1061, 635, 197,1328, 495, 583, 313, 335, 249, 706]

y = train['SalePrice']

pos = [1298,523, 297]

y.drop(y.index[pos], inplace=True)
X = test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(Train,y,test_size = 0.33,random_state = 45)
#scaling the input data for better results

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.fit_transform(x_test)

#X = scaler.fit_transform(X)
from sklearn.metrics import mean_squared_error

def build_model(model,X_train,X_test,Y_train,Y_test):

    model = model.fit(X_train,Y_train)

    y_predict_train = model.predict(X_train)

    y_predict_test = model.predict(X_test)

    rmse_test = np.sqrt(mean_squared_error(Y_test,y_predict_test))

    rmse_train = np.sqrt(mean_squared_error(Y_train,y_predict_train))

    print("Model RMSE on training set : {} \nModel RMSE on test set : {}".format(rmse_train,rmse_test))
#linear regression

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(Train,y)

MSEs = cross_val_score(lr,Train,y,scoring = 'neg_mean_squared_error', cv = 5)

np.sqrt(-np.mean(MSEs))
#Our goal is to reduce the RMSE. let's try Lasso, Ridge, ElasticNet
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge
parameters ={'alpha': [i for i in range(0,101)]}

rd = Ridge()
rd = GridSearchCV(rd,param_grid = parameters, scoring = 'neg_mean_squared_error',cv = 5)
rd.fit(x_train,y_train)
print(rd.best_params_)

print(math.sqrt(-rd.best_score_))
rd = Ridge(alpha = 4)
rd = rd.fit(Train,y)
preds = np.expm1(rf.predict(Test))
preds
sln = pd.DataFrame({'Id':test.Id,'SalePrice':preds})
sln.to_csv("housing_price_pred_Jaikiran2.csv", index = False)
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor()
params = {'learning_rate':[0.01,0.1,0.5,0.8,1,2,3,5,8],

         'n_estimators':[50,100,200,300],

         }
gb = GridSearchCV(gb, param_grid = params ,cv = 5)
gb.fit(Train,y)
gb.best_params_
gb = GradientBoostingRegressor(learning_rate= 0.1,n_estimators=200)
gb.fit(Train,y)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

parameters = {'n_estimators':[50,100,200,300,500,800],

              'max_depth':[3,5,10,15,20],

             }

rf = GridSearchCV(rf,param_grid =parameters ,cv = 5)
rf.fit(Train,y)
y_predict = rf.predict(Train)

np.sqrt(mean_squared_error(y,y_predict))
rf.best_params_
rf = RandomForestRegressor(n_estimators = 800,max_depth=15)
rf.fit(Train,y)