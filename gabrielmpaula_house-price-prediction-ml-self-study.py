import numpy as np
import pandas as pd
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from xgboost import XGBRegressor

import os
print(os.listdir("../input"))

%matplotlib inline
train = pd.read_csv("../input/train.csv", index_col = 'Id')
be_predict = pd.read_csv("../input/test.csv", index_col = 'Id')
be_predict_id = be_predict.index.values.tolist()
#display dataset information
train.info()
be_predict.info()
train.describe(include=['O'])
be_predict.describe(include=['O'])
train.SalePrice.head(5), np.log1p(train.SalePrice.head(5))
#Spliting Data into different dtypes
#categorical = train_df.select_dtypes(include = ['object'])
#numeric = train_df.select_dtypes(include = ['int64','float64'])
#Correlation Matrix & HeatMap
corr_matrix = train.corr()
sns.heatmap(corr_matrix, vmax=.8, square=True); #HeatMap
#Top 10 Heat Map
k = 15 #number of variables for heatmap
cols = corr_matrix.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 5}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#With higher correlations against SalePrice, we must understand the behavior for each variable
sns.boxplot(x=train.OverallQual, y=train.SalePrice) #OverallQual, discret
#Removing Outliers not necessary
#GrLivArea, continuos
sns.jointplot(x=train['GrLivArea'], y=train['SalePrice'], kind='reg') #GrLivArea, continuos
#There is 2 outliers in our plot and we could remove it to get a higher pearson correlation
#Removing outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index).reset_index(drop=True)
sns.jointplot(x=train['GrLivArea'], y=train['SalePrice'], kind='reg') #0,02 higher in pearson correlation
#GarageCars, discret
sns.boxplot(x=train.GarageCars, y=train.SalePrice) #GarageCars, discret
# We see some houses with lower price and 4 garagecars, it does not make sense at all. We must count how many houses with 4 cars and remove as an outlier
len(train[train.GarageCars == 4]) #There are just 5 houses with 4 garagecars. Not relevant
#Removing outliers
train = train.drop(train[(train['GarageCars']==4) & (train['SalePrice']<300000)].index).reset_index(drop=True)
sns.boxplot(x=train.GarageCars, y=train.SalePrice) #GarageCars, discret
#GarageArea, continuos
sns.jointplot(x=train['GarageArea'], y=train['SalePrice'], kind='reg') #GrLivArea, continuos
#Removing outliers
train = train.drop(train[(train['GarageArea']>1200) & (train['SalePrice']<300000)].index).reset_index(drop=True)
sns.jointplot(x=train['GarageArea'], y=train['SalePrice'], kind='reg') #0,02 higher in pearson correlation
#TotalBsmtSF, continuos
sns.jointplot(x=train['TotalBsmtSF'], y=train['SalePrice'], kind='reg') #TotalBsmtSF, continuos
#1stFlrSF, continuos

sns.jointplot(x=train['1stFlrSF'], y=train['SalePrice'], kind='reg') #1stFlrSF, continuos

#FullBath, discret

sns.boxplot(x=train.FullBath, y=train.SalePrice) #FullBath, discret
#TotRmsAbvGrd, discret

sns.boxplot(x=train.TotRmsAbvGrd, y=train.SalePrice) #TotRmsAbvGrd, discret
len(train[train.TotRmsAbvGrd == 14]) #There is just 1 houses with 14 rooms. Not relevant
#Removing Outlier
train = train.drop(train[(train['TotRmsAbvGrd']==14)].index).reset_index(drop=True)
sns.boxplot(x=train.TotRmsAbvGrd, y=train.SalePrice) #TotRmsAbvGrd, discret
#New Correlation Matrix

corr_matrix2 = train.corr()
cols2 = corr_matrix2.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(train[cols2].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

corr_saleprice2 = corr_matrix2.iloc[:-1, -1:].sort_values(by = 'SalePrice',ascending = False)
#Combine Datasets

combine= train.append(be_predict, ignore_index = True)

#LotFrontage by Neighborhood median
sns.distplot(combine['LotFrontage'].dropna())
combine['LotFrontage'] = combine.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
#GarageYrBlt <> YearBuilt
combine['GarageYrBlt'].fillna(combine['YearBuilt'], inplace = True)

#Change NA to "None"
combine.update(combine[["Alley","Functional",'MasVnrType','PavedDrive','CentralAir',"BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu","GarageFinish","GarageCond","GarageQual","GarageType","PoolQC","Fence","MiscFeature"]].fillna("None"))
#Encoding Categorical Treatment

#Label Encoding

lb_make = LabelEncoder()

combine ['MSSubClass'] = combine ['MSSubClass'].apply(str)

combine["Street"] = lb_make.fit_transform(combine["Street"]) #Street
combine['Alley'] = combine['Alley'].map({"None": 0, "Pave": 2, "Grvl": 1}) #Alley
combine['LotShape'] = combine['LotShape'].map({"IR3": 1, "IR2": 2, "IR1": 3, "Reg": 4}) #LotShape
combine['LandSlope'] = combine['LandSlope'].map({"Sev": 1, "Mod": 2, "Gtl": 3}) #LandSlope
combine['ExterQual'] = combine['ExterQual'].map({"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}) #ExterQual
combine['ExterCond'] = combine['ExterCond'].map({"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}) #ExterCond
combine['BsmtQual'] = combine['BsmtQual'].map({"None": 0,"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}) #BsmtQual
combine['BsmtCond'] = combine['BsmtCond'].map({"None": 0,"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}) #BsmtCond
combine['FireplaceQu'] = combine['FireplaceQu'].map({"None": 0,"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}) #FireplaceQu
combine['BsmtExposure'] = combine['BsmtExposure'].map({"None": 0,"No": 1, "Mn": 2, "Av": 3, "Gd": 4}) #BsmtExposure
combine['BsmtFinType1'] = combine['BsmtFinType1'].map({"None": 0,"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}) #BsmtFinType1
combine['BsmtFinType2'] = combine['BsmtFinType2'].map({"None": 0,"Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}) #BsmtFinType2
combine['HeatingQC'] = combine['HeatingQC'].map({"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}) #HeatingQC
combine['KitchenQual'] = combine['KitchenQual'].map({"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}) #KitchenQual
combine['GarageFinish'] = combine['GarageFinish'].map({"None": 0, "Unf": 1, "RFn": 2, "Fin": 3}) #GarageFinish
combine['GarageCond'] = combine['GarageCond'].map({"None": 0,"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}) #GarageCond
combine['GarageQual'] = combine['GarageQual'].map({"None": 0,"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}) #GarageQual
combine['PoolQC'] = combine['PoolQC'].map({"None": 0, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}) #PoolQC
combine['Fence'] = combine['Fence'].map({"None": 0, "MnWw": 2, "GdWo": 3, "MnPrv": 4, "GdPrv": 5}) #Fence
combine['Functional'] = combine['Functional'].map({"None": 0, "Sal": 1, "Sev": 2, "Maj2": 3, "Maj1": 4, "Mod": 5, "Min2": 6, "Min1": 7,"Typ": 8}) #Functional
combine['PavedDrive'] = combine['PavedDrive'].map({"N": 0, "P": 1, "Y": 2}) #PavedDrive
combine['CentralAir'] = combine['CentralAir'].map({"N": 0, "Y": 1}) #CentralAir
combine["MSSubClass"] = combine["MSSubClass"].map({'180':1, '30':2, '45':2, '190':3, '50':3, '90':3, '85':4, '40':4, '160':4,'70':5, '20':5, '75':5, '80':5, '150':5, '120': 6, '60':6})
#combine ['MSSubClass'] = combine ['MSSubClass'].apply(str)
combine ['YrSold'] = combine ['YrSold'].apply(str)
combine ['MoSold'] = combine['MoSold'].apply(str)
combine.columns[combine.isna().any()].tolist()

combine.Electrical.fillna(combine['Electrical'].mode()[0], inplace = True) #Electrical
combine.MasVnrType.fillna(combine['MasVnrType'].mode()[0], inplace = True) #MasVnrType
combine.MasVnrArea.fillna(0,inplace = True) #MasVnrArea
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    combine[col] = combine[col].fillna(0)
  
combine.Exterior1st.fillna(combine['Exterior1st'].mode()[0], inplace = True) #Exterior1st
combine.Exterior2nd.fillna(combine['Exterior2nd'].mode()[0], inplace = True) #Exterior2nd
combine.GarageArea.fillna(0, inplace = True) #GarageArea
combine.GarageCars.fillna(0, inplace = True) #GarageCars
combine.KitchenQual.fillna(combine['KitchenQual'].mode()[0], inplace = True) #KitchenQual
combine['MSZoning'] = combine.groupby('Neighborhood')['MSZoning'].apply(lambda x:x.fillna(x.mode()[0])) #Fill MSZoning for mode in Neighborhood
combine.SaleType.fillna(combine.SaleType.mode()[0],inplace = True)
combine = combine.drop(['Utilities'], axis=1) #all data is AllPub which means that there is no difference

combine.columns[combine.isna().any()].tolist()

combine.columns[combine.isna().any()].tolist()
train = combine[:train.shape[0]]
be_predict2 = combine[train.shape[0]:]
train.columns[train.isna().any()].tolist()

be_predict2.columns[be_predict2.isna().any()].tolist()
#Fixing "skewed" features
train["SalePriceLog"] = np.log1p(train["SalePrice"])
sns.distplot(train['SalePriceLog'], fit=stats.norm)

#Numeric Features
combine_saleprice = combine.SalePrice
combine = combine.drop(columns="SalePrice")

numeric_feats = combine.dtypes[combine.dtypes != "object"].index
skewed_feats = combine[numeric_feats].apply(lambda x: stats.skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skewed Features' :skewed_feats})
skewness.head()
skewness = skewness[abs(skewness) > 0.75]

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    combine[feat] = boxcox1p(combine[feat], lam)
    combine[feat] += 1

combine["SalePrice"] = combine_saleprice
#Get Dummies

combine_d = pd.get_dummies(combine)
combine_d.shape
cols_to_drop = ["GarageCars","TotalBsmtSF","TotRmsAbvGrd","GarageYrBlt",'Condition2_PosA',
  'RoofMatl_Membran',
  'RoofMatl_Metal',
  'Condition2_RRAe',
  'Condition2_PosN',
  'Exterior1st_CBlock',
  'MiscFeature_TenC',
  'Exterior1st_ImStucc',
  'Exterior1st_Stone',
  'MiscFeature_Gar2',
  'RoofStyle_Shed',
  'Condition2_RRNn',
  'SaleCondition_AdjLand',
  'Condition2_RRAn',
  'Heating_Floor',
  'SaleType_CWD',
  'SaleType_Con',
  'Street',
  'Electrical_FuseP',
  'Electrical_Mix',
  'GarageType_2Types',
  'Heating_Grav',
  'Exterior2nd_CBlock',
  'Condition1_RRNe',
  'MiscFeature_Othr',
  'RoofMatl_WdShake']

combine_d = combine_d.drop(columns = cols_to_drop)
train2 = combine_d[:train.shape[0]]
be_predict_d = combine_d[train.shape[0]:]

train2.columns[train2.isna().any()].tolist()
combine.columns[combine.isna().any()].tolist()
from sklearn.model_selection import train_test_split

X = train2.loc[:,train2.columns != 'SalePrice'] 
y = train2 ['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
# Linear Regression

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
y_pred = regressor.predict(X_test)
print('Liner Regression R squared Train: %.4f' % regressor.score(X_train, y_train))

print('Liner Regression R squared Test: %.4f' % regressor.score(X_test, y_test))
#Ridge Regression

from sklearn.linear_model import Ridge
for alpha in (1, 3, 5, 8, 10, 100, 1000):
    ridge_model = Ridge(alpha = alpha)
    ridge_model.fit(X_train, y_train)
    y_pred = ridge_model.predict(X_test)
    print ('Alpha =', alpha)
    print('Ridge Regression R squared Train:' '%.4f' % ridge_model.score(X_train, y_train))
    print('Ridge Regression R squared Test:' '%.4f' % ridge_model.score(X_test, y_test))

ridge_model = Ridge(alpha = 10)
ridge_model.fit(X_train, y_train)  
ridge_prediction = ridge_model.predict(be_predict_d.loc[:,train2.columns != 'SalePrice'])
final = pd.DataFrame(list(zip(be_predict_id,ridge_prediction)), columns=['Id','SalePrice'])
final.to_csv('submission.csv',index=False)

final.head(5)
#Lasso Regression

from sklearn.linear_model import Lasso
for alpha in (20, 25, 30, 35, 40):
    lasso_model = Lasso(alpha = alpha, max_iter = 5000)
    lasso_model.fit(X_train, y_train)
    lasso_y = lasso_model.predict(X_test)
    print ('Alpha =', alpha)
    print('Lasso Regression R squared Train: %.4f' % lasso_model.score(X_train, y_train))
    print('Lasso Regression R squared Test: %.4f' % lasso_model.score(X_test, y_test))
    #print ('Number of features:' .format(np.sum(regressor.coef_!=0)))
lasso = Lasso()
lasso_params = {'alpha':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
lasso_grid = GridSearchCV(lasso, lasso_params, cv = 5, verbose = 10, scoring = 'neg_mean_absolute_error');
lasso_grid.fit(X_train, y_train);
lasso_score = lasso_grid.cv_results_
print(lasso_score)
print(lasso_grid.best_params_)
print(lasso_score['mean_test_score'])
print(lasso_score['params'])
ridge = Ridge()
ridge_params = {'alpha': [1, 10, 100, 1000]}
ridge_grid = GridSearchCV(ridge, ridge_params, cv = 5, verbose = 10, scoring = 'neg_mean_absolute_error')
ridge_grid.fit(X_train, y_train)
ridge_score = ridge_grid.cv_results_
print(ridge_grid.best_params_)
print(ridge_score['mean_test_score'])
print(ridge_score['params'])
'''
X_train.head(5)

cols_to_drop = ["GarageCars","TotalBsmtSF","TotRmsAbvGrd","GarageYrBlt",'Condition2_PosA',
  'RoofMatl_Membran',
  'RoofMatl_Metal',
  'Condition2_RRAe',
  'Condition2_PosN',
  'Exterior1st_CBlock',
  'MiscFeature_TenC',
  'Exterior1st_ImStucc',
  'Exterior1st_Stone',
  'MiscFeature_Gar2',
  'RoofStyle_Shed',
  'Condition2_RRNn',
  'SaleCondition_AdjLand',
  'Condition2_RRAn',
  'Heating_Floor',
  'SaleType_CWD',
  'SaleType_Con',
  'Street',
  'Electrical_FuseP',
  'Electrical_Mix',
  'GarageType_2Types',
  'Heating_Grav',
  'Exterior2nd_CBlock',
  'Condition1_RRNe',
  'MiscFeature_Othr',
  'RoofMatl_WdShake']

X_train_drop = X_train.drop(columns = cols_to_drop)
X_test_drop = X_test.drop(columns = cols_to_drop)
be_predict_d_drop = be_predict_d.drop(columns = cols_to_drop)
'''
xgboosting = XGBRegressor(n_estimators=5000, \
                          learning_rate=0.05, \
                          gamma=2, \
                          max_depth=12, \
                          min_child_weight=1, \
                          colsample_bytree=0.5, \
                          subsample=0.8, \
                          reg_alpha=1, \
                          objective='reg:linear', \
                          base_score = 7.76)

xgboosting.fit(X_train_drop, y_train)
xgb_test_y = xgboosting.predict(X_test_drop)
xgboosting.score(X_test_drop, y_test)
mean_absolute_error(xgb_test_y, y_test)
xgb_prediction = xgboosting.predict(be_predict_d.drop(columns="SalePrice"))
final = pd.DataFrame(list(zip(be_predict_id, xgb_prediction)), columns=['Id','SalePrice'])
final.to_csv('submission.csv',index=False)
final.head(10)
X_train_drop.shape
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import PReLU
from keras.layers import BatchNormalization
from keras.layers import Dropout

model = Sequential()

model.add(Dense(1000, activation='relu', input_dim = X_train_drop.shape[1], kernel_initializer = 'he_normal'))
#model.add(PReLU())
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(800, activation='relu', kernel_initializer = 'he_normal'))
#model.add(PReLU())
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(800, activation='relu', kernel_initializer = 'he_normal'))
#model.add(PReLU())
model.add(BatchNormalization())
model.add(Dropout(0.0))

#model.add(Dense(300, activation = 'relu', init = 'he_normal'))
#model.add(PReLU())
#model.add(BatchNormalization())
#model.add(Dropout(0.1))

#model.add(Dense(50, activation = 'relu', init = 'he_normal'))
#model.add(PReLU())
#model.add(BatchNormalization())
#model.add(Dropout(0.0))

model.add(Dense(1, kernel_initializer = 'he_normal'))

#optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
#optimizer = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
optimizer = keras.optimizers.Adadelta(lr=0.5, rho=0.95, epsilon=None, decay=0.0)
#optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#optimizer = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
#optimizer = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

model.compile(loss = 'mae', optimizer = optimizer)

model.fit(X_train_drop, y_train, validation_data=(X_test_drop,y_test), epochs=150, batch_size=50, verbose=0)
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import PReLU
from keras.layers import BatchNormalization
from keras.layers import Dropout

model = Sequential()

model.add(Dense(1000, activation='relu', input_dim = X_train.shape[1], kernel_initializer = 'he_normal'))
#model.add(PReLU())
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(800, activation='relu', kernel_initializer = 'he_normal'))
#model.add(PReLU())
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(800, activation='relu', kernel_initializer = 'he_normal'))
#model.add(PReLU())
model.add(BatchNormalization())
model.add(Dropout(0.0))

#model.add(Dense(300, activation = 'relu', init = 'he_normal'))
#model.add(PReLU())
#model.add(BatchNormalization())
#model.add(Dropout(0.1))

#model.add(Dense(50, activation = 'relu', init = 'he_normal'))
#model.add(PReLU())
#model.add(BatchNormalization())
#model.add(Dropout(0.0))

model.add(Dense(1, kernel_initializer = 'he_normal'))

#optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
#optimizer = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
optimizer = keras.optimizers.Adadelta(lr=0.5, rho=0.95, epsilon=None, decay=0.0)
#optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#optimizer = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
#optimizer = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

model.compile(loss = 'mae', optimizer = optimizer)

model.fit(X, y, epochs=150, batch_size=50, verbose=1)
nn_prediction = model.predict(be_predict_d.drop(columns="SalePrice"))[:,0]
final = pd.DataFrame(list(zip(be_predict_id, nn_prediction)), columns=['Id','SalePrice'])
final.to_csv('submission.csv',index=False)