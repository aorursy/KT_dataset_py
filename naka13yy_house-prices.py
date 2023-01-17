# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

%matplotlib inline

mpl.style.use( 'ggplot' )

sns.set_style( 'whitegrid' )

sns.set_palette("Set3")

pylab.rcParams[ 'figure.figsize' ] = 8 , 6



# machine learning

from sklearn import linear_model

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV



def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = 5))

    return(rmse)

#XGboost

import xgboost as xgb

#statmodelsのGLMを使えるようにするモジュール

import statsmodels.api as sm

#GLMの中で用いる統計処理関数のインポート, chi**2の値などの算出に使用している 

from scipy import stats 

#Rのglmを使用可能にするための関数

import statsmodels.formula.api as smf 



# ## OLS, GLM: Gaussian response data

"""def plot_distribution( df , var , target , **kwargs ):

    row = kwargs.get( 'row' , None )

    col = kwargs.get( 'col' , None )

    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )

    facet.map( sns.kdeplot , var , shade= True )

    facet.set( xlim=( 0 , df[ var ].max() ) )

    facet.add_legend()"""
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]
train_df.head(10)
train_df.columns
train_df.head()



pd.set_option('display.max_columns', 100)
train_df.info()

print('-'*40)

test_df.info()
train_df.describe()
train_df.describe(include=['O'])
sns.distplot(train_df['SalePrice'])
corr_df = pd.DataFrame(train_df.corr()['SalePrice'])

corr_df.sort_values(by='SalePrice', ascending=False)
X = train_df['OverallQual']

X = sm.add_constant(X)

y = train_df['SalePrice']

results = sm.OLS(y, X).fit()

print(results.summary())

sns.jointplot('OverallQual','SalePrice', data=train_df, kind='reg', size=8)
result_df = pd.DataFrame(index=[], columns=[])

stepwise = train_df

#stepwise = stepwise.drop(stepwise.dtypes == np.object)

stepwise = pd.get_dummies(stepwise)

stepwise = stepwise.fillna(stepwise.mean())

for column_name, col in stepwise.iteritems():

    X = np.array(col)

    X = sm.add_constant(X)

    y = stepwise['SalePrice'].as_matrix()

    results = sm.OLS(y, X).fit()

    results_list = [[column_name, results.fvalue, results.aic, results.rsquared, results.rsquared_adj]]

    result_df = result_df.append(results_list, ignore_index = True)



result_df.columns = ['variable', 'F-stat', 'AIC', 'R-squared', 'Adj. R-squared'] 

#result_df = result_df.drop(result_df[(result_df['Adj. R-squared']<0.4.index) 

result_df.sort_values(by='Adj. R-squared', ascending=False)
#sns.pairplot(train_df, vars=['OverallQual', 'SalePrice', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearRemodAdd'], kind="reg")
#sns.lmplot('OverallQual', 'SalePrice', train_df)

#sns.lmplot('GrLivArea', 'SalePrice', train_df)

#sns.lmplot('OverallQual', 'GrLivArea', train_df)
train_df2 = pd.DataFrame()

test_df2 = pd.DataFrame()

combine2 = [train_df2, test_df2]
#df_lf = train_df[['SalePrice','LotFrontage']]

#df_lf = df_lf.dropna()
#sns.pairplot(df_lf)
#plt.subplot(2,1,1)

#sns.boxplot(df_lf['LotFrontage'])

#plt.subplot(2,1,2)

#sns.distplot(df_lf['LotFrontage'])
#df_na = train_df.copy()

#df_nalf=df_na[df_na['LotFrontage'].isnull()]

#df_nalf['LotFrontage']=df_nalf['LotFrontage'].fillna(df_lf['LotFrontage'].mean())

#df_nalf
#g = sns.jointplot(x='LotFrontage', y='SalePrice',data=df_lf)

#g.x = df_nalf['LotFrontage']

#g.y = df_nalf['SalePrice']

#g.plot_joint(plt.scatter, marker='x', color='r')
#int_df = train_df.ix[:, train_df.dtypes == np.int64]

#float_df = train_df.ix[:, train_df.dtypes == np.float64]

#num_df = pd.concat([int_df, float_df], axis=1)
#num_df2 = num_df.drop("A", axis=1)

#num_df2 = num_df2.dropna()

#sns.pairplot(num_df2)
oq_df = train_df[['OverallQual', 'SalePrice']]
obj_df = train_df.loc[:, train_df.dtypes == np.object]

obj_df = obj_df.fillna("None")
df = pd.concat([obj_df, oq_df], axis=1)
#g = sns.FacetGrid(df, hue="MSZoning", size=4 ,aspect=5)

#g = g.map(sns.distplot, "SalePrice")

#g.add_legend()
for col in obj_df:

    g = sns.FacetGrid(df, hue=col, size=4 ,aspect=5)

    g = g.map(sns.kdeplot, "SalePrice", shade=True)

    g.add_legend()
for col in obj_df:

    sns.factorplot(x=col, y='SalePrice', data=df,  kind='box',aspect=3,size=6)

#    sns.factorplot(x=col, data=df,  kind='count',aspect=3,size=6)

#    sns.factorplot(x='OverallQual', y='SalePrice', data=df,  kind='violin', col=col, split=True, inner="quartile")

#    sns.factorplot(x='OverallQual', data=df,  kind='count', col=col)
train_df2['OverallQual'] = train_df['OverallQual']

test_df2['OverallQual'] = test_df['OverallQual']

train_df2['GrLivArea'] = train_df['GrLivArea']

test_df2['GrLivArea'] = test_df['GrLivArea']

train_df2['GarageArea'] = train_df['GarageArea']

test_df2['GarageArea'] = test_df['GarageArea']

train_df2['TotalBsmtSF'] = train_df['TotalBsmtSF']

test_df2['TotalBsmtSF'] = test_df['TotalBsmtSF']

train_df2['1stFlrSF'] = train_df['1stFlrSF']

test_df2['1stFlrSF'] = test_df['1stFlrSF']

train_df2['FullBath'] = train_df['FullBath']

test_df2['FullBath'] = test_df['FullBath']

train_df2['TotRmsAbvGrd'] = train_df['TotRmsAbvGrd']

test_df2['TotRmsAbvGrd'] = test_df['TotRmsAbvGrd']

train_df2['YearRemodAdd'] = train_df['YearRemodAdd']

test_df2['YearRemodAdd'] = test_df['YearRemodAdd']

combine2 = [train_df2, test_df2]
train_df2['MSZoning'] = train_df['MSZoning']

train_dummy = pd.get_dummies(train_df2[['MSZoning']]) 

train_df2 = train_df2.drop("MSZoning", axis=1)

test_df2['MSZoning'] = test_df['MSZoning']

test_dummy = pd.get_dummies(test_df2[['MSZoning']]) 

test_df2 = test_df2.drop("MSZoning", axis=1)

train_df2 = pd.merge(train_df2, train_dummy, left_index=True, right_index=True)

test_df2 = pd.merge(test_df2, test_dummy, left_index=True, right_index=True)

train_df2 = train_df2.drop("MSZoning_C (all)", axis=1)

test_df2 = test_df2.drop("MSZoning_C (all)", axis=1)



train_df2['Alley'] = train_df['Alley']

train_dummy = pd.get_dummies(train_df2[['Alley']]) 

train_df2 = train_df2.drop("Alley", axis=1)

test_df2['Alley'] = test_df['Alley']

test_dummy = pd.get_dummies(test_df2[['Alley']]) 

test_df2 = test_df2.drop("Alley", axis=1)

train_df2 = pd.merge(train_df2, train_dummy, left_index=True, right_index=True)

test_df2 = pd.merge(test_df2, test_dummy, left_index=True, right_index=True)

train_df2 = train_df2.drop("Alley_Pave", axis=1)

test_df2 = test_df2.drop("Alley_Pave", axis=1)



train_df2['MasVnrType'] = train_df['MasVnrType']

train_dummy = pd.get_dummies(train_df2[['MasVnrType']]) 

train_df2 = train_df2.drop("MasVnrType", axis=1)

test_df2['MasVnrType'] = test_df['MasVnrType']

test_dummy = pd.get_dummies(test_df2[['MasVnrType']]) 

test_df2 = test_df2.drop("MasVnrType", axis=1)

train_df2 = pd.merge(train_df2, train_dummy, left_index=True, right_index=True)

test_df2 = pd.merge(test_df2, test_dummy, left_index=True, right_index=True)

train_df2 = train_df2.drop("MasVnrType_None", axis=1)

test_df2 = test_df2.drop("MasVnrType_None", axis=1)



train_df2['ExterQual'] = train_df['ExterQual']

train_dummy = pd.get_dummies(train_df2[['ExterQual']]) 

train_df2 = train_df2.drop("ExterQual", axis=1)

test_df2['ExterQual'] = test_df['ExterQual']

test_dummy = pd.get_dummies(test_df2[['ExterQual']]) 

test_df2 = test_df2.drop("ExterQual", axis=1)

train_df2 = pd.merge(train_df2, train_dummy, left_index=True, right_index=True)

test_df2 = pd.merge(test_df2, test_dummy, left_index=True, right_index=True)

train_df2 = train_df2.drop("ExterQual_Ex", axis=1)

test_df2 = test_df2.drop("ExterQual_Ex", axis=1)



train_df2['BsmtQual'] = train_df['BsmtQual']

train_dummy = pd.get_dummies(train_df2[['BsmtQual']]) 

train_df2 = train_df2.drop("BsmtQual", axis=1)

test_df2['BsmtQual'] = test_df['BsmtQual']

test_dummy = pd.get_dummies(test_df2[['BsmtQual']]) 

test_df2 = test_df2.drop("BsmtQual", axis=1)

train_df2 = pd.merge(train_df2, train_dummy, left_index=True, right_index=True)

test_df2 = pd.merge(test_df2, test_dummy, left_index=True, right_index=True)

train_df2 = train_df2.drop("BsmtQual_Ex", axis=1)

test_df2 = test_df2.drop("BsmtQual_Ex", axis=1)



train_df2['CentralAir'] = train_df['CentralAir']

train_dummy = pd.get_dummies(train_df2[['CentralAir']]) 

train_df2 = train_df2.drop("CentralAir", axis=1)

test_df2['CentralAir'] = test_df['CentralAir']

test_dummy = pd.get_dummies(test_df2[['CentralAir']]) 

test_df2 = test_df2.drop("CentralAir", axis=1)

train_df2 = pd.merge(train_df2, train_dummy, left_index=True, right_index=True)

test_df2 = pd.merge(test_df2, test_dummy, left_index=True, right_index=True)

train_df2 = train_df2.drop("CentralAir_N", axis=1)

test_df2 = test_df2.drop("CentralAir_N", axis=1)



train_df2['KitchenQual'] = train_df['KitchenQual']

train_dummy = pd.get_dummies(train_df2[['KitchenQual']]) 

train_df2 = train_df2.drop("KitchenQual", axis=1)

test_df2['KitchenQual'] = test_df['KitchenQual']

test_dummy = pd.get_dummies(test_df2[['KitchenQual']]) 

test_df2 = test_df2.drop("KitchenQual", axis=1)

train_df2 = pd.merge(train_df2, train_dummy, left_index=True, right_index=True)

test_df2 = pd.merge(test_df2, test_dummy, left_index=True, right_index=True)

train_df2 = train_df2.drop("KitchenQual_Ex", axis=1)

test_df2 = test_df2.drop("KitchenQual_Ex", axis=1)



combine2 = [train_df2, test_df2]
train_df2.info()
test_df2.info()
train_df2['SalePrice'] = train_df['SalePrice']
train_int_df = train_df2.loc[:, train_df2.dtypes == np.int64]

train_uin_df = train_df2.loc[:, train_df2.dtypes == np.uint8]

train_float_df = train_df2.loc[:, train_df2.dtypes == np.float64]

test_int_df = test_df2.loc[:, test_df2.dtypes == np.int64]

test_uin_df = test_df2.loc[:, test_df2.dtypes == np.uint8]

test_float_df = test_df2.loc[:, test_df2.dtypes == np.float64]

#float_df = train_df2.loc[:, train_df.dtypes == np.float64]

#obj_df = train_df2.loc[:, train_df.dtypes == np.object]



#num_df = pd.concat([int_df, float_df], axis=1)
train_int_df = train_int_df.fillna(train_int_df.mean())

train_uin_df = train_uin_df.fillna(train_uin_df.mean())

train_float_df = train_float_df.fillna(train_float_df.mean())

test_int_df = test_int_df.fillna(test_int_df.mean())

test_uin_df = test_uin_df.fillna(test_uin_df.mean())

test_float_df = test_float_df.fillna(test_float_df.mean())
#train_df2 = pd.concat([train_int_df, train_uin_df, train_float_df], axis=1)

#test_df2 = pd.concat([test_int_df, test_uin_df, test_float_df], axis=1)

#combine2 = [train_df2, test_df2]
train_df2 = train_df2.fillna(train_df2.mean())

test_df2 = test_df2.fillna(test_df2.mean())

combine2 = [train_df2, test_df2]
train_df2 = train_df2.rename(columns={'1stFlrSF': 'FirstFlrSF'})

test_df2 = test_df2.rename(columns={'1stFlrSF': 'FirstFlrSF'})

combine2 = [train_df2, test_df2]

train_df2
train_df2.shape
clf = linear_model.LinearRegression()

 

# 説明変数に "SalePrice品質スコア以外すべて)" を利用

explanatory_variable = train_df2.drop("SalePrice", axis=1)

X = explanatory_variable.as_matrix()

 

# 目的変数に "SalePrice(品質スコア)" を利用

y = train_df2['SalePrice'].as_matrix()

 

# 予測モデルを作成

clf.fit(X, y)

 

# 偏回帰係数

print(pd.DataFrame({"Name":explanatory_variable.columns,"Coefficients":clf.coef_}).sort_values(by='Coefficients') )

 

# 切片 (誤差)

print(clf.intercept_)
# train_test_splitをインポート

from sklearn.cross_validation import train_test_split

# 70%を学習用、30%を検証用データにするよう分割

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 666)

# 学習用データでパラメータ推定

clf.fit(X_train, y_train)

# 作成したモデルから予測（学習用、検証用モデル使用）

y_train_pred = clf.predict(X_train)

y_test_pred = clf.predict(X_test)
# 学習用、検証用それぞれで残差をプロット

plt.scatter(y_train_pred, y_train_pred - y_train, c = 'blue', marker = 'o', label = 'Train Data')

plt.scatter(y_test_pred, y_test_pred - y_test, c = 'lightgreen', marker = 's', label = 'Test Data')

plt.xlabel('Predicted Values')

plt.ylabel('Residuals')

# 凡例を左上に表示

plt.legend(loc = 'upper left')

# y = 0に直線を引く

plt.hlines(y = 0, xmin = -100000, xmax = 500000, lw = 2, color = 'red')

#plt.xlim([10, 50])

plt.show()
clf = linear_model.LinearRegression(normalize=True)

 

# 説明変数に "SalePrice品質スコア以外すべて)" を利用

explanatory_variable = train_df2.drop("SalePrice", axis=1)

X = explanatory_variable.as_matrix()

 

# 目的変数に "SalePrice(品質スコア)" を利用

y = train_df2['SalePrice'].as_matrix()

 

# 予測モデルを作成

clf.fit(X, y)

 

# 偏回帰係数

print(pd.DataFrame({"Name":explanatory_variable.columns,"Coefficients":clf.coef_}).sort_values(by='Coefficients') )

 

# 切片 (誤差)

print(clf.intercept_)
# train_test_splitをインポート

from sklearn.cross_validation import train_test_split

# 70%を学習用、30%を検証用データにするよう分割

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 666)

# 学習用データでパラメータ推定

clf.fit(X_train, y_train)

# 作成したモデルから予測（学習用、検証用モデル使用）

y_train_pred = clf.predict(X_train)

y_test_pred = clf.predict(X_test)
# 学習用、検証用それぞれで残差をプロット

plt.scatter(y_train_pred, y_train_pred - y_train, c = 'blue', marker = 'o', label = 'Train Data')

plt.scatter(y_test_pred, y_test_pred - y_test, c = 'lightgreen', marker = 's', label = 'Test Data')

plt.xlabel('Predicted Values')

plt.ylabel('Residuals')

# 凡例を左上に表示

plt.legend(loc = 'upper left')

# y = 0に直線を引く

plt.hlines(y = 0, xmin = -100000, xmax = 500000, lw = 2, color = 'red')

#plt.xlim([10, 50])

plt.show()
#Y_pred = clf.predict(test_df2)
clf = linear_model.Ridge(alpha=1.0)
# 70%を学習用、30%を検証用データにするよう分割

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 666)

# 学習用データでパラメータ推定

clf.fit(X_train, y_train)

# 作成したモデルから予測（学習用、検証用モデル使用）

y_train_pred = clf.predict(X_train)

y_test_pred = clf.predict(X_test)
# 学習用、検証用それぞれで残差をプロット

plt.scatter(y_train_pred, y_train_pred - y_train, c = 'blue', marker = 'o', label = 'Train Data')

plt.scatter(y_test_pred, y_test_pred - y_test, c = 'lightgreen', marker = 's', label = 'Test Data')

plt.xlabel('Predicted Values')

plt.ylabel('Residuals')

# 凡例を左上に表示

plt.legend(loc = 'upper left')

# y = 0に直線を引く

plt.hlines(y = 0, xmin = -100000, xmax = 500000, lw = 2, color = 'red')

#plt.xlim([10, 50])

plt.show()
#model_ridge = Ridge()
#train_df2.l
alphas = [0.05, 0.075, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]

cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 

            for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation - Just Do It")

plt.xlabel("alpha")

plt.ylabel("rmse")
cv_ridge.values
reg = xgb.XGBRegressor()
# 70%を学習用、30%を検証用データにするよう分割

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state =666)
# ハイパーパラメータ探索

reg_cv = GridSearchCV(reg, {'max_depth': [2,4,6],'learning_rate':[0.025, 0.05, 0.075, 0.1],'n_estimators': [25,50,75,100]}, verbose=1)

reg_cv.fit(X_train, y_train)

print(reg_cv.best_params_, reg_cv.best_score_)
# 改めて最適パラメータで学習

reg = xgb.XGBRegressor(**reg_cv.best_params_)

reg.fit(X_train, y_train)
# 学習モデルの評価

pred_train = reg.predict(X_train)

pred_test = reg.predict(X_test)

print (np.sqrt(mean_squared_error(y_train, pred_train)))

print (np.sqrt(mean_squared_error(y_test, pred_test)))
# feature importance のプロット

importances = pd.Series(reg.feature_importances_, index = explanatory_variable.columns)

importances = importances.sort_values()

importances.plot(kind = "barh")

plt.title("imporance in the xgboost Model")

plt.show()
test_df2.columns
test_df2.shape
explanatory_variable.columns
explanatory_variable.shape
dtest = test_df2

"""dtest.ix[:,['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'FirstFlrSF',

       'FullBath', 'TotRmsAbvGrd', 'YearRemodAdd', 'MSZoning_FV',

       'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'Alley_Grvl',

       'MasVnrType_BrkCmn', 'MasVnrType_BrkFace', 'MasVnrType_Stone',

       'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA', 'BsmtQual_Fa',

       'BsmtQual_Gd', 'BsmtQual_TA', 'CentralAir_Y', 'KitchenQual_Fa',

       'KitchenQual_Gd', 'KitchenQual_TA']]"""
dtest.columns =  ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25']

Y_pred = reg.predict(dtest)
#線形回帰モデルの構築, 最小自乗法を使う場合

#modelの構築

ols_model = "SalePrice ~ OverallQual + GrLivArea + GarageArea + TotalBsmtSF + FirstFlrSF + FullBath + TotRmsAbvGrd + YearRemodAdd + MSZoning_FV + MSZoning_RH + MSZoning_RL + MSZoning_RM + Alley_Grvl + MasVnrType_BrkCmn + MasVnrType_BrkFace + MasVnrType_Stone + ExterQual_Fa + ExterQual_Gd + ExterQual_TA + BsmtQual_Fa + BsmtQual_Gd + BsmtQual_TA + KitchenQual_Fa + KitchenQual_Gd + KitchenQual_TA"

#モデルの構築

mod = smf.ols(formula=ols_model, data=train_df2)

#fitに回帰した結果が入っているので、これをresに代入する

res = mod.fit()

#結果の参照

res.summary()
#線形回帰モデルの構築, glmによるニュートン法を用いる場合

#modelの構築

glm_gauss_model = "SalePrice ~ OverallQual + GrLivArea + GarageArea + TotalBsmtSF + FirstFlrSF + FullBath + TotRmsAbvGrd + YearRemodAdd + MSZoning_FV + MSZoning_RH + MSZoning_RL + MSZoning_RM + Alley_Grvl + MasVnrType_BrkCmn + MasVnrType_BrkFace + MasVnrType_Stone + ExterQual_Fa + ExterQual_Gd + ExterQual_TA + BsmtQual_Fa + BsmtQual_Gd + BsmtQual_TA + KitchenQual_Fa + KitchenQual_Gd + KitchenQual_TA"

#モデルの構築

mod = smf.glm(formula=glm_gauss_model, data=train_df2 ,family=sm.families.Poisson())

#fitに回帰した結果が入っているので、これをresに代入する

res = mod.fit()

#結果の参照

res.summary()
sm.tools.eval_measures.rmse(y, np.array(res.predict(train_df2)))
formulas = [

    "SalePrice ~ OverallQual + GrLivArea + GarageArea + TotalBsmtSF + FirstFlrSF + FullBath + TotRmsAbvGrd + YearRemodAdd + MSZoning_FV + MSZoning_RH + MSZoning_RL + MSZoning_RM + Alley_Grvl + MasVnrType_BrkCmn + MasVnrType_BrkFace + MasVnrType_Stone + ExterQual_Fa + ExterQual_Gd + ExterQual_TA + BsmtQual_Fa + BsmtQual_Gd + BsmtQual_TA + KitchenQual_Fa + KitchenQual_Gd + KitchenQual_TA",

    "SalePrice ~ OverallQual + GrLivArea + GarageArea + TotalBsmtSF + FirstFlrSF + FullBath + TotRmsAbvGrd + YearRemodAdd + MSZoning_FV + MSZoning_RH + MSZoning_RL + MSZoning_RM + Alley_Grvl + MasVnrType_BrkCmn + MasVnrType_BrkFace + MasVnrType_Stone + ExterQual_Fa + ExterQual_Gd + ExterQual_TA + BsmtQual_Fa + BsmtQual_Gd + BsmtQual_TA",

    "SalePrice ~ OverallQual + GrLivArea + GarageArea + TotalBsmtSF + FirstFlrSF + FullBath + TotRmsAbvGrd + YearRemodAdd + MSZoning_FV + MSZoning_RH + MSZoning_RL + MSZoning_RM + Alley_Grvl + MasVnrType_BrkCmn + MasVnrType_BrkFace + MasVnrType_Stone + ExterQual_Fa + ExterQual_Gd + ExterQual_TA",

    "SalePrice ~ OverallQual + GrLivArea + GarageArea + TotalBsmtSF + FirstFlrSF + FullBath + TotRmsAbvGrd + YearRemodAdd + MSZoning_FV + MSZoning_RH + MSZoning_RL + MSZoning_RM + Alley_Grvl + MasVnrType_BrkCmn + MasVnrType_BrkFace + MasVnrType_Stone",

    "SalePrice ~ OverallQual + GrLivArea + GarageArea + TotalBsmtSF + FirstFlrSF + FullBath + TotRmsAbvGrd + YearRemodAdd + MSZoning_FV + MSZoning_RH + MSZoning_RL + MSZoning_RM + Alley_Grvl",

    "SalePrice ~ OverallQual + GrLivArea + GarageArea + TotalBsmtSF + FirstFlrSF + FullBath + TotRmsAbvGrd + YearRemodAdd + MSZoning_FV + MSZoning_RH + MSZoning_RL + MSZoning_RM",

    "SalePrice ~ OverallQual + GrLivArea + GarageArea + TotalBsmtSF + FirstFlrSF + FullBath + TotRmsAbvGrd + YearRemodAdd",

    "SalePrice ~ OverallQual + GrLivArea + GarageArea + TotalBsmtSF + FirstFlrSF + FullBath + TotRmsAbvGrd",

    "SalePrice ~ OverallQual + GrLivArea + GarageArea + TotalBsmtSF + FirstFlrSF + FullBath",

    "SalePrice ~ OverallQual + GrLivArea + GarageArea + TotalBsmtSF + FirstFlrSF",

    "SalePrice ~ OverallQual + GrLivArea + GarageArea + TotalBsmtSF",

    "SalePrice ~ OverallQual + GrLivArea + GarageArea",

    "SalePrice ~ OverallQual + GrLivArea"

]

steps = [[res.llf, res.aic, res.rsquared, res.rsquared_adj, formula]

         for (res, formula) in [(smf.ols(formula=formula, data=train_df2).fit(), formula) for formula in formulas]]

pd.DataFrame(steps, columns=['Log-Likelihood', 'AIC', 'R-squared', 'Adj. R-squared', 'formula'])
#Y_pred = res.predict(test_df2)
submission = pd.DataFrame({

        "Id": test_df["Id"],

        "SalePrice": Y_pred

    })

submission
submission.to_csv('submission.csv', index=False)