import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm,skew

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

print(df_train.shape) # 'shape' gives number of rows and columns in dataset

print(df_test.shape)

print(df_train.columns)

print(df_train.dtypes)
df_train.head()

df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice']);

#positive skewness
#skewness and kurtosis

print("Skewness: %f" % df_train['SalePrice'].skew()) # 0(Zero) for Normal distribution

print("Kurtosis: %f" % df_train['SalePrice'].kurt()) # 3 for Normal distribution
#standardizing data

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);

#https://stackoverflow.com/questions/29241056/how-does-numpy-newaxis-work-and-when-to-use-it

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)
#Plots to show relation between dependent(here SalePrice) and independent variables



def scatter_plot(var): #when independent variable is numerical 

    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));

    



def box_plot(var): #usually when independent variable is categorical 

    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

    f, ax = plt.subplots(figsize=(16, 8))

    fig = sns.boxplot(x=var, y="SalePrice", data=data)

    fig.axis(ymin=0, ymax=800000);

    plt.xticks(rotation=90);

    print(df_train[var].unique())

    print(df_test[var].unique())

    
box_plot('YearBuilt')
var = 'GrLivArea'

scatter_plot(var)

#linear relationship
#removing outliers

df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]


df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)

df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

df_train.shape
#correlation matrix

corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat,vmax=.8, square=True,cmap="YlGnBu");
#saleprice correlation matrix

def Corrmat(var):

    k = 10 #number of variables for heatmap

    cols = corrmat.nlargest(k, var)[var].index

    cm = np.corrcoef(df_train[cols].values.T)   #T: Transpose

    sns.set(font_scale=1.25)

    f, ax = plt.subplots(figsize=(12, 9))

    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols.values, xticklabels=cols.values,cmap="YlGnBu")

    #plt.show()

    

Corrmat('SalePrice')
#concat train and test

ntrain = df_train.shape[0] #used later to split all_data into train and test set again

ntest = df_test.shape[0]

y_train = df_train.SalePrice.values

all_data = pd.concat((df_train, df_test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))
all_total = all_data.isnull().sum().sort_values(ascending=False)

all_percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)

all_missing_data = pd.concat([all_total, all_percent], axis=1, keys=['Total', 'Percent'])

print(all_missing_data.head(35))



#dealing with missing data

all_data = all_data.drop((all_missing_data[all_missing_data['Total'] > 2000]).index,1)

#all_data.isnull().sum().max() #just checking that there's no missing data missing...

all_data.shape



"""

all_data = all_data.drop((train_missing_data[train_missing_data['Total'] > 1]).index,1)

all_data = all_data.drop(all_data.loc[all_data['Electrical'].isnull()].index)

all_data.isnull().sum().max() #just checking that there's no missing data missing...

"""
qualitative = [f for f in all_data.columns if all_data.dtypes[f] == 'object'] #non numerical attributes

quantitative = [f for f in all_data.columns if all_data.dtypes[f] != 'object']
qualitative
#plotting box plot between every qualitative attribute and SalePrice



def boxplot(x, y, **kwargs):

    sns.boxplot(x=x, y=y)

    x=plt.xticks(rotation=90)

    

f = pd.melt(df_train, id_vars=['SalePrice'], value_vars=qualitative)

#https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.melt.html



g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, size=5)

g = g.map(boxplot, "value", "SalePrice")
#incase you wish to see any specific attribute

#box_plot('FireplaceQu') 
#values of these attributes have almost no effect on SalePrice hence we remove them



all_data = all_data.drop(['Utilities','LotConfig','LandSlope','BsmtFinType1','BsmtFinType2'], axis=1)

all_data.shape
#replacing each missing categorical attribute its most frequent occuring entry



all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data["Functional"] = all_data["Functional"].fillna("Typ") #data description says NA means typical

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
from sklearn.preprocessing import LabelEncoder



cols = (  

        'ExterQual', 'ExterCond','HeatingQC', 'Electrical', 'KitchenQual', 'Functional',

        'LotShape', 'PavedDrive', 'Street', 'LandContour', 'CentralAir','FireplaceQu','GarageQual','GarageFinish',

'GarageCond','GarageType','BsmtCond','BsmtExposure','BsmtQual','MasVnrType')



# process columns, apply LabelEncoder to categorical features

for c in cols:

    all_data[c].replace(np.nan, 'NAN', inplace=True) #replace missing values with 'NAN'

    lbl=LabelEncoder()

    lbl.fit(list(all_data[c].values)) 

    all_data['en'+c] = lbl.transform(list(all_data[c].values))







# shape        

print('Shape all_data: {}'.format(all_data.shape))
"""

I haven't tried this out however, it seems a good shot

#Method 2:

# Encode some categorical features manually as ordered numbers

all_data = all_data.replace({"BldgType" : {"1Fam" : 2, "TwnhsE" : 2, "2fmCon" : 1, "Duplex" : 1, "Twnhs" : 1},

                       "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},

                       "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 

                                         "ALQ" : 5, "GLQ" : 6},

                       "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 

                                         "ALQ" : 5, "GLQ" : 6},

                       "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},

                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 

                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},

                       "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "BldgType" : {"1Fam" : 1, "2fmCon" : 2, "Duplex" : 3, "TwnhsE" : 4, "Twnhs" : 5},

                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},

                       "LandContour" : {"Lvl" : 1, "Bnk" : 2, "Low" : 3, "HLS" : 4},

                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},

                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},

                       "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},

                       "Street" : {"Grvl" : 1, "Pave" : 2},

                       "RoofMatl" : {"CompShg" : 1, "Metal" : 1,"Tar&Grv" : 1,"Roll" : 1,"WdShake" : 2,"Membran" : 2,

                                     "WdShngl" : 3},

                       }

                     )

"""
#see remianing missing values

all_data.isna().sum().sort_values(ascending=False).head(15)
quantitative #leaving out recently encoded attributes
#scatterplot

sns.set()

sns.pairplot(all_data[quantitative], height = 2.5,diag_kind='kde')

plt.show();



#p.s. This may take quite a time
#in case you wish to see relation with a particular variable

#scatter_plot('FullBath')
#values of these attributes have almost no effect on SalePrice hence we remove them



all_data = all_data.drop(['Id','LotFrontage','MasVnrArea','YrSold','MoSold','MiscVal'], axis=1)

all_data.shape
#handling null values of Numerical attributes

all_data=all_data.fillna(0) #replace all null values with 0

all_data.isna().sum().sort_values(ascending=False).head(15) #No null values remaining

all_data.plot.scatter(x='TotalBsmtSF',y='GrLivArea')
all_data.plot.scatter(x='TotalBsmtSF',y='BsmtFinSF2')
#to check the attribute values before adding

all_data[['GrLivArea','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF']].head() 
# Adding total sqfootage feature 

all_data['TotalSF'] = all_data['GrLivArea']+all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']-all_data['LowQualFinSF']



#all_data=all_data.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'],1)
# Adding misceleneous total sqfootage feature 

all_data['MiscTotalSF'] = all_data['LotArea']+all_data['PoolArea']+all_data['WoodDeckSF'] + all_data['OpenPorchSF'] + all_data['EnclosedPorch']+all_data['3SsnPorch'] + all_data['ScreenPorch'] + all_data['PoolArea']



#all_data=all_data.drop(['WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea'],1)
all_data[['OverallQual','OverallCond']].head()
# Adding overall quality and condition of house 

all_data['Overall'] = all_data['OverallQual'] + all_data['OverallCond'] 



#all_data=all_data.drop(['OverallQual','OverallCond'],1)
# Adding number of baths

all_data['Bath'] = all_data['BsmtFullBath'] + all_data['BsmtHalfBath'] +all_data['FullBath'] + all_data['HalfBath'] 



#all_data=all_data.drop(['BsmtFullBath','BsmtHalfBath','FullBath','HalfBath'],1)
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    #all_data[feat] += 1

    all_data[feat] = boxcox1p(all_data[feat], lam)

    

#all_data[skewed_features] = np.log1p(all_data[skewed_features])
all_data = pd.get_dummies(all_data)

print(all_data.shape)
#histogram and normal probability plot of dependent variable

sns.distplot(y_train, fit=norm);

fig = plt.figure()

res = stats.probplot(y_train, plot=plt)
y_train = np.log1p(y_train)
sns.distplot(y_train, fit=norm);

fig = plt.figure()

res = stats.probplot(y_train, plot=plt)
#splitting train and test sets

train = all_data[:ntrain]

test = all_data[ntrain:]

print(df_test.shape)

print(test.shape)
from sklearn.preprocessing import RobustScaler #handles outliers better than StandardScaler



sc = RobustScaler()

train = sc.fit_transform(train)

test = sc.transform(test)
"""

from sklearn.decomposition import PCA



pca = PCA(n_components=281,svd_solver='full')

train = pca.fit_transform(train)

test = pca.transform(test)

"""
#ONE Stacked Regressions

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV



from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
#split the training set for cross validation

#these variables will be used in parameter tuning of models

dx_train, dx_test, dy_train, dy_test = train_test_split(train, 

                                                    y_train, test_size=0.30, 

                                                    random_state=42)

#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train)

    rmse= np.sqrt(-cross_val_score(model, train, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
#deciding on parameters

test_score = []

train_score = []

alpha_values = [0.011,0.033,0.1,0.33,0.67,1]

for alpha in alpha_values:

    lasso = make_pipeline(RobustScaler(), Lasso(alpha =alpha, random_state=1))

    lasso.fit(dx_train,dy_train)

    train_score.append(lasso.score(dx_train,dy_train))

    test_score.append(lasso.score(dx_test,dy_test))

    

plt.figure(figsize = (8,8))   

plt.plot(alpha_values,train_score)

plt.plot(alpha_values, test_score)

plt.xlabel('Alpha')

plt.ylabel('Accuracy')

plt.legend(['Training set','Test set'])



lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))



score = rmsle_cv(lasso)

print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#Elastic Net

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#Kernel Ridge

#deciding on parameters



test_score = []

train_score = []



degree_values = [1,2,3]

#coef0_values = [0.0005,0.005,0.05,0.1,0.2,0.25,0.3,0.5]

#alpha_values=[0.0005,0.005,0.05,0.1,0.3,1]

for degree in degree_values:

    KRR = KernelRidge(alpha=alpha, kernel='polynomial', degree=degree, coef0=0.25)

    KRR.fit(dx_train,dy_train)

    train_score.append(KRR.score(dx_train,dy_train))

    test_score.append(KRR.score(dx_test,dy_test))

    

plt.figure(figsize = (8,8))   

plt.plot(degree_values,train_score)

plt.plot(degree_values, test_score)

plt.xlabel('Alpha')

plt.ylabel('Accuracy')

plt.legend(['Training set','Test set'])



KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)



score = rmsle_cv(KRR)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
"""Gradient Boosting Regression :

With huber loss that makes it robust to outliers"""

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

score = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#XGboost

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)



score = rmsle_cv(model_xgb)

print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#LightGBM

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



score = rmsle_cv(model_lgb)

print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
#let's look at the residuals as well:

#residual is the difference between the actual and predicted values

matplotlib.rcParams['figure.figsize'] = (6.0, 6.0)

model_xgb.fit(train, y_train)

preds = pd.DataFrame({"preds":model_xgb.predict(train), "true":y_train})

preds["residuals"] = preds["true"] - preds["preds"]

preds.plot(x = "preds", y = "residuals",kind = "scatter")


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # we define clones of the original models to fit the data in

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self

    

    #Now we do the predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1)   
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))



score = rmsle_cv(averaged_models)

print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
averaged_models1 = AveragingModels(models = (model_xgb, model_lgb))



score1 = rmsle_cv(averaged_models1)

print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score1.mean(), score1.std()))
averaged_models2 = AveragingModels(models = (ENet, GBoost, KRR, lasso, model_xgb, model_lgb))



score2 = rmsle_cv(averaged_models2)

print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score2.mean(), score2.std()))
#Stacking is an ensemble learning technique to combine multiple regression models via a meta-regressor.



class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds

   

    # We again fit the data on clones of the original models

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        

        # Train cloned base models then create out-of-fold predictions

        # that are needed to train the cloned meta-model

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(X[train_index], y[train_index])

                y_pred = instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

                

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

   

    #Do the predictions of all base models on the test data and use the averaged predictions as 

    #meta-features for the final prediction which is done by the meta-model

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),

                                                 meta_model = lasso)



score = rmsle_cv(stacked_averaged_models)

print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
#There'e also a library available for stacking models 



from mlxtend.regressor import StackingCVRegressor
"""

stack = StackingCVRegressor(regressors=(ENet, GBoost, KRR),

                            meta_regressor=lasso,

                            random_state=42)

score = rmsle_cv(stack)

print("Stack score: {:.4f} ({:.4f})".format(score.mean(), score.std()))

"""
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))

averaged_models.fit(train, y_train)

averaged_train_pred = averaged_models.predict(train)

averaged_pred = np.expm1(averaged_models.predict(test))

print(rmsle(y_train, averaged_train_pred))
averaged_models1.fit(train, y_train)

averaged_train_pred1 = averaged_models1.predict(train)

averaged_pred1 = np.expm1(averaged_models1.predict(test))

print(rmsle(y_train, averaged_train_pred1))
averaged_models2.fit(train, y_train)

averaged_train_pred2 = averaged_models2.predict(train)

averaged_pred2 = np.expm1(averaged_models2.predict(test))

print(rmsle(y_train, averaged_train_pred2))
stacked_averaged_models.fit(train, y_train)

stacked_train_pred = stacked_averaged_models.predict(train)

stacked_pred = np.expm1(stacked_averaged_models.predict(test))

print(rmsle(y_train, stacked_train_pred))
model_xgb.fit(train, y_train)

xgb_train_pred = model_xgb.predict(train)

xgb_pred = np.expm1(model_xgb.predict(test))

print(rmsle(y_train, xgb_train_pred))
model_lgb.fit(train, y_train)

lgb_train_pred = model_lgb.predict(train)

lgb_pred = np.expm1(model_lgb.predict(test))

print(rmsle(y_train, lgb_train_pred))
'''RMSE on the entire Train data when averaging'''



print('RMSLE score on train data:')

print(rmsle(y_train,(stacked_train_pred*0.70 + xgb_train_pred*0.15 + lgb_train_pred*0.15)))

print(rmsle(y_train,(stacked_train_pred + lgb_train_pred)/2))

print(rmsle(y_train,(stacked_train_pred + xgb_train_pred + lgb_train_pred)/3))

print(rmsle(y_train,(averaged_train_pred1*0.4+stacked_train_pred*0.6)))

print(rmsle(y_train,(averaged_train_pred1+averaged_train_pred2)/2))

print(rmsle(y_train,(averaged_train_pred1+averaged_train_pred2+xgb_train_pred+lgb_train_pred+stacked_train_pred)/5))

print(rmsle(y_train,(averaged_train_pred2+lgb_train_pred+stacked_train_pred)/3))

print(rmsle(y_train,(averaged_train_pred1+averaged_train_pred2+lgb_train_pred)/3))
#ensemble = (averaged_pred + lgb_pred)/2

#ensemble = (stacked_pred + lgb_pred)/2

ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15

#ensemble = (stacked_pred+ xgb_pred + lgb_pred)/3

#ensemble = (averaged_pred2+lgb_pred+stacked_pred)/3

#ensemble = (averaged_pred1*0.4+stacked_pred*0.6)

#ensemble = (averaged_pred1+averaged_pred2)/2

#ensemble = (averaged_pred1+averaged_pred2+xgb_pred+lgb_pred+stacked_pred)/5

#ensemble = (averaged_pred*0.25 + averaged_pred1*0.15 + averaged_pred2*0.10 + xgb_pred*0.1 + lgb_pred*0.1 + stacked_pred*0.3)

#ensemble=(averaged_pred1+averaged_pred2+lgb_pred)/3
sub = pd.DataFrame()

sub['Id'] = df_test.Id

sub['SalePrice'] = ensemble

sub.to_csv('SalePrice.csv',index=False)
output=pd.read_csv("SalePrice.csv")

output.head()
output.shape
sub.dtypes