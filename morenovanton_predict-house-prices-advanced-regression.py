import numpy as np 

import pandas as pd

from pandas.plotting import scatter_matrix

import seaborn as sns

from scipy.stats import skew



from sklearn.impute import SimpleImputer

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import Pipeline



#from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_log_error



from sklearn.linear_model import SGDRegressor

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet



from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor



from sklearn.model_selection import GridSearchCV 

from sklearn.model_selection import RandomizedSearchCV



import warnings

warnings.filterwarnings('ignore') # отключить все предупреждения



sns.set(rc={'figure.figsize': (8, 5)})
sample_submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
X_train = train.copy()

X_test = test.copy()



Y = train['SalePrice']

X_train = X_train.drop(['SalePrice'], axis=1)
Y.head()
Y.shape
X_train_num = X_train.select_dtypes(include = ['float64', 'int64'])

X_train_encoder = X_train.select_dtypes(include = ['object'])



X_test_num = X_test.select_dtypes(include = ['float64', 'int64'])

X_test_encoder = X_test.select_dtypes(include = ['object'])
sns.distplot(Y, kde_kws = {'color': 'gray', 'lw':1, 'label': 'Predictions' })
#Y = np.log1p(Y)

#sns.distplot(Y, kde_kws = {'color': 'gray', 'lw':1, 'label': 'Predictions' })
scaler_imput = Pipeline([

        ("imputer", SimpleImputer(strategy="mean")),

        ('RobustScaler', RobustScaler()),

        ("scaler", MinMaxScaler())

    ])
X_train_num_tr = pd.DataFrame(scaler_imput.fit_transform(X_train_num), columns=X_train_num.columns)

X_test_num_tr = pd.DataFrame(scaler_imput.transform(X_test_num), columns=X_test_num.columns)
print(X_train_num_tr.shape)

print(X_test_num_tr.shape)
def Pearson_correlation():

    cor_map = plt.cm.RdBu

    plt.figure(figsize=(15,17))

    plt.title('Pearson Correlation', y=1.05, size=15)

    sns.heatmap(X_train_num_tr.astype(float).corr(),linewidths=0.1,vmax=1.0, 

                square=True, cmap=cor_map, linecolor='white', annot=True) 
Pearson_correlation()
#skewed_feats_train = X_train_num_tr.apply(lambda x: skew(x.dropna()))

#skewed_feats_train = skewed_feats_train[skewed_feats_train > 0.75]

#skewed_feats_train = skewed_feats_train.index

#X_train_num_tr[skewed_feats_train] = np.log1p(X_train_num_tr[skewed_feats_train])
#skewed_feats_test = X_test_num_tr.apply(lambda x: skew(x.dropna()))

#skewed_feats_test = skewed_feats_test[skewed_feats_test > 0.75]

#skewed_feats_test = skewed_feats_test.index

#X_test_num_tr[skewed_feats_test] = np.log1p(X_test_num_tr[skewed_feats_test])
X_train_encoder.head(5)
X_train_encoder['SalePrice'] = Y 
sns.set(rc={'figure.figsize': (24, 10)})

sum_isnull_encodtrain = X_train_encoder.isnull().sum() 

sns.barplot(x=sum_isnull_encodtrain[sum_isnull_encodtrain > 0].index, 

            y=list(sum_isnull_encodtrain[sum_isnull_encodtrain > 0]))
sns.set(rc={'figure.figsize': (20, 5)})

Alley_bar = X_train_encoder.groupby('Alley').aggregate({'Alley': 'count'})

MasVnrType_bar = X_train_encoder.groupby('MasVnrType').aggregate({'MasVnrType': 'count'})



plt.subplot(141)

sns.barplot(x=Alley_bar.T.columns, y=Alley_bar['Alley'])

plt.subplot(142)

sns.barplot('Alley', 'SalePrice', data=X_train_encoder)



plt.subplot(143)

sns.barplot(x=MasVnrType_bar.T.columns, y=MasVnrType_bar['MasVnrType'])

plt.subplot(144)

sns.barplot('MasVnrType', 'SalePrice', data=X_train_encoder)
#X_train_encoder["Alley"] = X_train_encoder["Alley"].fillna("Grvl")

#X_train_encoder["MasVnrType"] = X_train_encoder["MasVnrType"].fillna("BrkFace")
sns.set(rc={'figure.figsize': (20, 12)})

BsmtQual_bar = X_train_encoder.groupby('BsmtQual').aggregate({'BsmtQual': 'count'})

BsmtCond_bar = X_train_encoder.groupby('BsmtCond').aggregate({'BsmtCond': 'count'})

BsmtExposure_bar = X_train_encoder.groupby('BsmtExposure').aggregate({'BsmtExposure': 'count'})

BsmtFinType1_bar = X_train_encoder.groupby('BsmtFinType1').aggregate({'BsmtFinType1': 'count'})

BsmtFinType2_bar = X_train_encoder.groupby('BsmtFinType2').aggregate({'BsmtFinType2': 'count'})

Electrical_bar = X_train_encoder.groupby('Electrical').aggregate({'Electrical': 'count'})



plt.subplot(341)

sns.barplot(x=BsmtQual_bar.T.columns, y=BsmtQual_bar['BsmtQual'])

plt.subplot(342)

sns.barplot('BsmtQual', 'SalePrice', data=X_train_encoder)



plt.subplot(343)

sns.barplot(x=BsmtCond_bar.T.columns, y=BsmtCond_bar['BsmtCond'])

plt.subplot(344)

sns.barplot('BsmtCond', 'SalePrice', data=X_train_encoder)





plt.subplot(345)

sns.barplot(x=BsmtExposure_bar.T.columns, y=BsmtExposure_bar['BsmtExposure'])

plt.subplot(346)

sns.barplot('BsmtExposure', 'SalePrice', data=X_train_encoder)





plt.subplot(347)

sns.barplot(x=BsmtFinType1_bar.T.columns, y=BsmtFinType1_bar['BsmtFinType1'])

plt.subplot(348)

sns.barplot('BsmtFinType1', 'SalePrice', data=X_train_encoder)
#X_train_encoder["BsmtQual"] = X_train_encoder["BsmtQual"].fillna("TA")

#X_train_encoder["BsmtCond"] = X_train_encoder["BsmtCond"].fillna("TA")

#X_train_encoder["BsmtExposure"] = X_train_encoder["BsmtExposure"].fillna("No")

#X_train_encoder["BsmtFinType1"] = X_train_encoder["BsmtFinType1"].fillna("Unf")
sns.set(rc={'figure.figsize': (20, 5)})

plt.subplot(141)

sns.barplot(x=BsmtFinType2_bar.T.columns, y=BsmtFinType2_bar['BsmtFinType2'])

plt.subplot(142)

sns.barplot('BsmtFinType2', 'SalePrice', data=X_train_encoder)



plt.subplot(143)

sns.barplot(x=Electrical_bar.T.columns, y=Electrical_bar['Electrical'])

plt.subplot(144)

sns.barplot('Electrical', 'SalePrice', data=X_train_encoder)
#X_train_encoder["BsmtFinType2"] = X_train_encoder["BsmtFinType2"].fillna("Unf")

#X_train_encoder["Electrical"] = X_train_encoder["Electrical"].fillna("SBrkr")
sns.set(rc={'figure.figsize': (20, 10)})

GarageType_bar = X_train_encoder.groupby('GarageType').aggregate({'GarageType': 'count'})

GarageFinish_bar = X_train_encoder.groupby('GarageFinish').aggregate({'GarageFinish': 'count'})

GarageQual_bar = X_train_encoder.groupby('GarageQual').aggregate({'GarageQual': 'count'})

GarageCond_bar = X_train_encoder.groupby('GarageCond').aggregate({'GarageCond': 'count'})





plt.subplot(241)

sns.barplot(x=GarageType_bar.T.columns, y=GarageType_bar['GarageType'])

plt.subplot(242)

sns.barplot('GarageType', 'SalePrice', data=X_train_encoder)



plt.subplot(243)

sns.barplot(x=GarageFinish_bar.T.columns, y=GarageFinish_bar['GarageFinish'])

plt.subplot(244)

sns.barplot('GarageFinish', 'SalePrice', data=X_train_encoder)



plt.subplot(245)

sns.barplot(x=GarageQual_bar.T.columns, y=GarageQual_bar['GarageQual'])

plt.subplot(246)

sns.barplot('GarageQual', 'SalePrice', data=X_train_encoder)



plt.subplot(247)

sns.barplot(x=GarageCond_bar.T.columns, y=GarageCond_bar['GarageCond'])

plt.subplot(248)

sns.barplot('GarageCond', 'SalePrice', data=X_train_encoder)
#X_train_encoder["GarageType"] = X_train_encoder["GarageType"].fillna("Attchd")

#X_train_encoder["GarageFinish"] = X_train_encoder["GarageFinish"].fillna("Unf")

#X_train_encoder["GarageQual"] = X_train_encoder["GarageQual"].fillna("TA")

#X_train_encoder["GarageCond"] = X_train_encoder["GarageCond"].fillna("TA")
sns.set(rc={'figure.figsize': (18, 10)})

PoolQC_bar = X_train_encoder.groupby('PoolQC').aggregate({'PoolQC': 'count'})

Fence_bar = X_train_encoder.groupby('Fence').aggregate({'Fence': 'count'})

MiscFeature_bar = X_train_encoder.groupby('MiscFeature').aggregate({'MiscFeature': 'count'})

FireplaceQu_bar = X_train_encoder.groupby('FireplaceQu').aggregate({'FireplaceQu': 'count'})





plt.subplot(241)

sns.barplot(x=PoolQC_bar.T.columns, y=PoolQC_bar['PoolQC'])

plt.subplot(242)

sns.barplot('PoolQC', 'SalePrice', data=X_train_encoder)



plt.subplot(243)

sns.barplot(x=Fence_bar.T.columns, y=Fence_bar['Fence'])

plt.subplot(244)

sns.barplot('Fence', 'SalePrice', data=X_train_encoder)



plt.subplot(245)

sns.barplot(x=MiscFeature_bar.T.columns, y=MiscFeature_bar['MiscFeature'])

plt.subplot(246)

sns.barplot('MiscFeature', 'SalePrice', data=X_train_encoder)



plt.subplot(247)

sns.barplot(x=FireplaceQu_bar.T.columns, y=FireplaceQu_bar['FireplaceQu'])

plt.subplot(248)

sns.barplot('FireplaceQu', 'SalePrice', data=X_train_encoder)
#X_train_encoder["PoolQC"] = X_train_encoder["PoolQC"].fillna("Gd")

#X_train_encoder["Fence"] = X_train_encoder["Fence"].fillna("MnPrv")

#X_train_encoder["MiscFeature"] = X_train_encoder["MiscFeature"].fillna("Shed")

#X_train_encoder["FireplaceQu"] = X_train_encoder["FireplaceQu"].fillna("Gd")
X_train_encoder_cetegor_tr = pd.get_dummies(X_train_encoder)

X_test_encoder_cetegor_tr = pd.get_dummies(X_test_encoder)
print(X_train_encoder_cetegor_tr.shape)

print(X_test_encoder_cetegor_tr.shape)
# delete columns from train which are not in test

colums_X_train_encoder_cetegor_tr = list(X_train_encoder_cetegor_tr)

colums_X_test_encoder_cetegor_tr = list(X_test_encoder_cetegor_tr)



mas_colums_ismissing_in = []



for i in colums_X_train_encoder_cetegor_tr:

    if i not in colums_X_test_encoder_cetegor_tr:

        mas_colums_ismissing_in.append(i)

        

X_train_encoder_cetegor_tr = X_train_encoder_cetegor_tr.drop(mas_colums_ismissing_in, axis=1)
print(X_train_encoder_cetegor_tr.shape)

print(X_test_encoder_cetegor_tr.shape)
X_train_num_tr = X_train_num_tr.reset_index()

X_train_encoder_cetegor_tr = X_train_encoder_cetegor_tr.reset_index()



X_test_num_tr = X_test_num_tr.reset_index()

X_test_encoder_cetegor_tr = X_test_encoder_cetegor_tr.reset_index()
X_train_new_df = pd.merge(X_train_num_tr, X_train_encoder_cetegor_tr)

X_test_new_df = pd.merge(X_test_num_tr, X_test_encoder_cetegor_tr)
X_train_new_df = X_train_new_df.drop(['index', 'Id'], axis=1)

X_test_new_df = X_test_new_df.drop(['index', 'Id'], axis=1)
print(X_train_new_df.shape) 

print(X_test_new_df.shape)
sns.set(rc={'figure.figsize': (8, 5)})

X_train_new_df['SalePrice'] = Y

sns.scatterplot(x='GrLivArea', y='SalePrice', data=X_train_new_df) 
X_train_new_df = X_train_new_df.drop(X_train_new_df[(X_train_new_df['GrLivArea']>0.8) & 

                                                    (X_train_new_df['SalePrice']<200000)].index)
sns.scatterplot(x='GrLivArea', y='SalePrice', data=X_train_new_df)  
Y = X_train_new_df['SalePrice']

X_train_new_df = X_train_new_df.drop(['SalePrice'], axis=1)
def plot_learning_curves(model, X, y):

    

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)

    train_errors, val_errors = [], [] 

    

    for m in range(1, len(X_train)):

        

        model.fit(X_train[:m], y_train[:m]) 

        y_train_predict = abs(model.predict(X_train[:m])) 

        y_val_predict = abs(model.predict(X_val)) 

    

        train_errors.append(mean_squared_log_error(y_train[:m], y_train_predict))

        val_errors.append(mean_squared_log_error(y_val, y_val_predict))



    plt.plot(np.sqrt(train_errors), "r-", linewidth=1, label="train")

    plt.plot(np.sqrt(val_errors), "b-", linewidth=1, label="val")

    plt.legend(loc="upper right", fontsize=14)   # not shown in the book

    plt.xlabel("Training set size", fontsize=14) # not shown

    plt.ylabel("rmsle", fontsize=14)              # not shown
def plot_SaleYandpredY(predict_X_test):

    sns.kdeplot(Y, label = 'train')

    sns.kdeplot(predict_X_test, label = 'test')
def train_validate_test_split(df, train_percent=0.14, seed=None):

    np.random.seed(seed)

    perm = np.random.permutation(df.index) # Произвольно переставить последовательность или вернуть переставленный диапазон.

    m = len(df.index)

    

    train_linREG_end = int(train_percent * m)  # 0.6 *10

    train_SGD_end = int(train_percent * m) + train_linREG_end

    train_Poly_end = int(train_percent * m) + train_SGD_end

    train_Ridge_end = int(train_percent * m) + train_Poly_end

    train_Lasso_end = int(train_percent * m) + train_Ridge_end

    train_Enet_end = int(train_percent * m) + train_Lasso_end

    

    train_linREG = df.ix[perm[:train_linREG_end]] 

    train_SGD = df.ix[perm[train_linREG_end:train_SGD_end]] 

    train_Poly = df.ix[perm[train_SGD_end:train_Poly_end]] 

    train_Ridge = df.ix[perm[train_Poly_end:train_Ridge_end]] 

    train_Lasso = df.ix[perm[train_Ridge_end:train_Lasso_end]] 

    train_Enet = df.ix[perm[train_Lasso_end:train_Enet_end]] 

    train_RF = df.ix[perm[train_Enet_end:]]

    

    return train_linREG, train_SGD, train_Poly, train_Ridge, train_Lasso, train_Enet, train_RF
class SklearnHelper(object):

    def __init__(self, clf, params=None):

        if params:

            self.clf = clf(**params)

        else:

            self.clf = clf



    def predict(self, x):

        return self.clf.predict(x)

    

    def fit(self,x,y):

        return self.clf.fit(x,y)

    

    def RandomizedSearchCV(self, params_random, cv=5):

        return RandomizedSearchCV(estimator = self.clf, param_distributions = params_random, cv=5)

    

    def GridSearchCV(self, params_random, cv=5):

        return  GridSearchCV(estimator = self.clf, param_grid = params_random, cv=5)
Stoch_gradient_descent_params = {

    'max_iter': 3000, 

    'tol': 0.001, 

    'penalty': 'l2', 

    'eta0': 0.04, 

    'random_state': 42,

    'learning_rate': 'invscaling', 

    'loss': 'squared_loss'

}



Ridge_params = {

    'alpha': 2, 

    'max_iter': 1000,

    'tol': 0.001,

    #'solver': "saga", 

    'normalize': False,

    'fit_intercept': True,

    'random_state': 42

}



Lasso_params = {

    'alpha': 5,

    'max_iter': 1000,

    'random_state': 42,

    'tol': 0.001, 

    'warm_start': True,

    'selection': 'random',

    'normalize': True,

    'positive': True,

    

}





ElasticNet_params = {

    'alpha': 2, 

    'l1_ratio': 1, 

    'random_state': 42,

    'max_iter': 3000,

    'fit_intercept': True,

    'selection': 'random',

    'tol': 0.001,

    'positive': True,

    'precompute': False,

    'warm_start': False

    

}



random_grid_params = {

    'max_depth': 14, 

    'min_samples_leaf': 1,

    'min_samples_split': 2,

    'n_estimators': 500,

    'random_state': 42

}



Grad_boosting_params = {

    'loss': 'lad',

    'learning_rate': 0.04,

    'n_estimators': 3000,

    'criterion': 'mse',

    'max_depth': 10,

    'min_samples_leaf': 7,

    'min_samples_split': 16,

    'random_state': 42

}
# Create 5 objects that represent our 4 models

sgd_reg = SklearnHelper(clf=SGDRegressor, params=Stoch_gradient_descent_params)

ridge_reg = SklearnHelper(clf=Ridge, params=Ridge_params)

lasso_reg = SklearnHelper(clf=Lasso, params=Lasso_params)

elastic_net = SklearnHelper(clf=ElasticNet, params=ElasticNet_params)

rf_search_one = SklearnHelper(clf=RandomForestRegressor, params=random_grid_params)

gr_boos_search_one = SklearnHelper(clf=GradientBoostingRegressor, params=Grad_boosting_params)
gr_boos_search_one.fit(X_train_new_df, Y)

predict_gr_boos_search_one = gr_boos_search_one.predict(X_test_new_df)



rf_search_one.fit(X_train_new_df, Y)

predict_rf_search_one = rf_search_one.predict(X_test_new_df)



sgd_reg.fit(X_train_new_df, Y)

predict_X_test_stoch_gradient_descent = sgd_reg.predict(X_test_new_df)



ridge_reg.fit(X_train_new_df, Y)

Ridge_Standart = ridge_reg.predict(X_test_new_df)



lasso_reg.fit(X_train_new_df, Y)

Lasso_Standart = lasso_reg.predict(X_test_new_df)



elastic_net.fit(X_train_new_df, Y)

predict_elastic_net = elastic_net.predict(X_test_new_df)
sns.set(rc={'figure.figsize': (15, 9)})

plt.subplot (231)

plot_SaleYandpredY(predict_gr_boos_search_one)

plt.subplot (232)

plot_SaleYandpredY(predict_rf_search_one)

plt.subplot (233)

plot_SaleYandpredY(predict_X_test_stoch_gradient_descent)

plt.subplot (234)

plot_SaleYandpredY(Ridge_Standart)

plt.subplot (235)

plot_SaleYandpredY(Lasso_Standart)

plt.subplot (236)

plot_SaleYandpredY(predict_elastic_net)
sns.kdeplot(predict_gr_boos_search_one, label = 'predict_gr_boos_search_one')

sns.kdeplot(predict_rf_search_one, label = 'predict_rf_search_one')



sns.kdeplot(predict_X_test_stoch_gradient_descent, label = 'predict_X_test_stoch_gradient_descent')

sns.kdeplot(Ridge_Standart, label = 'Ridge_Standart')



sns.kdeplot(Lasso_Standart, label = 'Lasso_Standart')

sns.kdeplot(predict_elastic_net, label = 'predict_elastic_net')
id_test = test[['Id']]

id_test = id_test.astype(int)
predict_SalePrice = pd.DataFrame.from_dict({

    

    'GradientBoostingRegressor': predict_gr_boos_search_one,

    'RandomForestRegressor': predict_rf_search_one,

    'ElasticNet': predict_elastic_net,

    'Lasso': Lasso_Standart,

    'Ridge': Ridge_Standart,

    'SGDRegressor': predict_X_test_stoch_gradient_descent

})
predict_SalePrice.head()
scatter = scatter_matrix(predict_SalePrice, figsize=(19, 19))
predict_SalePrice.plot(x = "GradientBoostingRegressor", y = "Lasso", kind = "scatter")
pred = predict_SalePrice.mean(axis=1)
finall_F_mean = pd.DataFrame.from_dict({'Id': list(id_test.Id), 

                                  'SalePrice': pred})



finall_F_mean2 = pd.DataFrame.from_dict({'Id': list(id_test.Id), 

                                  'SalePrice': predict_SalePrice[['RandomForestRegressor', 'SGDRegressor']].mean(axis=1)})
finall_F_mean.head()
finall_F_mean2.head()
finall_F_mean.to_csv("Submission_mean.csv", index=False)

finall_F_mean2.to_csv("Submission_mean2.csv", index=False)