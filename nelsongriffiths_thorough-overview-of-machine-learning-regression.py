#Import libraries to visualize and clean data

import pandas as pd

import numpy as np



import seaborn as sns

#Set color palette for graphs

sns.set_palette(sns.color_palette('hls', 7))



import matplotlib.pyplot as plt

import scipy.stats

from scipy.stats import norm, skew #for some statistics

import time

from sklearn.preprocessing import LabelEncoder, RobustScaler



#Silencing deprication warnings

import warnings

warnings.filterwarnings("ignore")



#Import libraries for modeling and validation

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.kernel_ridge import KernelRidge

from xgboost import XGBRegressor

import lightgbm as lgb

from sklearn.model_selection import cross_val_score, RandomizedSearchCV, train_test_split, learning_curve

from sklearn.metrics import mean_squared_error

from sklearn.feature_selection import SelectKBest

from sklearn.pipeline import Pipeline, make_pipeline



#Import libraries for stacking 

from mlens.metrics.metrics import rmse

from mlens.visualization import corrmat

from mlens.ensemble import SuperLearner
#Get the train and test data and merge them into one dataframe

train = pd.read_csv('../input/train.csv')

test = pd.read_csv("../input/test.csv")

Id = test.Id

train.shape[0]
#Make sure the data read in okay

train.describe()
#Create a new set of correlation data. 

corr_train = train.corr()

#Select thevariables most correlated with sales price and show a heat map.

best_corr = corr_train.index[abs(corr_train["SalePrice"])>0.5]

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(train[best_corr].corr(), square = True, annot = True)
#Overall Quality boxplot

f, ax = plt.subplots(figsize=(12, 9))

sns.boxplot(x = 'OverallQual', y = 'SalePrice', data = train)

plt.xlabel('Overall Quality', fontsize=15)

plt.ylabel('Sale Price', fontsize=15)

plt.title('Sale Price by Overall Quality', fontsize=15)
#Scatterplot looking at GrLivArea

sns.lmplot(x = 'GrLivArea', y = 'SalePrice', data = train, aspect = 2)

plt.xlabel('Living Area', fontsize=12)

plt.ylabel('Sale Price', fontsize=12)

plt.title('Sale Price by Living Area', fontsize=15)
#Drop outliers

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

sns.lmplot(x = 'GrLivArea', y = 'SalePrice', data = train, aspect = 2)

plt.xlabel('Living Area', fontsize=12)

plt.ylabel('Sale Price', fontsize=12)

plt.title('Sale Price by Living Area', fontsize=15)
#Boxplot for GarageCars

f, ax = plt.subplots(figsize=(12, 9))

sns.boxplot(x = 'GarageCars', y = 'SalePrice', data = train)

plt.xlabel('Cars in Garage', fontsize=15)

plt.ylabel('Sale Price', fontsize=15)

plt.title('Sale Price by Cars in Garage', fontsize=15)
#Look at the first floor square footage

sns.lmplot(x = '1stFlrSF', y = 'SalePrice', data = train, aspect = 2)

plt.xlabel('1st Floor Square Footage', fontsize=12)

plt.ylabel('Sale Price', fontsize=12)

plt.title('Sale Price by 1st Floor Square Footage', fontsize=15)
#Graph total rooms above ground

f, ax = plt.subplots(figsize=(12, 9))

sns.boxplot(x = 'TotRmsAbvGrd', y = 'SalePrice', data = train)

plt.xlabel('Rooms Above Ground', fontsize=15)

plt.ylabel('Sale Price', fontsize=15)

plt.title('Sale Price by Rooms Above Ground', fontsize=15)
#Look at histogram of SalePrice

f, ax = plt.subplots(figsize=(12, 9))

sns.distplot(train.SalePrice)

plt.xlabel('Sale Price', fontsize=15)

plt.title('Sale Price Distribution', fontsize=15)
#Log transformation and graph again to show fixed distribution

train.SalePrice = np.log1p(train.SalePrice)

f, ax = plt.subplots(figsize=(12, 9))

sns.distplot(train.SalePrice)

plt.xlabel('Sale Price', fontsize=15)

plt.title('Sale Price Distribution', fontsize=15)
#Set target variable as y, combine datasets, and drop unnecessary variables

y = train.SalePrice

n_train = train.shape[0]

data = pd.concat((train.drop('SalePrice', axis = 1), test)).reset_index(drop = True)

data = data.drop('Id', axis = 1)

n_train
#Check to see missing values

#data.isna().sum().sort_values(ascending = False)[:35]

data_na = (data.isnull().sum() / len(data)) * 100

data_na = data_na.drop(data_na[data_na == 0].index).sort_values(ascending=False)[:30]

f, ax = plt.subplots(figsize=(12, 10))

plt.xticks(rotation='90')

sns.barplot(x=data_na.index, y = data_na)

plt.xlabel('Features', fontsize=15)

plt.ylabel('Percent of missing values', fontsize=15)

plt.title('Percent missing data by feature', fontsize=15)
data = data.drop('PoolQC', axis = 1)

data = data.drop('Utilities', axis = 1)
#Use a for loop to quickly fill in the missing values for all of the categorical variables that don't exist

cat_missing = ['MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

              'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', "MasVnrType",

               'MSSubClass']

for col in cat_missing:

    data[col] = data[col].fillna('None')
#Use a for loop to fill in the missing values for numerical variables

num_missing = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea', 'GarageCars',

               'GarageArea', 'GarageYrBlt']

for col in num_missing:

    data[col] = data[col].fillna(0)
data["Functional"] = data["Functional"].fillna("Typ")

data['KitchenQual'] = data['KitchenQual'].fillna("TA")
#Impute with median from neighborhood group

data["LotFrontage"] = data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
fill_mode = ['MSZoning', 'SaleType', 'Exterior1st', 'Electrical', 'Exterior2nd']

for col in fill_mode:

    data[col] = data[col].fillna(data[col].mode()[0])
#Check for any more missing values

data.isnull().any().sum()
#Create new variable for total square footage

data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']

data['Total_Bathrooms'] = (data['FullBath'] + (0.5*data['HalfBath']) + data['BsmtFullBath'] + (0.5*data['BsmtHalfBath']))

data['Total_porch_sf'] = (data['OpenPorchSF'] + data['3SsnPorch'] +data['EnclosedPorch'] 

                          + data['ScreenPorch'] + data['WoodDeckSF'])



#simplified data

data['haspool'] = data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

data['has2ndfloor'] = data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

data['hasgarage'] = data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

data['hasbsmt'] = data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

data['hasfireplace'] = data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
#Change month and year to categorical features; also the condition of the house

data['YrSold'] = data['YrSold'].astype(str)

data['MoSold'] = data['MoSold'].astype(str)

data['OverallCond'] = data['OverallCond'].astype(str)

data['MSSubClass'] = data['MSSubClass'].apply(str)



#Label Encoder for categorical variables with order (such as those that are ranked 1-10)

cols = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold']

for i in cols:

        lbl = LabelEncoder()

        lbl.fit(list(data[i].values)) 

        data[i] = lbl.transform(list(data[i].values))
numeric_feats = data.dtypes[data.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

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

    data[feat] = boxcox1p(data[feat], lam)
#Get dummy variables

data = pd.get_dummies(data)

#Get new shape of data

data.shape
x = data[:n_train]

test = data[n_train:]
#Define scoring for our models

def get_rmse(model, x, y):

    scores = np.sqrt(-cross_val_score(model, x, y, cv = 5, scoring = 'neg_mean_squared_error'))

    return scores.mean()
#Function from sklearn for plotting learning curves

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    """

    Generate a simple plot of the test and training learning curve.



    Parameters

    ----------

    estimator : object type that implements the "fit" and "predict" methods

        An object of that type which is cloned for each validation.



    title : string

        Title for the chart.



    X : array-like, shape (n_samples, n_features)

        Training vector, where n_samples is the number of samples and

        n_features is the number of features.



    y : array-like, shape (n_samples) or (n_samples, n_features), optional

        Target relative to X for classification or regression;

        None for unsupervised learning.



    ylim : tuple, shape (ymin, ymax), optional

        Defines minimum and maximum yvalues plotted.



    cv : int, cross-validation generator or an iterable, optional

        Determines the cross-validation splitting strategy.

        Possible inputs for cv are:

          - None, to use the default 3-fold cross-validation,

          - integer, to specify the number of folds.

          - An object to be used as a cross-validation generator.

          - An iterable yielding train/test splits.



        For integer/None inputs, if ``y`` is binary or multiclass,

        :class:`StratifiedKFold` used. If the estimator is not a classifier

        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.



        Refer :ref:`User Guide <cross_validation>` for the various

        cross-validators that can be used here.



    n_jobs : int or None, optional (default=None)

        Number of jobs to run in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.

        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`

        for more details.



    train_sizes : array-like, shape (n_ticks,), dtype float or int

        Relative or absolute numbers of training examples that will be used to

        generate the learning curve. If the dtype is float, it is regarded as a

        fraction of the maximum size of the training set (that is determined

        by the selected validation method), i.e. it has to be within (0, 1].

        Otherwise it is interpreted as absolute sizes of the training sets.

        Note that for classification the number of samples usually have to

        be big enough to contain at least one sample from each class.

        (default: np.linspace(0.1, 1.0, 5))

    """

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring = 'neg_mean_squared_error')

    train_scores_mean = np.mean(np.sqrt(-train_scores), axis=1)

    train_scores_std = np.std(np.sqrt(-train_scores), axis=1)

    test_scores_mean = np.mean(np.sqrt(-test_scores), axis=1)

    test_scores_std = np.std(np.sqrt(-test_scores), axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt

#Loop through models and compare results

en = ElasticNet()

lr = Lasso()

br = BayesianRidge()

rf = RandomForestRegressor()

xgbm = XGBRegressor()

svr = SVR()

lgbm = lgb.LGBMRegressor()

krr = KernelRidge()

gb = GradientBoostingRegressor()

reg = [en, lr, br, rf, xgbm, svr, lgbm, krr, gb]

#Use cross validation with the root mean squared error to get an idea of which models will work best for this problem

for model in reg:

    scores = get_rmse(model, x, y)

    print(model.__class__.__name__,": RMSE =", scores)

    plot_learning_curve(model, model.__class__.__name__, x, y, cv = 5)

    plt.show()
#Function for getting the best hyperparameters for each model

def tune_parameters(pipe, x, y, params, n_iter, base_model):

    orig_score = get_rmse(base_model, x, y)

    model = RandomizedSearchCV(pipe, param_distributions = params, n_iter = n_iter, cv = 5, scoring = 'neg_mean_squared_error', n_jobs=-1)

    start_time = time.time()

    model.fit(x,y)

    print("--- Finding the best hyperparameters took %s seconds --- \n" % (time.time() - start_time))

    print("--- Best Hyperparameters for %s --- \n" % (pipe.named_steps.model.__class__.__name__))

    print(model.best_params_, '\n')

    model = model.best_estimator_

    new_score = get_rmse(model, x, y)

    print('----- SCORE IMPROVEMENTS ----- \n')

    print("The original RMSE was %s \n" %(orig_score))

    print("The RMSE after finding the best parameters is %s \n" %(new_score))

    print("That is an improvement of %s \n" %(orig_score - new_score))

    print("----- NEW LEARNING CURVE -----")

    plot_learning_curve(model, pipe.named_steps.model.__class__.__name__, x, y, cv = 5)

    plt.show()

    return model
#Create the steps for each of the model pipelines

select = SelectKBest(k = 'all')

en_steps = [('Scaler', RobustScaler()), ('feature_selection', select), ('model', ElasticNet(alpha = .0005))]

lr_steps = [('Scaler', RobustScaler()), ('model', Lasso(alpha = .0005))]

xgb_steps = [('feature_selection', select), ('model', xgbm)]

lgb_steps = [('feature_selection', select), ('model', lgbm)]



#Build Pipelines

en_pipe = Pipeline(en_steps)

lr_pipe = Pipeline(lr_steps)

xgb_pipe = Pipeline(xgb_steps)

lgb_pipe = Pipeline(lgb_steps)
#Set up parameters for each pipeline to search through

params = {'xgb' : {'feature_selection__k' : np.arange(10,220, 10),

                  'model__n_estimators' : np.arange(100, 2000, 100),

                  'model__learning_rate' : np.arange(.01, .5, .01),

                  'model__gamma' : np.arange(0, 1, .01),

                  'model__max_depth' : np.arange(3, 10, 1),

                  'model__min_child_weight' : np.arange(0, 10, 1),

                  'model__colsample_bytree' : np.arange(.1, 1, .1)},

         'en' : {'feature_selection__k' : np.arange(10,220, 10),

                'model__l1_ratio' : np.arange(0, 1, .1),

                'model__selection' : ['cyclic', 'random']},

         'lr' : {'model__max_iter' : np.arange(700, 1400, 100),

                'model__selection' : ['cyclic', 'random']},

         'lgb' : {'feature_selection__k' : np.arange(10,220, 10),

                 'model__n_estimators' : np.arange(500, 3000, 200),

                  'model__learning_rate' : np.arange(.01, .5, .01),

                  'model__max_depth' : np.arange(3,8,1),

                  'model__colsample_bytree' : np.arange(.1, 1, .1)}

         }
#xgb_model = tune_parameters(xgb_pipe, x, y, params['xgb'], 50, xgb)
xgb_model = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.1, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             nthread = -1)
lasso = tune_parameters(lr_pipe, x, y, params['lr'], 10, lr)
en_model = tune_parameters(en_pipe, x, y, params['en'], 10, en)
#lgb_model = tune_parameters(lgb_pipe, x, y, params['lgb'], 50, lgb)
lgb_model = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



plot_learning_curve(lgb_model, "LGBM Regressor", x, y, cv = 5)
gb_model = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
lr_model = Lasso(alpha = .0005)

lr_model.fit(x.values, y.values)

feature_importances = pd.DataFrame(columns = ['Variable', 'Coef'])

feature_importances['Variable'] = list(x)

feature_importances['Coef'] = abs(lr_model.coef_)

feature_importances.sort_values(by = 'Coef', ascending = False, inplace = True)

feature_importances = feature_importances[0:10]

feature_importances.head(15)
f, ax = plt.subplots(figsize=(12, 10))

plt.xticks(rotation='90')

sns.barplot(x=feature_importances['Variable'], y = feature_importances['Coef'])

plt.xlabel('Features', fontsize=15)

plt.ylabel('Importance', fontsize=15)

plt.title('Most Important Features in Lasso Model', fontsize=15)
en_model = en_model.fit(x.values,y.values)

krr_model = krr.fit(x.values,y.values)

gb_model = gb_model.fit(x.values,y.values)

lr_model = lasso.fit(x.values, y.values)

xgb_model = xgb_model.fit(x.values, y.values)

lgb_model = lgb_model.fit(x.values, y.values)

br_model = br.fit(x.values, y.values)
#Prediction scored .11856

lasso_pred = np.expm1(lr_model.predict(test.values))

submission = pd.DataFrame({'Id':Id,'SalePrice':lasso_pred})

submission.to_csv("submission.csv",index=False)
#Make predictions and combine into dataframe

xgb_pred = np.expm1(xgb_model.predict(test.values))

lgb_pred = np.expm1(lgb_model.predict(test.values))

br_pred = np.expm1(br_model.predict(test.values))

en_pred = np.expm1(en_model.predict(test.values))

gb_pred = np.expm1(gb_model.predict(test.values))

krr_pred = np.expm1(krr_model.predict(test.values))



predictions = pd.DataFrame({"XGBoost":xgb_pred.ravel(), "Elastic Net":en_pred.ravel(), "Bayesian Ridge": br_pred.ravel(),

                            "LightGB": lgb_pred.ravel(), "Gradient Boosting": gb_pred.ravel(), "Lasso": lasso_pred.ravel(),

                           "Kernel": krr_pred.ravel()})
corrmat(predictions.corr())

plt.show
finalMd = (np.expm1(lr_model.predict(test.values)) + np.expm1(en_model.predict(test.values)) 

           + np.expm1(krr_model.predict(test.values)) + np.expm1(gb_model.predict(test.values))) / 4

finalMd
#Scored .11584

submission = pd.DataFrame({'Id':Id,'SalePrice':finalMd})

submission.to_csv("submission.csv",index=False)
x_train, x_test, y_train, y_test = train_test_split(x, y)
#Stacking

sl = SuperLearner(

    scorer = rmse,

    folds=7,

    random_state=42,

    verbose=2,

    n_jobs=-1

)

sl.add([en_model, krr_model, gb_model])

sl.add_meta(lr_model)
sl.fit(x_train, y_train)

preds = sl.predict(x_test)

scores = mean_squared_error(y_test, preds)

scores = np.sqrt(scores)

print(scores)
sl.fit(x.values, y.values)

ens_pred = np.expm1(sl.predict(test.values))

ens_pred
lgb_pred = np.expm1(lgb_model.predict(test.values))
#.11500

y_pred = .72*ens_pred + .28*lgb_pred
submission = pd.DataFrame({'Id':Id,'SalePrice':y_pred})

submission.to_csv("submission.csv",index=False)