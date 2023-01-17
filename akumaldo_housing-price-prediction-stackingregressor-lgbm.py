# Python ≥3.5 is required

import sys

assert sys.version_info >= (3, 5)



# Scikit-Learn ≥0.20 is required

import sklearn

assert sklearn.__version__ >= "0.20"



# Common imports

import numpy as np

import os



from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns #for better and easier plots

%matplotlib inline





# Ignore useless warnings (see SciPy issue #5998)

import warnings

warnings.filterwarnings(action="ignore")
train = pd.read_csv("../input/train.csv")#loading datasets

test = pd.read_csv("../input/test.csv")

train["SalePrice"] = np.log1p(train["SalePrice"])# logs respond better to skewed data as we have in this one,decided to change the column here at the beggining for clarity



print(train.shape, test.shape) #printing the shape
train.head() #head(), taking a look at the 5 first entries
#train.info()
#let's create a function to check for null values, calculate the percentage relative to the total size

#only shows null values.

def missing_values_calculate(trainset): 

    nulldata = (trainset.isnull().sum() / len(trainset)) * 100

    nulldata = nulldata.drop(nulldata[nulldata == 0].index).sort_values(ascending=False)

    ratio_missing_data = pd.DataFrame({'Ratio' : nulldata})

    return ratio_missing_data.head(30)
missing_values_calculate(train)
#let's createa function that fill all the NaN values that means None, eg: NaN for pool means NO poll

def imputer_specific_categories(dataset, listfeatures):

    data = dataset.copy()

    for names in listfeatures:

        data[names] = data[names].fillna("None")

    return data
features_meaning_none_value= ["Alley", "PoolQC", "MiscFeature","Fence","MiscVal","FireplaceQu",

                              'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','MSSubClass']

train_drop = imputer_specific_categories(train, features_meaning_none_value)

test = imputer_specific_categories(test,features_meaning_none_value)
missing_values_calculate(train_drop)#way better now, the highest value, now is lot frontage, 18%
missing_values_calculate(test) #for the given test dataset, this is what we see...
train_drop['total_year_build_and_remod']= train_drop['YearBuilt']+train_drop['YearRemodAdd']



train_drop['Total_sqr_are'] = (train_drop['BsmtFinSF1'] + train_drop['BsmtFinSF2'] +

                                 train_drop['1stFlrSF'] + train_drop['2ndFlrSF'])



train_drop['Total_Bathrooms'] = (train_drop['FullBath'] + (0.5 * train_drop['HalfBath']) +

                               train_drop['BsmtFullBath'] + (0.5 * train_drop['BsmtHalfBath']))



train_drop['Total_porch_sf'] = (train_drop['OpenPorchSF'] + train_drop['3SsnPorch'] +

                              train_drop['EnclosedPorch'] + train_drop['ScreenPorch'] +

                              train_drop['WoodDeckSF'])



test['total_year_build_and_remod']= test['YearBuilt']+test['YearRemodAdd']



test['Total_sqr_are'] = (test['BsmtFinSF1'] + test['BsmtFinSF2'] +

                                 test['1stFlrSF'] + test['2ndFlrSF'])



test['Total_Bathrooms'] = (test['FullBath'] + (0.5 * test['HalfBath']) +

                               test['BsmtFullBath'] + (0.5 * test['BsmtHalfBath']))



test['Total_porch_sf'] = (test['OpenPorchSF'] + test['3SsnPorch'] +

                              test['EnclosedPorch'] + test['ScreenPorch'] +

                              test['WoodDeckSF'])
def has_feature_function(data, column_name_features={}):

    for column,name in column_name_features.items():

        data[name] = data[column].apply(lambda x: 1 if x > 0 else 0)
columns = {'PoolArea': 'haspool',

           '2ndFlrSF': 'has2ndfloor',

           'GarageArea': 'hasgarage',

           'TotalBsmtSF': 'hasbasement',

           'Fireplaces': 'hasfireplace',

          }
has_feature_function(train_drop, columns) #calling our function in our training set

has_feature_function(test, columns) #and calling our function for our testing set
train_drop.shape #it has increased the number of features significantly
corr = train_drop.corr() #Let's take a look at the pearson's corr, just to have an overall view of how the attributes influence the price.

#using this correlation, we can have an idea of the linear correlation, positive and negative.

ax = sns.set(rc={'figure.figsize':(40,25)})

plt.xticks(fontsize=20)

plt.yticks(fontsize=20)

sns.heatmap(corr[:30], annot=True).set_title('Pearsons Correlation Factors Heat Map', color='black', size='20')
ax = sns.set(rc={'figure.figsize':(10,5)})

train_drop["OverallQual"].value_counts().plot.bar()
#let's use the seaborn plot to have a nice plot,showing the bell shape curve of this distribution.

sns.countplot(train_drop["OverallQual"]) #not surprisingly, average quality is more representative.
# Plots the disribution of a variable colored by value of the target



def kde_target(var_name, df, corr_target, shade_or_not = False):

    sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

    sns.set_style("darkgrid")    

    # Calculate the correlation coefficient between the new variable and the target

    corr = df[corr_target].corr(df[var_name])

    for var in df[var_name].sort_values().unique():            

        sns.kdeplot(df.ix[df[var_name] == var, corr_target], label = '%s : %d' %(var_name,var),shade=shade_or_not)

        

    # label the plot

    plt.xticks(fontsize=10)

    plt.yticks(fontsize=10)

    plt.xlabel('Distribution(%s)'%(corr_target), fontsize=10); plt.ylabel(corr_target,fontsize=10); plt.title('%s Distribution' % var_name,fontsize=10);

    plt.legend(fontsize=10);

    

    # print out the correlation

    print('The correlation between %s and the %s is %0.4f' % (var_name, corr_target, corr))
kde_target("OverallQual", train_drop, "SalePrice", True)
sns.countplot(y=train_drop["OverallCond"]); plt.title('Overall Condition distribution', fontsize=15)

plt.ylabel('Overall Condition', fontsize=15)

plt.xlabel('Count', fontsize=15)
sns.catplot(x='OverallCond', y='SalePrice', data=train_drop)

plt.title('Sale Price in relationship to Overall Condition')

sns.catplot(x='OverallQual', y='SalePrice', data=train_drop)

plt.title('Sale Price in relationship to Overall Quality')

plt.tight_layout(h_pad = 2.5)
from_2000 = train_drop[train_drop['YearBuilt'] > 2000]

ax = sns.set(rc={'figure.figsize':(10,5)})

sns.countplot(from_2000["YearBuilt"])

plt.subplot(1,1,1)

sns.catplot(y='SalePrice', x='YearBuilt', data=from_2000); plt.xticks(rotation='vertical')
#let's check the relationship between sale price and living area, as it could be seen in the pearson's correlation

#there's a positive correlation between sales price and living area.

fig, ax = plt.subplots()

ax.scatter(x = train_drop['GrLivArea'], y = train_drop['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)
#as can be seen above, only two ouliers that should be taken care of, lower prices relative to higher living area

#we have, as well, two outliers properly placed at the top of the chart, let's remove the two bottom outliers

train_drop.drop(train_drop[(train_drop["SalePrice"] < 300000) & (train_drop["GrLivArea"] > 4000)].index, inplace=True)
train_labels = train_drop["SalePrice"].copy() #creating the prediction target.
train_drop.drop(["SalePrice",'Utilities', 'Street', 'PoolQC','Id'], axis=1, inplace=True) #this is really important, separate our target Y from our X

test.drop(['Utilities', 'Street', 'PoolQC','Id'], axis=1, inplace=True)
num_attribs = train_drop.select_dtypes(exclude=['object']) #selecting all the numerical data to use in our function DataFrameSelector

cat_attribs = train_drop.select_dtypes(exclude=['int64','float64']) #selecting non numerical data to use in our function DataFrameSelector
#this pipeline is gonna be use for numerical atributes and standard scaler

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler,RobustScaler, MinMaxScaler





num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        #('std_scaler', StandardScaler()),

        #('robust_scaler', RobustScaler()),

        ('minmaxscaler', MinMaxScaler()),

    ])
# Inspired from stackoverflow.com/questions/25239958

#this is gonna be used to imput categorical values

from sklearn.base import BaseEstimator, TransformerMixin



class MostFrequentImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],

                                        index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.most_frequent_)
from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import OneHotEncoder #gonna try this one later



cat_pipeline = Pipeline([

        ("imputer", MostFrequentImputer()),

        ("cat_encoder", OneHotEncoder()),

    ])
from sklearn.compose import ColumnTransformer





full_pipeline = ColumnTransformer([

        ("num", num_pipeline, list(num_attribs)),

        ("cat", cat_pipeline, list(cat_attribs)),

    ])
train_dummies = pd.get_dummies(train_drop)

test_dummies = pd.get_dummies(test)
train_dummies, test_dummies = train_dummies.align(test_dummies, join='inner', axis=1)
print(train_dummies.shape, test_dummies.shape)
#let's create this function to make it easier and clean to fit the model and use the cross_val_score and obtain results

import time #implementing in this function the time spent on training the model

from sklearn import metrics

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score, KFold



def modelfit(alg, dtrain, target, only_predict = False):

    #Fit the algorithm on the data

    time_start = time.perf_counter() #start counting the time

    if not only_predict:

        alg.fit(dtrain, target)

        

    #Predict training set:

    dtrain_predictions = alg.predict(dtrain)

    

    kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

    

    cv_score = cross_val_score(alg, dtrain,target, cv=kfolds, scoring='neg_mean_squared_error')

    cv_score = np.sqrt(-cv_score)

    

    time_end = time.perf_counter()

    

    total_time = time_end-time_start

    #Print model report:

    print("\nModel Report")

    print("RMSE :  {:.4f}".format(np.sqrt(mean_squared_error(target, dtrain_predictions))))

    print("CV Score : Mean -  %.4f | Std -  %.4f | Min -  %.4f | Max - %.4f" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))

    print("Amount of time spent during training the model and cross validation: %4.3f seconds" % (total_time))
# Plot feature importance

def plot_feature_importance(model, df):

    feature_importance = model.feature_importances_[:30]

    # make importances relative to max importance

    plt.figure(figsize=(20, 20)) #figure size

    feature_importance = 100.0 * (feature_importance / feature_importance.max()) #making it a percentage relative to the max value

    sorted_idx = np.argsort(feature_importance)

    pos = np.arange(sorted_idx.shape[0]) + .5

    plt.barh(pos, feature_importance[sorted_idx], align='center')

    plt.yticks(pos, df.columns[sorted_idx], fontsize=15) #used train_drop here to show the name of each feature instead of our train_prepared 

    plt.xlabel('Relative Importance', fontsize=20)

    plt.ylabel('Features', fontsize=20)

    plt.title('Variable Importance', fontsize=30)
train_prepared = full_pipeline.fit_transform(train_drop)

train_dummies_prepared = num_pipeline.fit_transform(train_dummies)
#ok, nice, now that we have our data prepared, let's start testing some models

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

modelfit(lin_reg, train_prepared, train_labels)
from sklearn.linear_model import Lasso



model_lasso =  Lasso()

modelfit(model_lasso, train_dummies_prepared, train_labels)
from sklearn.linear_model import ElasticNet



model_elastic = ElasticNet(max_iter=1e7, l1_ratio=0.9)

modelfit(model_elastic, train_prepared, train_labels)
from sklearn.tree import DecisionTreeRegressor



tree_reg = DecisionTreeRegressor(random_state=42)

modelfit(tree_reg, train_prepared, train_labels) 
from sklearn.kernel_ridge import KernelRidge



model_kernel = KernelRidge(alpha=0.6, kernel='polynomial', degree=5, coef0=2.5) #the higher the degree, the polynomial degree, the better it fits the data, but it increases variance, and the model does not generalize well

modelfit(model_kernel, train_prepared, train_labels)
from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor(n_estimators=40, random_state=42, 

                                   min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

modelfit(forest_reg, train_dummies_prepared, train_labels)
plot_feature_importance(forest_reg, train_dummies)
from sklearn.ensemble import AdaBoostRegressor



tree_ada = DecisionTreeRegressor(random_state = 42,max_depth = 4)



ada_reg = AdaBoostRegressor(

    tree_reg, n_estimators=3000, random_state=42,learning_rate=0.009, loss='square')

modelfit(ada_reg, train_dummies_prepared, train_labels)
plot_feature_importance(ada_reg, train_dummies)
from sklearn.ensemble import GradientBoostingRegressor



params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': 0.01, 'loss': 'ls'}



params_1 = {'n_estimators': 3000, 'learning_rate': 0.05, 'max_depth' : 4,

            'max_features': 'sqrt', 'min_samples_leaf': 15, 'min_samples_split': 10, 

            'loss': 'huber', 'random_state': 42}



train_dummies_prepared = num_pipeline.fit_transform(train_dummies)



gdb_model = GradientBoostingRegressor(**params_1)

modelfit(gdb_model, train_dummies_prepared, train_labels)
plot_feature_importance(gdb_model, train_dummies)
#this function is gonna be used when it has been estimated the best features, eg: using random forest, in our case Gradient Boosting

## then we would want to especify only those features when we train our model.

from sklearn.base import BaseEstimator, TransformerMixin



def indices_of_top_k(arr, k):

    return np.sort(np.argpartition(np.array(arr), -k)[-k:])



class TopFeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, feature_importances, k):

        self.feature_importances = feature_importances

        self.k = k

    def fit(self, X, y=None):

        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)

        return self

    def transform(self, X):

        return X[:, self.feature_indices_]
import xgboost as xgb



    

params = {'objective': 'reg:linear', 

              'eta': 0.01, 

              'max_depth': 6, 

              'subsample': 0.6, 

              'colsample_bytree': 0.7,  

              'eval_metric': 'rmse', 

              'seed': 42, 

              'silent': True,

    }



model_xgb = xgb.XGBRegressor(**params,n_estimators=3000,learning_rate=0.05,verbose_eval=True)
#train_prepared_feature_importances = preparation_and_feature_selection_pipeline.fit_transform(train_drop)

#in case you wanna test with only some features, the ones you found to be more important, this is why you have this pipeline
from datetime import datetime





fold_val_pred = []

fold_err = []

fold_importance = []

record = dict()

kfold = list(KFold(10,shuffle = True, random_state = 42).split(train_dummies))



def xgb_model(train, params, train_label, fold, verbose, pipeline):



    

    for i, (trn, val) in enumerate(fold) :

        print(i+1, "fold.    RMSE")

    

        trn_x = train.loc[trn, :]

        trn_y = train_label[trn]

        val_x = train.loc[val, :]

        val_y = train_label[val]

    

        fold_val_pred = []

    

        start = datetime.now()

        model = xgb.train(params

                      , xgb.DMatrix(trn_x, trn_y)

                      , 3500

                      , [(xgb.DMatrix(trn_x, trn_y), 'train'), (xgb.DMatrix(val_x, val_y), 'valid')]

                      , verbose_eval=verbose

                      , early_stopping_rounds=500

                      , callbacks = [xgb.callback.record_evaluation(record)])

        best_idx = np.argmin(np.array(record['valid']['rmse']))



        val_pred = model.predict(xgb.DMatrix(val_x), ntree_limit=model.best_ntree_limit)

                

        fold_val_pred.append(val_pred*0.2)

        fold_err.append(record['valid_0']['rmse'][best_idx])

        fold_importance.append(model.feature_importance('gain'))

        

        print("xgb model.", "{0:.5f}".format(result['error']), '(' + str(int((datetime.now()-start).seconds/60)) + 'm)')
modelfit(model_xgb, train_dummies, train_labels)
plot_feature_importance(model_xgb, train_dummies)
#modelfit(xgb_model, train_prepared_feature_importances, train_labels)
from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV





pipe = Pipeline([

    ('pca', PCA(5)),

    ('xgb', xgb.XGBClassifier())

])



param_grid = {

    'pca__n_components': [3, 5, 7, 10,15],

    'xgb__n_estimators': [500,1000,2000,3500],

    'xgb__learning_rate': [0.09,0.1,0.05,0.001]

}

#grid_xgb = GridSearchCV(pipe, param_grid, scoring='neg_mean_absolute_error', verbose=10)

#grid_xgb.fit(train_prepared, train_labels)

#print(grid_xgb.best_params_)
#pipe.fit(train_prepared, train_labels)
#modelfit(pipe, train_prepared, train_labels, True)
import lightgbm as lgb



lgb_model = lgb.LGBMRegressor(objective='regression', 

                                       num_leaves=4,

                                       learning_rate=0.005, 

                                       n_estimators=10000,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose=-1,

                                       )



modelfit(lgb_model, train_dummies, train_labels)
plot_feature_importance(lgb_model, train_dummies)
### THIS CODE HAS BEEN USED BEFORE I KNEW ABOUT THE STACKING REGRESSOR ###

#this class is gonna be used to average the results of different models and get the best result out of it

from sklearn.base import RegressorMixin

from sklearn.base import clone





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
from mlxtend.regressor import StackingCVRegressor

models_total = (model_kernel, model_lasso, model_elastic, gdb_model, lgb_model)

model_ada_gbd = (ada_reg,gdb_model)

stack_gen = StackingCVRegressor(regressors= models_total,

                                meta_regressor=model_xgb,

                                use_features_in_secondary=True)
#running again some models, now using only some features: 

modelfit(stack_gen, train_dummies_prepared, train_labels)
test_dummies_prepared = num_pipeline.fit_transform(test_dummies)

#test_prepared = preparation_and_feature_selection_pipeline.fit_transform(test) #if you want to train with only specific features
prediction_some_regressors = stack_gen.predict(test_dummies_prepared)
sub = pd.read_csv('../input/sample_submission.csv')

sub['SalePrice'] = np.expm1(prediction_some_regressors)

sub.to_csv("stack_regression.csv", index=False)