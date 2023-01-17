# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# python version

import sys

assert sys.version_info > (3,5)



# sklearn version

import sklearn

assert sklearn.__version__ > '0.20'



# common imports

import os

import pandas as pd

import numpy as np



#visualization imports

import matplotlib.pyplot as plt

import seaborn as sns



# display visuals in the notebook

%matplotlib inline



# handle internal library warnings

import warnings

warnings.filterwarnings(action='ignore',message='')



# consistent plot size

from pylab import rcParams

rcParams['figure.figsize'] = 12,5

rcParams['xtick.labelsize'] = 12

rcParams['ytick.labelsize'] = 12

rcParams['axes.labelsize'] = 12

house_train_full =pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

house_test =pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
# view all the columns of the dataframe

pd.options.display.max_columns = None
# inspect the first few rows

house_train_full.head(10)
#house_train_full.info()
# function to return the percentage of null values in a list of features

def percent_na(feature, df = house_train_full):

    for val in feature:

        return df[feature].isna().sum()/len(df)
percent_na(list(house_train_full.columns),house_train_full).sort_values(ascending=False).head()
plt.hist(house_train_full['SalePrice'],bins=30)

plt.title('Histogram of House Sale Price')

plt.xlabel('Sale Price');
# create boxplot of SalePrice

plt.boxplot(house_train_full['SalePrice'],vert=False)

plt.title('Boxplot of House Sale Price');
# log transform of the sale price and check the histogram and boxplot

plt.hist(np.log(house_train_full['SalePrice']),bins=30)

plt.title('Histogram of House Sale Price (Log transformed)')

plt.xlabel('Sale Price (log transformed)');
# test for normality of the log transformed sale price

from statsmodels.graphics.gofplots import qqplot



qqplot(np.log(house_train_full['SalePrice']),line='s')

plt.title('Quantile-Quantile Plot Log Transformed Sale Price');
# create boxplot of the log transformed SalePrice

plt.boxplot(np.log(house_train_full['SalePrice']),vert=False)

plt.title('Boxplot of House Sale Price (log transformed)');
# function to return the quantile of the numerical feature

def quantile_num(df,num_feature,quant):

    quantiles = []

    for q in quant:

        quantiles.append(np.quantile(df[num_feature],q))

    return quantiles        
quantile_num(df=house_train_full,num_feature='SalePrice',quant = [0.5,0.75,0.9,0.95,0.99])
house_train_full[house_train_full['SalePrice']>np.quantile(house_train_full['SalePrice'],0.99)]
house_train_full.describe().transpose()
# uncomment to read the file content



#with  open('/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt') as file:

 #   file_content = file.read()

  #  print (file_content)
# check the abnormal sales neighborhoods

house_train_full[house_train_full['SaleCondition']=='Abnorml']['Neighborhood'].value_counts().sort_values(ascending=False)
# check the neighborhood of the houses in sale price above the 99 percentile

house_train_full[house_train_full['SalePrice']>np.quantile(house_train_full['SalePrice'],0.99)]['Neighborhood'].value_counts().sort_values(ascending=False)
# function to list all the categorical features

def type_features(df,features):

    cat_features = []

    num_features = []

    for feat in features:

        if df[feat].dtype == 'O':

            cat_features.append(feat)

        else:

            num_features.append(feat)

    return (cat_features,num_features)
categorical_features, numerical_features = type_features(house_train_full,house_train_full.columns)

len(categorical_features)
house_train_full.corr()['SalePrice'].sort_values(ascending=False).head(15)
house_train_full.corr()['SalePrice'].sort_values(ascending=False).tail(15)
from pandas.plotting import scatter_matrix

attributes = ['SalePrice','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF']

scatter_matrix(house_train_full[attributes],figsize=(15,8),grid=True);
for features in categorical_features:

    print (house_train_full[features].value_counts())
sns.countplot('SaleCondition',hue='SaleType',data=house_train_full)

plt.legend(loc='upper right')
sns.countplot('SaleCondition',data=house_train_full)

plt.legend(loc='upper right')
sns.countplot('OverallQual',data=house_train_full,palette='viridis')

plt.title('House Count per Overall Quality');

sns.swarmplot('OverallQual','SalePrice',data=house_train_full)

plt.title('Sale Price vs Overall Quality')

plt.legend(loc='upper right');
sns.swarmplot('Neighborhood','SalePrice',data=house_train_full[house_train_full['OverallQual']==10])

plt.title('Houses with Quality Rating 10, SalePrice vs Neighborhood');

# Boxplot SalePrice vs Neighborhood

plt.figure(figsize=(20,10))

sns.boxplot('Neighborhood','SalePrice',data=house_train_full,palette='viridis')

plt.tight_layout(True)

plt.title('SalePrice vs Neighborhood');
# Visualize the boxplot sorted by median SalePrice and plotted in descending order



# create the sorted dataframe

grouped = house_train_full.groupby(['Neighborhood'])

df = pd.DataFrame({col:vals['SalePrice'] for col,vals in grouped})



meds = df.median()

meds.sort_values(ascending=False, inplace=True)

df = df[meds.index]



# generate boxplot

plt.figure(figsize=(20,10))



df.boxplot(grid=False)



plt.tight_layout(True)

plt.title('SalePrice vs Neighborhood (sorted based on descending order of median price)')

plt.xlabel('Ames Neighborhoods')

plt.ylabel('House Sale Prices');
#create a copy of the original train set

housing = house_train_full.copy()
# read the categorical and the numerical features

categorical_features, numerical_features = type_features(housing,housing.columns)

len(categorical_features)
percent_na(list(housing.columns),housing).sort_values(ascending=False).head(10)
# drop list of columns

drop_list_1 = ['Id','PoolQC','MiscFeature','Alley','Fence']
# function to drop the columns

def drop_feature(df,drop_list):

    for feature in drop_list:

        df.drop(feature,axis=1,inplace=True)

    return df
save_id = house_test.copy()
# drop the features from the training set and the test set

drop_feature(housing,drop_list_1)

drop_feature(house_test,drop_list_1)
housing.head()
#update the feature list

categorical_features, numerical_features = type_features(housing,housing.columns)

len(categorical_features)
num_to_cat_list = ['MSSubClass','OverallQual','OverallCond','YearBuilt','YearRemodAdd','GarageYrBlt','MoSold','YrSold']
#convert the selected numerical features to categorical

def to_category(df,num_list):

    for feature in num_list:

        df[feature] = df[feature].astype('str')

    return df
to_category(housing,num_to_cat_list)

to_category(house_test,num_to_cat_list)
# update the categorical and the numerical feature list

categorical_features,numerical_features = type_features(housing,housing.columns)

len(categorical_features)
housing['OverallQual'].value_counts()
from sklearn.model_selection import StratifiedShuffleSplit



strat_split = StratifiedShuffleSplit(n_splits=10,test_size=0.2,random_state=42)

for train_index,valid_index in strat_split.split(housing,housing['OverallQual']):

    strat_house_train = housing.loc[train_index]

    strat_house_valid = housing.loc[valid_index]

len(strat_house_train)
strat_house_train.head()
# separate the SalePrice, the final label to be predicted

X_train = strat_house_train.drop('SalePrice',axis=1)

#evaluation is based on the log transformed value of the SalePrice

y_train = np.log(strat_house_train['SalePrice'])



X_valid = strat_house_valid.drop('SalePrice',axis=1)

y_valid = np.log(strat_house_valid['SalePrice'])



X_test = house_test.copy()

# reapply the function to extract the categorical and the numeric features

categorical_features,numerical_features = type_features(X_train,X_train.columns)
X_train_num = X_train[numerical_features]

X_train_num.head()
X_train_cat = X_train[categorical_features]

X_train_cat.head()
from sklearn.impute import KNNImputer

from sklearn.preprocessing import StandardScaler
imputer = KNNImputer()

X_train_num_imp = imputer.fit_transform(X_train_num)
scalar = StandardScaler()

X_train_num_prep = scalar.fit_transform(X_train_num_imp)
X_train_num_prep
# Reconstruct the numerical features dataframe

X_train_num_prepared = pd.DataFrame(X_train_num_prep,

                                   columns=list(X_train_num.columns),index=X_train_num.index)

X_train_num_prepared.head()
from sklearn.pipeline import Pipeline



num_pipeline = Pipeline([('num_imputer',KNNImputer()),

                         ('std_scalar',StandardScaler()),

                         ])
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
hot_encoder = OneHotEncoder(sparse=False)

ord_encoder = OrdinalEncoder()
# define a function to apply one hot encoder and then impute the missing values

def encode(df):

    # keep only the non null values 

    arr = np.array(df.dropna())

    # reshape the data for encoding

    arr_reshape = arr.reshape(-1,1)

    # encode the data

    arr_encoded = ord_encoder.fit_transform(arr_reshape)

    # bring the encoded data back to the df

    df.loc[df.notnull()] = np.squeeze(arr_encoded)

    return df

for cat_feature in categorical_features:    

    encode(X_train_cat[cat_feature])
X_train_cat.head()
cat_imputer = KNNImputer()

X_train_cat_imp = cat_imputer.fit_transform(X_train_cat)
# Reconstruct the categorical features dataframe

X_train_cat_prepared = pd.DataFrame(X_train_cat_imp,

                                   columns=list(X_train_cat.columns),index=X_train_cat.index)

X_train_cat_prepared.head()
#np.max(X_train_cat_prepared)
cat_pipeline = Pipeline([('cat_imputer',cat_imputer),

                         ('cat_std_sclar',StandardScaler())])
from sklearn.compose import ColumnTransformer



num_attribs = numerical_features

cat_attribs = categorical_features



full_pipeline = ColumnTransformer([

    ('num',num_pipeline,num_attribs),

    ('cat',cat_pipeline,cat_attribs),

])
X_train_num_prepared.head()
X_train_cat_prepared.head()
X_train_prepared = pd.concat([X_train_num_prepared,X_train_cat_prepared],axis=1)
X_train_prepared.head()
X_train_num = X_train[numerical_features]

X_train_cat = X_train[categorical_features]



for cat_feature in categorical_features:

    encode(X_train_cat[cat_feature])

    

warnings.filterwarnings(action='ignore',message='')    



X_train_num_prep = num_pipeline.fit_transform(X_train_num)

X_train_num_prepared = pd.DataFrame(X_train_num_prep,

                                   columns=list(X_train_num.columns),index=X_train_num.index)



X_train_cat_imp = cat_pipeline.fit_transform(X_train_cat)

X_train_cat_prepared = pd.DataFrame(X_train_cat_imp,

                                    columns=list(X_train_cat.columns),index=X_train_cat.index)



X_train_prepared = pd.concat([X_train_num_prepared,X_train_cat_prepared],axis=1)
X_valid_num = X_valid[numerical_features]

X_valid_cat = X_valid[categorical_features]



for cat_feature in categorical_features:

    encode(X_valid_cat[cat_feature])

    

X_valid_num_prep = num_pipeline.transform(X_valid_num)

X_valid_num_prepared = pd.DataFrame(X_valid_num_prep,

                                   columns=list(X_valid_num.columns),index=X_valid_num.index)



X_valid_cat_imp = cat_pipeline.transform(X_valid_cat)

X_valid_cat_prepared = pd.DataFrame(X_valid_cat_imp,

                                    columns=list(X_valid_cat.columns),index=X_valid_cat.index)



X_valid_prepared = pd.concat([X_valid_num_prepared,X_valid_cat_prepared],axis=1)
X_valid_prepared.head()
### Prepare the test dataset

X_test_num = X_test[numerical_features]

X_test_cat = X_test[categorical_features]



for cat_feature in categorical_features:

    encode(X_test_cat[cat_feature])

    

X_test_num_prep = num_pipeline.transform(X_test_num)

X_test_num_prepared = pd.DataFrame(X_test_num_prep,

                                   columns=list(X_test_num.columns),index=X_test_num.index)



X_test_cat_imp = cat_pipeline.transform(X_test_cat)

X_test_cat_prepared = pd.DataFrame(X_test_cat_imp,

                                   columns=list(X_test_cat.columns),index=X_test_cat.index)



X_test_prepared = pd.concat([X_test_num_prepared,X_test_cat_prepared],axis=1)
X_test_prepared.head()
print(X_train_prepared.shape,X_valid_prepared.shape,X_test_prepared.shape)
pd.Series(X_train_prepared.columns)[5:30]
## Custom Attribute Class Transformer

from sklearn.base import BaseEstimator,TransformerMixin



BsmtFinSF1_idx, BsmtFinSF2_idx ,BsmtUnfSF_idx, TotalBsmtSF_idx = 3,4,5,6

FstFlrSF_idx, SndFlrSF_idx, GrLivArea_idx = 7,8,10

BsmtFullBath_idx, BsmtHalfBath_idx = 11,12

FullBath_idx, HalfBath_idx = 13,14

BedroomAbvGr_idx, KitchenAbvGr_idx, TotRmsAbvGrd_idx = 15,16,17

GarageCars_idx, GarageArea_idx = 19,20

OverallQual_idx, OverallCond_idx =41,42

BsmtQual_idx, BsmtCond_idx = 53,54 

GarageQual_idx, GarageCond_idx = 68,69







class CustomAttribsAdder(BaseEstimator,TransformerMixin):

    def __init__(self,trans=True):

        self.trans = trans

    def fit(self,X,y=None):

        return self

    def transform(self,X):

        if self.trans:

            

            X[:,GrLivArea_idx] = X[:,GrLivArea_idx] * 3 

            X[:, OverallQual_idx] = X[:, OverallQual_idx] * 3

            

            overall_quality = X[:, OverallQual_idx] * X[:, OverallCond_idx]

            garage_quality = X[:,GarageQual_idx] * X[:,GarageCond_idx]

            bsmt_quality = X[:,BsmtQual_idx] * X[:,BsmtCond_idx]

            

            #garage_area_per_car = X[:, GarageCars_idx] + X[:,GarageArea_idx]

            

            rooms_above_ground = X[:,TotRmsAbvGrd_idx] + X[:,KitchenAbvGr_idx] + X[:,BedroomAbvGr_idx]

            

            full_sqft = X[:,GrLivArea_idx] + X[:,FstFlrSF_idx] + X[:,SndFlrSF_idx]

            

            bsmt_fin = (X[:,BsmtFinSF1_idx] + X[:,BsmtFinSF2_idx]) / X[:,TotalBsmtSF_idx]

            bsmt_unfin = X[:,BsmtUnfSF_idx] / X[:,TotalBsmtSF_idx]

            

            total_bath_grd = X[:,FullBath_idx] + X[:,HalfBath_idx]

            total_bath_bsmt = X[:,BsmtFullBath_idx] + X[:,BsmtHalfBath_idx]

            return np.c_[X,overall_quality,garage_quality,bsmt_quality,rooms_above_ground,full_sqft,

                        bsmt_fin,bsmt_unfin,total_bath_grd,total_bath_bsmt]

        else:

            return X

        

        
attr_adder =  CustomAttribsAdder(trans=True)

X_train_attribs = attr_adder.transform(X_train_prepared.values)

X_valid_attribs = attr_adder.transform(X_valid_prepared.values)

X_test_attribs = attr_adder.transform(X_test_prepared.values)
X_train_prepared = pd.DataFrame(X_train_attribs,

                                columns=list(X_train_prepared.columns)+['overall_quality','garage_quality','bsmt_quality','rooms_above_ground','full_sqft',\

                                                                        'bsmt_fin','bsmt_unfin','total_bath_grd','total_bath_bsmt'],index=X_train_prepared.index)
X_valid_prepared = pd.DataFrame(X_valid_attribs,

                                columns=list(X_valid_prepared.columns)+['overall_quality','garage_quality','bsmt_quality','rooms_above_ground','full_sqft',\

                                                                        'bsmt_fin','bsmt_unfin','total_bath_grd','total_bath_bsmt'],index=X_valid_prepared.index)
X_test_prepared = pd.DataFrame(X_test_attribs,

                                columns=list(X_test_prepared.columns)+['overall_quality','garage_quality','bsmt_quality','rooms_above_ground','full_sqft',\

                                                                        'bsmt_fin','bsmt_unfin','total_bath_grd','total_bath_bsmt'],index=X_test_prepared.index)
# remove the features - try out with various combinations 

drop_list_2 = ['OverallCond','GarageArea','GarageCond','BsmtCond',\

               'BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','1stFlrSF','2ndFlrSF',\

              'TotalBsmtSF','BsmtUnfSF','HalfBath','BsmtHalfBath',\

              'FullBath','BsmtFullBath']



#drop_list_2 = ['YearBuilt', 'OverallCond','GarageCars',\

#              'BedroomAbvGr','KitchenAbvGr','1stFlrSF','2ndFlrSF']



# call the function defined earlier to drop the columns from the dataset

drop_feature(X_train_prepared,drop_list_2)

drop_feature(X_valid_prepared,drop_list_2)

drop_feature(X_test_prepared,drop_list_2)
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor

from sklearn.ensemble import StackingRegressor

from sklearn.svm import SVR

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error,mean_squared_error

from sklearn.model_selection import cross_val_score, cross_val_predict,RepeatedKFold
rf_pipe = Pipeline([('RandomForest',RandomForestRegressor())])

gb_pipe = Pipeline([('GradientBoost',GradientBoostingRegressor())])

ada_pipe = Pipeline([('AdaBoost',AdaBoostRegressor())])

svr_pipe = Pipeline([('SupportVector',SVR())])
# get a list of models to evaluate

def get_models():

    models = dict()

    models['RF'] = rf_pipe                                        

    models['GB'] = gb_pipe

    models['SVR'] = svr_pipe

    return models
# evaluate a given model using k fold cross validation

def evaluate_model(model,X,y):

    cv = RepeatedKFold(n_splits=10,n_repeats=3,random_state=42)

    scores = cross_val_score(model,X,y,scoring='neg_mean_squared_error',cv=cv,n_jobs=-1,error_score='raise')

    return scores
# compare machine learning models for comparison 

models = get_models()

# evaluate the models and store the results

results,names = list(),list()



for name,model in models.items():

    scores = evaluate_model(model, X_train_prepared, y_train)

    results.append(scores)

    names.append(name)

    print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))

# get a stacking ensemble of models

def get_stacking():

    # define the base models

    level0 =list()

    level0.append(('RF',RandomForestRegressor(random_state=42)))

    level0.append(('GB',GradientBoostingRegressor(random_state=42)))

    level0.append(('AdaBoost',AdaBoostRegressor()))

    #define the meta learner model 

    level1=LinearRegression()

    #define the stacking ensemble

    model = StackingRegressor(estimators=level0,final_estimator=level1,cv=5)

    return model

    
stack_model = get_stacking()
stack_model.fit(X_train_prepared,y_train)

predictions_stack_model = stack_model.predict(X_valid_prepared)

print(f'RMSE = {np.sqrt(mean_squared_error(y_valid,predictions_stack_model))}')
rf_reg = RandomForestRegressor(random_state=42) 

rf_reg.fit(X_train_prepared,y_train)

predictions_rf_reg = rf_reg.predict(X_valid_prepared)

print(f'RMSE = {np.sqrt(mean_squared_error(y_valid,predictions_rf_reg))}')
gb_reg = GradientBoostingRegressor(random_state=42)

gb_reg.fit(X_train_prepared,y_train)

predictions_gb_reg = gb_reg.predict(X_valid_prepared)

print(f'RMSE = {np.sqrt(mean_squared_error(y_valid,predictions_gb_reg))}')
ada_reg = AdaBoostRegressor(random_state=42) 

ada_reg.fit(X_train_prepared,y_train)

predictions_ada_reg = ada_reg.predict(X_valid_prepared)

print(f'RMSE = {np.sqrt(mean_squared_error(y_valid,predictions_ada_reg))}')
# import the support vector regressor 

from sklearn.svm import SVR 

np.random.seed(42)

svr_reg = SVR()

svr_reg.fit(X_train_prepared,y_train)

valid_predict_svr_reg = svr_reg.predict(X_valid_prepared)

print(f'RMSE = {np.sqrt(mean_squared_error(y_valid,valid_predict_svr_reg))}')
rf_pipe = Pipeline([('RandomForest',RandomForestRegressor())])

gb_pipe = Pipeline([('GradientBoost',GradientBoostingRegressor())])

ada_pipe = Pipeline([('AdaBoost',AdaBoostRegressor())])

svr_pipe = Pipeline([('SupportVector',SVR())])
# Cross Validation

from sklearn.model_selection import cross_val_score, cross_val_predict



forest_scores = cross_val_score(rf_pipe,X_train_prepared,y_train,

                               scoring='neg_mean_squared_error',cv=10)

forest_rmse_scores = np.sqrt(-forest_scores)

print('Cross Validation Random Forest RMSE = {}'.format(forest_rmse_scores))

print (f'Mean is {np.mean(forest_rmse_scores)} Std. Deviation = {np.std(forest_rmse_scores)}')
#Cross validation on The gradient Boosting regressor

gb_scores = cross_val_score(gb_pipe,X_train_prepared,y_train,

                               scoring='neg_mean_squared_error',cv=10)

gb_rmse_scores = np.sqrt(-gb_scores)

print('Cross Validation Gradient Boosting RMSE = {}'.format(gb_rmse_scores))

print (f'Mean is {np.mean(gb_rmse_scores)} Std. Deviation = {np.std(gb_rmse_scores)}')
# cross validation on the support vector regressor

svm_scores = cross_val_score(svr_pipe,X_train_prepared,y_train,scoring='neg_mean_squared_error',

                            cv=10)

svm_scores = np.sqrt(-svm_scores)

print('Cross Validation Support Vector Regression RMSE = {}'.format(svm_scores))

print (f'Mean is {np.mean(svm_scores)} Std. Deviation = {np.std(svm_scores)}')

#cross validation on the AdaBoost regressor

ada_scores = cross_val_score(ada_pipe,X_train_prepared,y_train,scoring='neg_mean_squared_error',

                            cv=10)

ada_scores = np.sqrt(-ada_scores)

print('Cross Validation AdaBoost Regression RMSE = {}'.format(ada_scores))

print (f'Mean is {np.mean(ada_scores)} Std. Deviation = {np.std(ada_scores)}')
# Import Grid Search

from sklearn.model_selection import GridSearchCV
gb_reg = GradientBoostingRegressor()

gb_param_grid = [{'n_estimators':[100,200,300],'max_features':[8,16,32,64],

                 'max_depth':[3,5,7]}]



grid_search_gb = GridSearchCV(gb_reg,gb_param_grid,cv=5,scoring='neg_mean_squared_error',

                              return_train_score=True)

grid_search_gb.fit(X_train_prepared,y_train)
gb_best_reg = sklearn.base.clone(grid_search_gb.best_estimator_)

gb_best_reg.fit(X_train_prepared,y_train)

valid_pred_gb = gb_best_reg.predict(X_valid_prepared)

print('RMSE={}'.format(np.sqrt(mean_squared_error(y_valid,valid_pred_gb))))
rf_reg = RandomForestRegressor(random_state=42)

rf_param_grid = [

    # try 12 (3×4) combinations of hyperparameters

    {'n_estimators': [100,200,300], 'max_features': [8,16,32,64]},

    # then try 6 (2×3) combinations with bootstrap set as False

    {'bootstrap': [False], 'n_estimators': [100, 200], 'max_features': [2, 3, 4]},

  ]



grid_search_rf = GridSearchCV(rf_reg,rf_param_grid,cv=5,scoring='neg_mean_squared_error',

                              return_train_score=True)

grid_search_rf.fit(X_train_prepared,y_train)
# best estimator based on Grid search

grid_search_rf.best_estimator_
# print the test scores for each of the tried combination

cvres = grid_search_rf.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)
# check the RMSE score on the validation data based on the grid search best estimator

rf_reg_grid = sklearn.base.clone(grid_search_rf.best_estimator_ )

rf_reg_grid.fit(X_train_prepared,y_train)

pred_valid_rf_grid = rf_reg_grid.predict(X_valid_prepared)

print(f'RMSE = {np.sqrt(mean_squared_error(y_valid,pred_valid_rf_grid))}')
feature_importance = rf_reg_grid.feature_importances_

# create a basic plot - improvement , plot tge feature names on the x-axis 

plt.plot(feature_importance)
X_train_prepared.columns[65:70]
# import the pca package

from sklearn.decomposition import PCA

pca = PCA()

pca.fit(X_train_prepared)

cumsum = np.cumsum(pca.explained_variance_ratio_)

d = np.argmax(cumsum >= 0.95) + 1

print (f'{d}')
pca = PCA(n_components = d)

X_train_reduced = pca.fit_transform(X_train_prepared)

X_valid_reduced = pca.transform(X_valid_prepared)

X_test_reduced = pca.transform(X_test_prepared)
rf_pca = RandomForestRegressor(random_state=42)

rf_pca.fit(X_train_reduced,y_train)

valid_pred = rf_pca.predict(X_valid_reduced)

print(f'RMSE on validation data = {np.sqrt(mean_squared_error(y_valid,valid_pred))}')
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint
param_distribs = {'n_estimators':randint(100,300),

                 'max_features':randint(16,67),

                 'max_depth':randint(3,6)}
rf_reg_rand =  RandomForestRegressor(random_state=42)

rand_search = RandomizedSearchCV(rf_reg_rand,param_distributions=param_distribs,cv=10,

                                 scoring='neg_mean_squared_error',random_state=42,n_iter=10)



rand_search.fit(X_train_prepared,y_train)
rand_search.best_estimator_
rf_reg_rand_best = sklearn.base.clone(rand_search.best_estimator_)
rf_reg_rand_best.fit(X_train_prepared,y_train)

valid_pred = rf_reg_rand_best.predict(X_valid_prepared)
print(f'RMSE = {np.sqrt(mean_squared_error(y_valid,valid_pred))}')
final_pred = rf_reg_grid.predict(X_test_prepared)

price_arr = np.exp(final_pred)

output = pd.DataFrame({"Id":save_id['Id'], "SalePrice":price_arr})

output.to_csv('submission_rf_001.csv', index=False)