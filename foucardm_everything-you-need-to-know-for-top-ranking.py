from IPython.display import Image

Image("../input/image/image.jpg")
#import some necessary librairies



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)





from scipy import stats

from scipy.stats import norm, skew #for some statistics





pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the directory
#Now let's import and put the train and test datasets in  pandas dataframe



df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
#check the numbers of samples and features

print("The train data size before dropping Id feature is : {} ".format(df_train.shape))

print("The test data size before dropping Id feature is : {} ".format(df_test.shape))



#Save the 'Id' column

train_ID = df_train['Id']

test_ID = df_test['Id']



#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

df_train.drop("Id", axis = 1, inplace = True)

df_test.drop("Id", axis = 1, inplace = True)



#check again the data size after dropping the 'Id' variable

print("\nThe train data size after dropping Id feature is : {} ".format(df_train.shape)) 

print("The test data size after dropping Id feature is : {} ".format(df_test.shape))
plt.figure()

plt.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'], alpha = 0.5)

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.title('Spot the outliers')

plt.show()
df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<210000)].index)
plt.figure()

plt.scatter(x = df_train['GrLivArea'], y = df_train['SalePrice'], alpha = 0.5)

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.title('Spot the outliers')

plt.show()
sns.distplot(df_train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(df_train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)

plt.show()
#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

df_train["SalePrice"] = np.log1p(df_train["SalePrice"])



#Check the new distribution 

sns.distplot(df_train['SalePrice'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(df_train['SalePrice'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('SalePrice distribution')



#Get also the QQ-plot

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)

plt.show()
train_feature_ext = df_train['Exterior1st'].value_counts().index

test_feature_ext = df_test['Exterior1st'].value_counts().index



print( 'The number of catogries in the train set for the "Extorior1st" feature is:  {:.0f}'.format(len(train_feature_ext)))

print( '\nThe number of catogries in the test set for the "Extorior1st" feature is:  {:.0f}'.format(len(test_feature_ext)))
ntrain = df_train.shape[0]

ntest = df_test.shape[0]



y_train = df_train['SalePrice']



all_data = pd.concat((df_train, df_test)).reset_index(drop=True) # all_data contains all the data from the train and test set expect for the SalePrice

all_data.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))
nbr_missing = all_data.isnull().sum()

all_data_na = (nbr_missing / len(all_data)) * 100



all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:40]

nbr_missing = nbr_missing.drop(nbr_missing[nbr_missing == 0].index).sort_values(ascending=False)[:40]



missing_data = pd.DataFrame({'Percent missing' :all_data_na, 'Missing values' :nbr_missing})

missing_data.head(40)
plt.figure(figsize=(20,10))

sns.heatmap(all_data.isna(), cbar=False, cmap = 'plasma')

plt.title('Visual representation of the missing values in the dataset')
from sklearn.base import BaseEstimator, TransformerMixin



class NaN_transformer(BaseEstimator, TransformerMixin):

    '''

    This transformer fill the missing values in the given dataset,

    or delete the features with too much missing values.

    '''



    def fit( self, X, y=None ):

        return self 



    def transform(self, X, y=None):

        

        X["PoolQC"] = X["PoolQC"].fillna("None") #PoolQC : data description says NA means "No Pool"

        X["MiscFeature"] = X["MiscFeature"].fillna("None") #MiscFeature : data description says NA means "no misc feature"

        X["Alley"] = X["Alley"].fillna("None") #Alley : data description says NA means "no alley access"

        X["Fence"] = X["Fence"].fillna("None") #Fence : data description says NA means "no fence""

        X["FireplaceQu"] = X["FireplaceQu"].fillna("None") #FireplaceQu : data description says NA means "no fireplace"

        

        for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

            X[col] = X[col].fillna('None')

        

        X['GarageYrBlt'] = X['GarageYrBlt'].fillna(X['GarageYrBlt'].median()) # We fill the missing data with the medianvalue of the other variables

        X['GarageCars'] = X['GarageCars'].fillna(0) # We fill the missing data with 0 cars

        X = X.drop('GarageArea', axis = 1) #'GarageCars' and 'GarageArea' are correlated variables, we'll keep only one

        

        X["LotFrontage"] = X.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

        

        for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

            X[col] = X[col].fillna(0) # These samples are likely to have no basement, since the fill with zeros

            

        for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

            X[col] = X[col].fillna('None') # NaN means no basement

        

        X["MasVnrType"] = X["MasVnrType"].fillna("None")

        X["MasVnrArea"] = X["MasVnrArea"].fillna(0)

        

        X['MSZoning'] = X['MSZoning'].fillna(all_data['MSZoning'].mode()[0]) # Filling with the most common category

        

        X = X.drop(['Utilities'], axis=1)

        

        X["Functional"] = X["Functional"].fillna("Typ") #data description says NA means typical

        

        X['Electrical'] = X['Electrical'].fillna(X['Electrical'].mode()[0]) # Filling with the most common category

        X['KitchenQual'] = X['KitchenQual'].fillna(X['KitchenQual'].mode()[0])

        X['Exterior1st'] = X['Exterior1st'].fillna(X['Exterior1st'].mode()[0])

        X['Exterior2nd'] = X['Exterior2nd'].fillna(X['Exterior2nd'].mode()[0])

        X['SaleType'] = X['SaleType'].fillna(X['SaleType'].mode()[0])

        

        X['MSSubClass'] =  X['MSSubClass'].fillna("None")

        

        X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF'] # Adding a feature

        

        return(X)
tmp = all_data.copy() # make a copy of the dataframe to show an exemple



na_tr = NaN_transformer() #We pass nothing inside this transformer



tmp = na_tr.fit_transform(tmp)
plt.figure(figsize=(20,10))

sns.heatmap(tmp.isna(), cbar=False, cmap = 'plasma')

plt.title('Visual representation of the missing values in the dataset')
class ToCategorical_transformer(BaseEstimator, TransformerMixin):

    '''

    This transformer selects either numerical or categorical features.

    In this way we can build separate pipelines for separate data types.

    '''



    def fit( self, X, y=None ):

        return self 



    def transform(self, X, y=None):

        

        #MSSubClass=The building class

        X['MSSubClass'] = X['MSSubClass'].apply(str)



        #Changing OverallCond into a categorical variable

        X['OverallCond'] = X['OverallCond'].astype(str)



        #Year and month sold are transformed into categorical features.

        X['YrSold'] = X['YrSold'].astype(str)

        X['MoSold'] = X['MoSold'].astype(str)

        

        return(X)
tiny_tmp = all_data[['MSSubClass','OverallCond','YrSold','MoSold']]

tiny_tmp.dtypes
toCat_tr = ToCategorical_transformer()



tiny_tmp = toCat_tr.fit_transform(tiny_tmp)



tiny_tmp.dtypes
from sklearn.preprocessing import LabelEncoder



cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')



class Personal_LabelEncoder(BaseEstimator, TransformerMixin):

    '''

    This transformer selects either numerical or categorical features.

    In this way we can build separate pipelines for separate data types.

    '''

    def __init__(self, cols): #Now we need the __init__ method since we'll build variables we'll use in this transformer

        self.cols = cols # This variable will contains all the columns that will be transformed

        self.encoder = None # Since the encoder is already a transformer that exists, we'll define it in the fit method

    

    def fit( self, X, y=None ):

        self.encoder = LabelEncoder() # We fit the encoder to the data

        return self 



    def transform(self, X, y=None):

        for c in self.cols:

            X[c] = self.encoder.fit_transform(list(X[c].values)) # Here we actually do the transformation

        

        return(X)
tmp[['FireplaceQu', 'BsmtQual', 'BsmtCond']].head()
encoder = Personal_LabelEncoder(cols = cols)



tmp = encoder.fit_transform(tmp)



tmp[['FireplaceQu', 'BsmtQual', 'BsmtCond']].head()
from sklearn.base import BaseEstimator, TransformerMixin



class Dummy_transform(BaseEstimator, TransformerMixin):

    '''

    This transformer selects either numerical or categorical features.

    In this way we can build separate pipelines for separate data types.

    '''



    def fit( self, X, y=None ):

        return self 



    def transform(self, X, y=None):

        X = pd.get_dummies(X)

        return(X)
from sklearn.pipeline import Pipeline



full_pipe= Pipeline([

                ('NaN transformer', NaN_transformer()),

                ('To categorical Transform', ToCategorical_transformer()),

                ('Label Encoder', Personal_LabelEncoder(cols = cols)),

                ('Dummy transformation', Dummy_transform())

            ])



all_data_tr = all_data.copy()



all_data_tr = full_pipe.fit_transform(all_data_tr)
all_data_tr.head()
all_data_tr.shape #We have way more columns due to the dummy transformation
X_train = all_data_tr[:ntrain]

X_test = all_data_tr[ntrain:]
X_train.head()
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import Ridge, RidgeCV

from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LinearRegression, BayesianRidge

from sklearn.svm import SVR

from mlxtend.regressor import StackingCVRegressor

import lightgbm as lgb

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor



from sklearn.pipeline import make_pipeline, Pipeline

from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.metrics import mean_squared_error
# Setup cross validation folds

kf = KFold(n_splits=6, random_state=42, shuffle=True) #Because there are no shuffle implemented with the cross validation, we define here the folds for the cross validation
# Define error metrics

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def cv_rmse(model, X=X_train):

    rmse = np.sqrt(-cross_val_score(model, X.values, y_train.values, scoring="neg_mean_squared_error", cv=kf))

    return (rmse)
linear = make_pipeline(RobustScaler(),LinearRegression())
# Here the lambda parameter is called "alpha"

lasso = make_pipeline(StandardScaler(), Lasso(alpha =0.0035, random_state=42)) # Here lasso_pipe has to be used just like a regular model.
# But...what if we want to try different scaler, how to use GridSearchCV with a pipeline ???

# Check this out:



lasso_pipe = Pipeline([ #We have to create a pipeline, just like before, but create a "None" value for the scaler that we will fill later.

    ('scaler', None),

    ('classifier', Lasso())

])



alpha_params = [0.0025,0.003,0.0035]



# Now we have to define a list of dictionnary. Within each dictionnary we'll call a different scaler and the same alpha parameters

param_grid = [

    {

        'scaler': [StandardScaler()],

        'classifier__alpha': alpha_params,

    },

    {

        'scaler': [RobustScaler()],

        'classifier__alpha': alpha_params,

    },

]



#Then, well, you already know how to build a grid !



lasso_grid = GridSearchCV(lasso_pipe, param_grid, cv=kf)

lasso_grid.fit(X_train, y_train)



print('the best score obtained is {} with the parameters {}'.format(lasso_grid.best_score_,lasso_grid.best_params_))
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0045, l1_ratio=.9, random_state=3))
ENet_pipe = Pipeline([ #We have to create a pipeline, just like before, but create a "None" value for the scaler that we will fill later.

    ('scaler', None),

    ('classifier', ElasticNet())

])



alpha_params = [0.00045,0.0005,0.00055]

l1_params = [0.85,0.86,0.87,0.88,0.89,0.9]



# Now we have to define a list of dictionnary. Within each dictionnary we'll call a different scaler and the same alpha parameters

param_grid = [

    {

        'scaler': [StandardScaler()],

        'classifier__alpha': alpha_params,

        'classifier__l1_ratio': l1_params,

    },

    {

        'scaler': [RobustScaler()],

        'classifier__alpha': alpha_params,

        'classifier__l1_ratio': l1_params,

    },

]



#Then, well, you already know how to build a grid !



ENet_grid = GridSearchCV(ENet_pipe, param_grid, cv=kf)

ENet_grid.fit(X_train, y_train)



print('the best score obtained is {} with the parameters {}'.format(ENet_grid.best_score_,ENet_grid.best_params_))
# Random Forest Regressor

rf = RandomForestRegressor(n_estimators=1200,

                          max_depth=15,

                          min_samples_split=5,

                          min_samples_leaf=5,

                          max_features=None,

                          oob_score=True,

                          random_state=42)
adar = AdaBoostRegressor(n_estimators=4000,

                         learning_rate=0.01,

                         loss='linear',

                        random_state=42,)
param_grid={

    'n_estimators':[4000],

    'learning_rate':[0.001,0.005,0.01],

    'loss':['linear','exponential']

    

    

}

adar_grid = RandomizedSearchCV(AdaBoostRegressor(random_state=42), param_grid, cv=kf) # Just use it like a regular GridSearchCV

adar_grid.fit(X_train,y_train)
adar_grid.best_score_ # This is how to access the best score calculated
adar_grid.best_params_ # This is how to get the parameters that leads to the best score
model_adar = adar_grid.best_estimator_ # This is how to create a new model with the best parameters found in the grid.
score = cv_rmse(model_adar)

print("model_adar: {:.4f} ({:.4f})".format(score.mean(), score.std()))

#scores['model_adar'] = (score.mean(), score.std())
gbr = GradientBoostingRegressor(n_estimators=3000,

                                learning_rate=0.01,

                                max_depth=4,

                                max_features='sqrt',

                                min_samples_leaf=15,

                                min_samples_split=10,

                                loss='huber',

                                random_state=42)
# XGBoost Regressor

xgboost = XGBRegressor(learning_rate=0.01,

                       n_estimators=7000,

                       max_depth=8,

                       min_child_weight=10,

                       gamma=0.6,

                       subsample=0.5,

                       colsample_bytree=0.7,

                       objective='reg:squarederror',

                       nthread=-1,

                       seed=42,

                       reg_alpha=0.006,

                       random_state=42)
# Light Gradient Boosting Regressor

lightgbm = LGBMRegressor(objective='regression', 

                       num_leaves=6,

                       learning_rate=0.01, 

                       n_estimators=7000,

                       max_bin=200, # This is the maximum number of bins the features will be bucketed in.

                       bagging_fraction=0.8,

                       bagging_freq=4, # Means: perform the bagging every 4 iterations

                       bagging_seed=42, # Equivalent of random_state but with bagging.

                       feature_fraction=0.2, # As usual, the faction of features that will be randomly selected to build trees at each iteration

                       feature_fraction_seed=42, # Equivalent of random_state but with the feature_fraction

                       min_sum_hessian_in_leaf = 11,

                       verbose=-1, # To shut down the messages

                       random_state=42)
# Support Vector Regressor

svr = make_pipeline(StandardScaler(), SVR(C= 5, epsilon= 0.0003, gamma=0.0003))
svr_pipe = Pipeline([ #We have to create a pipeline, just like before, but create a "None" value for the scaler that we will fill later.

    ('scaler', None),

    ('classifier', SVR())

])



C_params = [4,5,6]

epsilon_params = [0.0002,0.0003,0.0004,0.0005]

gamma_params = [0.00005,0.0001,0.0002,0.0003]



param_grid ={

        'scaler': [StandardScaler()],

        'classifier__C': C_params,

        'classifier__epsilon': epsilon_params,

        'classifier__gamma': gamma_params,

    }



svr_grid = GridSearchCV(svr_pipe, param_grid, cv=kf)

svr_grid.fit(X_train, y_train)



print('the best score obtained is {} with the parameters {}'.format(svr_grid.best_score_,svr_grid.best_params_))



model_svr = svr_grid.best_estimator_
score = cv_rmse(model_svr)

print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))

#scores['xgboost'] = (score.mean(), score.std())
#Stack up all the models above, optimized using xgboost

stack_gen = StackingCVRegressor(regressors=(xgboost, lightgbm, svr, ENet, lasso, rf),

                                meta_regressor=xgboost,

                                use_features_in_secondary=True)
scores = {}



score = cv_rmse(linear)

print("linear: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['linear'] = (score.mean(), score.std())
score = cv_rmse(lasso)

print("lasso: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['lasso'] = (score.mean(), score.std())
score = cv_rmse(ENet)

print("ENet: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['ENet'] = (score.mean(), score.std())
score = cv_rmse(lightgbm)

print("lightgbm: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['lightgbm'] = (score.mean(), score.std())
score = cv_rmse(xgboost)

print("xgboost: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['xgboost'] = (score.mean(), score.std())
score = cv_rmse(gbr)

print("gbr: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['gbr'] = (score.mean(), score.std())
score = cv_rmse(rf)

print("rf: {:.4f} ({:.4f})".format(score.mean(), score.std()))

scores['rf'] = (score.mean(), score.std())
print('stack_gen')

stack_gen_model = stack_gen.fit(np.array(X_train), np.array(y_train))
print('lightgbm')

lgb_model_full_data = lightgbm.fit(X_train, y_train)
print('xgboost')

xgb_model_full_data = xgboost.fit(X_train, y_train)
print('linear')

linear_model_full_data = linear.fit(X_train, y_train)
print('lasso')

lasso_model_full_data = lasso.fit(X_train, y_train)
print('Svr')

svr_model_full_data = svr.fit(X_train, y_train)
print('RandomForest')

rf_model_full_data = rf.fit(X_train, y_train )
print('GradientBoosting')

gbr_model_full_data = gbr.fit(X_train, y_train)
# Blend models in order to make the final predictions more robust to overfitting

def blended_predictions(x1,x2,x3,x4,x5,x6,x7,X):

    assert(x1+x2+x3+x4+x5+x6+x7 == 1.0, 'Sum is not equal to 1')

    return ((x1 * lasso_model_full_data.predict(X)) + \

            (x2 * svr_model_full_data.predict(X)) + \

            (x3 * gbr_model_full_data.predict(X)) + \

            (x4 * xgb_model_full_data.predict(X)) + \

            (x5 * lgb_model_full_data.predict(X)) + \

            (x6 * rf_model_full_data.predict(X)) + \

            (x7 * stack_gen_model.predict(np.array(X))))
x1, x2, x3, x4, x5, x6, x7 = 0.3, 0.1, 0.05, 0.1, 0.3, 0.1, 0.05
# Get final precitions from the blended model

blended_score = rmsle(y_train, blended_predictions(x1,x2,x3,x4,x5,x6,x7,X_train))

scores['blended'] = (blended_score, 0)

print('RMSLE score on train data:')

print(blended_score)
# Plot the predictions for each model

sns.set_style("white")

fig = plt.figure(figsize=(24, 12))



ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()], markers=['o'], linestyles=['-'])

for i, score in enumerate(scores.values()):

    ax.text(i, score[0] + 0.002, '{:.6f}'.format(score[0]), horizontalalignment='left', size='large', color='black', weight='semibold')



plt.ylabel('Score (RMSE)', size=20, labelpad=12.5)

plt.xlabel('Model', size=20, labelpad=12.5)

plt.tick_params(axis='x', labelsize=13.5)

plt.tick_params(axis='y', labelsize=12.5)



plt.title('Scores of Models', size=20)



plt.show()
sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = np.expm1(blended_predictions(x1,x2,x3,x4,x5,x6,x7,X_test))

sub.to_csv('submission_house_price.csv',index=False)