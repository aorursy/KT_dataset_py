import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



movie=pd.read_csv(r'/kaggle/input/cinema-movie/Movie_regression.csv')

movie.head()
moviecbr=movie.copy()
movie.info()
movie.shape
movie.describe()
round((movie.isnull().sum() * 100 / len(movie)),2)


sns.distplot(movie['Time_taken'])

plt.show()
#Encode categorical data

dummy = pd.get_dummies(movie[["Genre","3D_available"]]).iloc[:,:-1]

movie = pd.concat([movie,dummy], axis=1)

movie = movie.drop(["Genre","3D_available"], axis=1)

movie.shape
from sklearn.experimental import enable_iterative_imputer

# now you can import normally from sklearn.impute

from sklearn.impute import IterativeImputer

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

it = IterativeImputer(estimator=LinearRegression())

newdata_lr = pd.DataFrame(it.fit_transform(movie))

newdata_lr.columns = movie.columns

newdata_lr.head()
import scipy.stats as stats

stats.ttest_ind(newdata_lr.Time_taken,movie.Time_taken,nan_policy='omit')
from sklearn.experimental import enable_iterative_imputer

# now you can import normally from sklearn.impute

from sklearn.impute import IterativeImputer

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

it = IterativeImputer(estimator=RandomForestRegressor(random_state=42))

newdata_rfr = pd.DataFrame(it.fit_transform(movie))

newdata_rfr.columns = movie.columns

newdata_rfr.head()
import scipy.stats as stats

stats.ttest_ind(newdata_rfr.Time_taken,movie.Time_taken,nan_policy='omit')
from sklearn.experimental import enable_iterative_imputer

# now you can import normally from sklearn.impute

from sklearn.impute import IterativeImputer

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn import tree

it = IterativeImputer(estimator=tree.DecisionTreeRegressor(random_state=42))

newdata_bg = pd.DataFrame(it.fit_transform(movie))

newdata_bg.columns = movie.columns

newdata_bg.head()
import scipy.stats as stats

stats.ttest_ind(newdata_bg.Time_taken,movie.Time_taken,nan_policy='omit')
from sklearn.experimental import enable_iterative_imputer

# now you can import normally from sklearn.impute

from sklearn.impute import IterativeImputer

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn import tree

it = IterativeImputer(estimator=AdaBoostRegressor(random_state=42))

newdata_abc = pd.DataFrame(it.fit_transform(movie))

newdata_abc.columns = movie.columns

newdata_abc.head()
import scipy.stats as stats

stats.ttest_ind(newdata_abc.Time_taken,movie.Time_taken,nan_policy='omit')
from sklearn.experimental import enable_iterative_imputer

# now you can import normally from sklearn.impute

from sklearn.impute import IterativeImputer

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn import tree

it = IterativeImputer(estimator=GradientBoostingRegressor(random_state=42))

newdata_gbr = pd.DataFrame(it.fit_transform(movie))

newdata_gbr.columns = movie.columns

newdata_gbr.head()
import scipy.stats as stats

stats.ttest_ind(newdata_gbr.Time_taken,movie.Time_taken,nan_policy='omit')
from sklearn.experimental import enable_iterative_imputer

# now you can import normally from sklearn.impute

from sklearn.impute import IterativeImputer

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

import xgboost as xgb

from sklearn import tree

it = IterativeImputer(estimator=xgb.XGBRegressor(random_state=42))

newdata_xgb = pd.DataFrame(it.fit_transform(movie))

newdata_xgb.columns = movie.columns

newdata_xgb.head()
import scipy.stats as stats

stats.ttest_ind(newdata_xgb.Time_taken,movie.Time_taken,nan_policy='omit')
# Compare with original v/s modified 
movie.Time_taken.describe(),newdata_lr.Time_taken.describe()
movie.Time_taken.describe(),newdata_rfr.Time_taken.describe()
movie.Time_taken.describe(),newdata_bg.Time_taken.describe()
movie.Time_taken.describe(),newdata_abc.Time_taken.describe()
movie.Time_taken.describe(),newdata_gbr.Time_taken.describe()
movie.Time_taken.describe(),newdata_xgb.Time_taken.describe()
movie=newdata_abc
moviecbr = moviecbr.assign(Time_taken=movie['Time_taken'])
movie.info()
import pandas_profiling as pp 

profile = pp.ProfileReport(movie) 

profile
movie.hist(figsize=(32,20),bins=50)

plt.xticks(rotation=90)

plt.show()
from sklearn.model_selection import train_test_split



# We specify this so that the train and test data set always have the same rows, respectively

np.random.seed(0)

df_train, df_test = train_test_split(movie, train_size = 0.7, random_state = 42)
df_train.shape, df_test.shape
X_train=df_train.drop('Collection',axis=1)

y_train=df_train['Collection']

X_test=df_test.drop('Collection',axis=1)

y_test=df_test['Collection']
X_train.info()
X_train.columns
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train[['Marketing expense', 'Production expense', 'Multiplex coverage',

         'Budget', 'Movie_length', 'Lead_ Actor_Rating', 'Lead_Actress_rating',

         'Director_rating', 'Producer_rating', 'Critic_rating', 'Trailer_views',

         'Time_taken', 'Twitter_hastags', 'Avg_age_actors', 'Num_multiplex']]= scaler.fit_transform(X_train[['Marketing expense', 'Production expense', 'Multiplex coverage',

                                                        'Budget', 'Movie_length', 'Lead_ Actor_Rating', 'Lead_Actress_rating',

                                                        'Director_rating', 'Producer_rating', 'Critic_rating', 'Trailer_views',

                                                        'Time_taken', 'Twitter_hastags', 'Avg_age_actors', 'Num_multiplex']])

X_train.head()
X_test[['Marketing expense', 'Production expense', 'Multiplex coverage',

         'Budget', 'Movie_length', 'Lead_ Actor_Rating', 'Lead_Actress_rating',

         'Director_rating', 'Producer_rating', 'Critic_rating', 'Trailer_views',

         'Time_taken', 'Twitter_hastags', 'Avg_age_actors', 'Num_multiplex']]= scaler.transform(X_test[['Marketing expense', 'Production expense', 'Multiplex coverage',

                                                        'Budget', 'Movie_length', 'Lead_ Actor_Rating', 'Lead_Actress_rating',

                                                        'Director_rating', 'Producer_rating', 'Critic_rating', 'Trailer_views',

                                                        'Time_taken', 'Twitter_hastags', 'Avg_age_actors', 'Num_multiplex']])

X_test.head()
X_test.shape,X_train.shape
# Calculate the VIFs for the new model

import statsmodels

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train = X_train.drop(["Lead_Actress_rating"], axis = 1)

X_test = X_test.drop(["Lead_Actress_rating"], axis = 1)
# Calculate the VIFs for the new model

import statsmodels

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train = X_train.drop(["Lead_ Actor_Rating"], axis = 1)

X_test = X_test.drop(["Lead_ Actor_Rating"], axis = 1)
# Calculate the VIFs for the new model

import statsmodels

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train = X_train.drop(["Producer_rating"], axis = 1)

X_test = X_test.drop(["Producer_rating"], axis = 1)
import statsmodels

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train = X_train.drop(["Multiplex coverage"], axis = 1)

X_test = X_test.drop(["Multiplex coverage"], axis = 1)
import statsmodels

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
import xgboost as xgb

model=xgb.XGBRegressor()

model.fit(X_train, y_train)

score_xgb=model.score(X_test,y_test)

print("Score XGBRegressor :", score_xgb)
from sklearn.metrics import mean_squared_error

rmse_xgb=mean_squared_error(model.predict(X_test),y_test)**0.5

print('RMSE XGBRegressor :',rmse_xgb)
from sklearn.metrics import mean_squared_log_error

RMSLE_xgb=np.sqrt(mean_squared_log_error( y_test, model.predict(X_test) ))

print("RMSLE for XGBRegressor :",RMSLE_xgb)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

X_trainlr=X_train.copy()

import statsmodels.api as sm

X_train_lm = sm.add_constant(X_trainlr)

lr = sm.OLS(y_train, X_trainlr).fit()

lr.summary()
X_train1 = X_trainlr.drop(["Avg_age_actors"], axis = 1)

X_train_lm = sm.add_constant(X_train1)

lr= sm.OLS(y_train, X_train_lm).fit()

lr.summary()
X_train1 = X_train1.drop(["Num_multiplex"], axis = 1)

X_train_lm = sm.add_constant(X_train1)

lr= sm.OLS(y_train, X_train_lm).fit()

lr.summary()
X_train1 = X_train1.drop(["Time_taken"], axis = 1)

X_train_lm = sm.add_constant(X_train1)

lr= sm.OLS(y_train, X_train_lm).fit()

lr.summary()
X_train1 = X_train1.drop(["Twitter_hastags"], axis = 1)

X_train_lm = sm.add_constant(X_train1)

lr= sm.OLS(y_train, X_train_lm).fit()

lr.summary()
X_train1 = X_train1.drop(["Production expense"], axis = 1)

X_train_lm = sm.add_constant(X_train1)

lr= sm.OLS(y_train, X_train_lm).fit()

lr.summary()
X_train1 = X_train1.drop(["Movie_length"], axis = 1)

X_train_lm = sm.add_constant(X_train1)

lr= sm.OLS(y_train, X_train_lm).fit()

lr.summary()
X_train1 = X_train1.drop(["3D_available_NO"], axis = 1)

X_train_lm = sm.add_constant(X_train1)

lr= sm.OLS(y_train, X_train_lm).fit()

lr.summary()
X_testlr=X_test.copy()

col1=X_train1.columns

X_testlr=X_test[col1]

# Adding constant variable to test dataframe

X_testlr= sm.add_constant(X_testlr)
from sklearn.metrics import mean_squared_error

rmse_lr=mean_squared_error(lr.predict(X_testlr),y_test)**0.5

print('RMSE Linear Regression:',rmse_lr)

from sklearn.metrics import mean_squared_log_error

RMSLE_lr=np.sqrt(mean_squared_log_error( y_test, abs(lr.predict(X_testlr))))

print("RMSLE for Linear Regression:",RMSLE_lr)
from sklearn.ensemble import GradientBoostingRegressor

model= GradientBoostingRegressor(random_state=42)

model.fit(X_train, y_train)

score_gbc=model.score(X_test,y_test)

print("Score GradientBoostingRegressor:", score_gbc)
from sklearn.metrics import mean_squared_error

rmse_gbc=mean_squared_error(model.predict(X_test),y_test)**0.5

print('RMSE GradientBoostingRegressor :',rmse_gbc)
from sklearn.metrics import mean_squared_log_error

RMSLE_gbc=np.sqrt(mean_squared_log_error( y_test, model.predict(X_test) ))

print("RMSLE for GradientBoostingRegressor :",RMSLE_gbc)
from sklearn.ensemble import AdaBoostRegressor

model = AdaBoostRegressor()

model.fit(X_train, y_train)

score_abr=model.score(X_test,y_test)

print("Score AdaBoostRegressor:", score_abr)
from sklearn.metrics import mean_squared_error

rmse_abr=mean_squared_error(model.predict(X_test),y_test)**0.5

print('RMSE AdaBoostRegressor :',rmse_abr)
from sklearn.metrics import mean_squared_log_error

RMSLE_abr=np.sqrt(mean_squared_log_error( y_test, model.predict(X_test) ))

print("RMSLE for AdaBoostRegressor :",RMSLE_abr)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42)

model.fit(X_train, y_train)

score_rfr=model.score(X_test,y_test)

print("Score RandomForestRegressor:", score_rfr)
from sklearn.metrics import mean_squared_error

rmse_rfr=mean_squared_error(model.predict(X_test),y_test)**0.5

print('RMSE RandomForestRegressor :',rmse_rfr)
from sklearn.metrics import mean_squared_log_error

RMSLE_rfr=np.sqrt(mean_squared_log_error( y_test, model.predict(X_test) ))

print("RMSLE for RandomForestRegressor :",RMSLE_rfr)
from sklearn.ensemble import BaggingRegressor

model = BaggingRegressor(tree.DecisionTreeRegressor(random_state=1))

model.fit(X_train, y_train)

score_br=model.score(X_test,y_test)

print("Score BaggingRegressor:", score_br)
from sklearn.metrics import mean_squared_error

rmse_br=mean_squared_error(model.predict(X_test),y_test)**0.5

print('RMSE  BaggingRegressor :',rmse_br)
from sklearn.metrics import mean_squared_log_error

RMSLE_br=np.sqrt(mean_squared_log_error( y_test, model.predict(X_test) ))

print("RMSLE for BaggingRegressor :",RMSLE_br)
from sklearn.metrics import mean_squared_log_error

RMSLE_br=np.sqrt(mean_squared_log_error( y_test, model.predict(X_test) ))

print("RMSLE for BaggingRegressor :",RMSLE_br)
moviecbr['Collection']=moviecbr['Collection'].astype('float')

moviecbr['Num_multiplex']=moviecbr['Num_multiplex'].astype('float')

moviecbr['Avg_age_actors']=moviecbr['Avg_age_actors'].astype('float')

moviecbr['Trailer_views']=moviecbr['Trailer_views'].astype('float')
from sklearn.model_selection import train_test_split



# We specify this so that the train and test data set always have the same rows, respectively

np.random.seed(0)

dftrain, dftest = train_test_split(moviecbr, train_size = 0.7, random_state = 42)
dftrain.shape, dftest.shape
Xtrain=dftrain.drop('Collection',axis=1)

ytrain=dftrain['Collection']

Xtest=dftest.drop('Collection',axis=1)

ytest=dftest['Collection']
Xtrain.info()
Xtrain.columns
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

Xtrain[['Marketing expense', 'Production expense', 'Multiplex coverage',

         'Budget', 'Movie_length', 'Lead_ Actor_Rating', 'Lead_Actress_rating',

         'Director_rating', 'Producer_rating', 'Critic_rating', 'Trailer_views',

         'Time_taken', 'Twitter_hastags', 'Avg_age_actors', 'Num_multiplex']]= scaler.fit_transform(Xtrain[['Marketing expense', 'Production expense', 'Multiplex coverage',

                                                        'Budget', 'Movie_length', 'Lead_ Actor_Rating', 'Lead_Actress_rating',

                                                        'Director_rating', 'Producer_rating', 'Critic_rating', 'Trailer_views',

                                                        'Time_taken', 'Twitter_hastags', 'Avg_age_actors', 'Num_multiplex']])

Xtrain.head()
Xtest[['Marketing expense', 'Production expense', 'Multiplex coverage',

         'Budget', 'Movie_length', 'Lead_ Actor_Rating', 'Lead_Actress_rating',

         'Director_rating', 'Producer_rating', 'Critic_rating', 'Trailer_views',

         'Time_taken', 'Twitter_hastags', 'Avg_age_actors', 'Num_multiplex']]= scaler.transform(Xtest[['Marketing expense', 'Production expense', 'Multiplex coverage',

                                                        'Budget', 'Movie_length', 'Lead_ Actor_Rating', 'Lead_Actress_rating',

                                                        'Director_rating', 'Producer_rating', 'Critic_rating', 'Trailer_views',

                                                        'Time_taken', 'Twitter_hastags', 'Avg_age_actors', 'Num_multiplex']])

Xtest.head()
Xtest.shape,Xtrain.shape
from catboost import CatBoostRegressor

model=CatBoostRegressor()

categorical_features_indices = np.where(Xtrain.dtypes != np.float)[0]

model.fit(Xtrain,ytrain,cat_features=([11, 14]),eval_set=(Xtest, ytest))

score_cbr=model.score(Xtest,ytest)

print("Score CatBoostRegressor:", score_cbr)
from sklearn.metrics import mean_squared_error

rmse_cbr=mean_squared_error(model.predict(Xtest),ytest)**0.5

print('RMSE CatBoostRegressor :',rmse_cbr)
from sklearn.metrics import mean_squared_log_error

RMSLE_cbr=np.sqrt(mean_squared_log_error( ytest, model.predict(Xtest) ))

print("RMSLE for CatBoostRegressor :",RMSLE_cbr)
import lightgbm as lgb

train_data=lgb.Dataset(X_train,label=y_train)

params = {'learning_rate':0.001}

model= lgb.train(params, train_data)

from sklearn.metrics import mean_squared_error

rmse_lgb=mean_squared_error(model.predict(X_test),y_test)**0.5

print('RMSE Light GBM :',rmse_lgb)
from sklearn.metrics import mean_squared_log_error

RMSLE_lgb=np.sqrt(mean_squared_log_error( y_test, model.predict(X_test) ))

print("RMSLE for Light GBM:",RMSLE_lgb)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=42)

rf.fit(X_train, y_train)
from sklearn.model_selection import GridSearchCV
params = {

    'max_depth': [25,30,35],

    'min_samples_leaf': [2,3,5],

    'max_samples': [50,75,100]

    

}
grid_search = GridSearchCV(estimator=rf,

                           param_grid=params,

                           cv=5,

                           n_jobs=-1, verbose=1)
%%time

grid_search.fit(X_train, y_train)
grid_search.best_score_
rf_best = grid_search.best_estimator_

rf_best
from sklearn.metrics import mean_squared_error

rmse_rfr_ht=mean_squared_error(rf_best.predict(X_test),y_test)**0.5

print('RMSE RandomForestRegressor With Hypertuning :',rmse_rfr_ht)
from sklearn.metrics import mean_squared_log_error

RMSLE_rfr_ht=np.sqrt(mean_squared_log_error( rf_best.predict(X_test),y_test ))

print("RMSLE for RandomForestRegressor With Hypertuning :",RMSLE_rfr_ht)
score_rfr_ht=rf_best.score(X_test,y_test)

print("Score RandomForestRegressor With Hypertuning :", score_rfr_ht)
print('RMSE Light GBM                                 :',rmse_lgb)

print('RMSE XGBRegressor                              :',rmse_xgb)

print('RMSE GradientBoostingRegressor                 :',rmse_gbc)

print('RMSE AdaBoostRegressor                         :',rmse_abr)

print('RMSE  BaggingRegressor                         :',rmse_br)

print('RMSE CatBoostRegressor                         :',rmse_cbr)

print("RMSE RandomForestRegressor Without Hypertuning :",rmse_rfr)

print('RMSE RandomForestRegressor With Hypertuning    :',rmse_rfr_ht)

print("RMSE for Linear Regression                     :",rmse_lr)
print("Score RandomForestRegressor Without Hypertuning :", score_rfr)

print("Score RandomForestRegressor With Hypertuning    :",score_rfr_ht)

print('Score XGBRegressor                              :',score_xgb)

print('Score GradientBoostingRegressor                 :',score_gbc)

print('Score AdaBoostRegressor                         :',score_abr)

print('Score  BaggingRegressor                         :',score_br)

print('Score CatBoostRegressor                         :',score_cbr)
print('RMSLE for Light GBM                                 :',RMSLE_lgb)

print('RMSLE for XGBRegressor                              :',RMSLE_xgb)

print('RMSLE for GradientBoostingRegressor                 :',RMSLE_gbc)

print('RMSLE for AdaBoostRegressor                         :',RMSLE_abr)

print('RMSLE for  BaggingRegressor                         :',RMSLE_br)

print('RMSLE for CatBoostRegressor                         :',RMSLE_cbr)

print("RMSLE for RandomForestRegressor Without Hypertuning :",RMSLE_rfr)

print("RMSLE for RandomForestRegressor With Hypertuning    :",RMSLE_rfr_ht)

print("RMSLE for Linear Regression                         :",RMSLE_lr)