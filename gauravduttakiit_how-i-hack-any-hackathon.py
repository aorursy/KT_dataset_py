import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
movie=pd.read_csv(r'/kaggle/input/cinema-movie/Movie_regression.csv')

movie.head()
#hidden layer starts 
movie['Movie ID'] = movie.index

movie.head()
cols = movie.columns.tolist()

cols = cols[-1:] + cols[:-1]

movie = movie[cols] 

movie.head()
movie= movie.sample(frac=1).reset_index(drop=True)

train= movie[:int(0.7*(len(movie)))]

train.head()

train.shape
test_df=movie[int(0.7*(len(movie))):]

test_df.shape
test=test_df.drop('Collection',axis=1)

test.head()
test=test.reset_index()

test=test.drop('index',axis=1)

test.head()
check=test_df[['Movie ID','Collection']]

check.shape
#hidden layer stops 
train.head()
test.head()
train.info(),test.info()
train.shape,test.shape
train.describe()
test.describe()
round((train.isnull().sum() * 100 / len(train)),2)
round((test.isnull().sum() * 100 / len(test)),2)
sns.distplot(train['Time_taken'])

plt.show()
sns.distplot(test['Time_taken'])

plt.show()
train['Time_taken'].describe()
train['Time_taken'].fillna(train['Time_taken'].mean(), inplace = True) 

test['Time_taken'].fillna(train['Time_taken'].mean(), inplace = True) 
train.info(),test.info()
# import pandas_profiling as pp 

# profile = pp.ProfileReport(train) 

# profile.to_file("MovieEDA.html")
train.info(),test.info()
train['Num_multiplex']=train['Num_multiplex'].astype('float')

train['Avg_age_actors']=train['Avg_age_actors'].astype('float')

train['Trailer_views']=train['Trailer_views'].astype('float')

test['Num_multiplex']=test['Num_multiplex'].astype('float')

test['Avg_age_actors']=test['Avg_age_actors'].astype('float')

test['Trailer_views']=test['Trailer_views'].astype('float')
train.info(),test.info()
train.hist(figsize=(32,20),bins=50)

plt.xticks(rotation=90)

plt.show()
trainfinal=train.copy()

testfinal=test.copy()
from sklearn.model_selection import train_test_split



# We specify this so that the train and test data set always have the same rows, respectively

np.random.seed(0)

df_train, df_test = train_test_split(train, train_size = 0.7, random_state = 42)
df_train.shape, df_test.shape
X_train=df_train.drop(['Collection','Movie ID'],axis=1)

y_train=df_train['Collection']

X_test=df_test.drop(['Collection','Movie ID'],axis=1)

y_test=df_test['Collection']
X_train.info()
X_train.columns
from catboost import CatBoostRegressor

model=CatBoostRegressor()

categorical_features_indices = np.where(X_train.dtypes != np.float)[0]

model.fit(X_train,y_train,cat_features=([11, 14]))

score_cbr=model.score(X_test,y_test)

print("Score :", score_cbr)
from sklearn.metrics import mean_squared_error

rmse_cbr=mean_squared_error(model.predict(X_test),y_test)**0.5

print('RMSE CatBoostRegressor :',rmse_cbr)
from sklearn.metrics import mean_squared_log_error

RMSLE_cbr=np.sqrt(mean_squared_log_error( y_test, model.predict(X_test) ))

print("RMSLE for CatBoostRegressor:",RMSLE_cbr)
#Encode categorical data

dummy = pd.get_dummies(train[["Genre","3D_available"]]).iloc[:,:-1]

train = pd.concat([train,dummy], axis=1)

train = train.drop(["Genre","3D_available"], axis=1)

train.shape
#Encode categorical data

dummy = pd.get_dummies(test[["Genre","3D_available"]]).iloc[:,:-1]

test = pd.concat([test,dummy], axis=1)

test = test.drop(["Genre","3D_available"], axis=1)

test.shape
from sklearn.model_selection import train_test_split



# We specify this so that the train and test data set always have the same rows, respectively

np.random.seed(0)

df_train, df_test = train_test_split(train, train_size = 0.7, random_state = 42)
df_train.shape, df_test.shape
X_train=df_train.drop(['Collection','Movie ID'],axis=1)

y_train=df_train['Collection']

X_test=df_test.drop(['Collection','Movie ID'],axis=1)

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
X_eval.head()
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

model=xgb.XGBRegressor(random_state=42)

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
X_train1 = X_train1.drop(["3D_available_NO"], axis = 1)

X_train_lm = sm.add_constant(X_train1)

lr= sm.OLS(y_train, X_train_lm).fit()

lr.summary()
X_train1 = X_train1.drop(["Critic_rating"], axis = 1)

X_train_lm = sm.add_constant(X_train1)

lr= sm.OLS(y_train, X_train_lm).fit()

lr.summary()
X_train1 = X_train1.drop(["Movie_length"], axis = 1)

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

model = AdaBoostRegressor(random_state=42)

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

from sklearn import tree

model = BaggingRegressor(tree.DecisionTreeRegressor(random_state=42))

model.fit(X_train, y_train)

score_br=model.score(X_test,y_test)

print("Score BaggingRegressor:", score_br)
from sklearn.metrics import mean_squared_error

rmse_br=mean_squared_error(model.predict(X_test),y_test)**0.5

print('RMSE  BaggingRegressor :',rmse_br)
from sklearn.metrics import mean_squared_log_error

RMSLE_br=np.sqrt(mean_squared_log_error( y_test, model.predict(X_test) ))

print("RMSLE for BaggingRegressor :",RMSLE_br)
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

    'max_depth': [8,10,15,],

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
X_train=trainfinal.drop(['Collection','Movie ID'],axis=1)

y_train=trainfinal['Collection']

X_test=testfinal.drop(['Movie ID'],axis=1)
X_train.info()
X_train.columns
X_train.info()
from catboost import CatBoostRegressor

model=CatBoostRegressor()

categorical_features_indices = np.where(X_train.dtypes != np.float)[0]

categorical_features_indices

model.fit(X_train,y_train,cat_features=([11, 14]))
from sklearn.metrics import r2_score

r2_score_rf_train=round(r2_score(y_train, model.predict(X_train)),2)

print("R-squared Train:",r2_score_rf_train)
model.feature_importances_
imp_df = pd.DataFrame({

    "Varname": X_train.columns,

    "Imp": model.feature_importances_})
imp_df.sort_values(by="Imp", ascending=False)
testfinal.info()
# predict the target on the train dataset

predict_test = model.predict(X_test)

print('Target on test data\n\n',predict_test)
submission = pd.DataFrame({

        "Movie ID": testfinal["Movie ID"],

        "Collection": predict_test 

    })

submission.to_csv('submission.csv', index=False)
submission.head()
check.head()
plt.grid

sns.lineplot(data=check, x="Movie ID", y="Collection")

sns.lineplot(data=submission, x="Movie ID", y="Collection")

plt.show()
from sklearn.metrics import mean_squared_log_error

np.sqrt(mean_squared_log_error( check['Collection'], submission['Collection']))
print("Score in the event :",np.sqrt(mean_squared_log_error( check['Collection'], submission['Collection'])))
from sklearn.metrics import mean_squared_error

mean_squared_error(check['Collection'], submission['Collection'])**0.5
print("Score in the event : ",mean_squared_error(check['Collection'], submission['Collection'])**0.5)