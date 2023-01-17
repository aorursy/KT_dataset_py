import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



movie=pd.read_csv(r'/kaggle/input/cinema-movie/Movie_regression.csv')

movie.head()
movie.info()
movie.shape
movie.describe()
round((movie.isnull().sum() * 100 / len(movie)),2)


sns.distplot(movie['Time_taken'])

plt.show()
movie['Time_taken'].describe()
movie['Time_taken'].fillna(movie['Time_taken'].mean(), inplace = True) 
movie.info()
movie['Collection']=movie['Collection'].astype('float')

movie['Num_multiplex']=movie['Num_multiplex'].astype('float')

movie['Avg_age_actors']=movie['Avg_age_actors'].astype('float')

movie['Trailer_views']=movie['Trailer_views'].astype('float')

movie.info()
# import pandas_profiling as pp 

# profile = pp.ProfileReport(movie) 

# profile.to_file("MovieEDA.html")
movie.hist(figsize=(32,20),bins=50)

plt.xticks(rotation=90)

plt.show()
from sklearn.model_selection import train_test_split



# We specify this so that the train and test data set always have the same rows, respectively

np.random.seed(0)

df_train, df_test = train_test_split(movie, train_size = 0.7, random_state = 42)
df_train.shape, df_test.shape
df_train1=df_train.copy()

df_test1=df_test.copy()

df_train2=df_train.copy()

df_test2=df_test.copy()
dftrain, dfeval = train_test_split(df_train, train_size = 0.8, random_state = 42)
X_train=dftrain.drop('Collection',axis=1)

y_train=dftrain['Collection']

X_test=df_test.drop('Collection',axis=1)

y_test=df_test['Collection']

X_eval=dfeval.drop('Collection',axis=1)

y_eval=dfeval['Collection']
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
X_eval[['Marketing expense', 'Production expense', 'Multiplex coverage',

         'Budget', 'Movie_length', 'Lead_ Actor_Rating', 'Lead_Actress_rating',

         'Director_rating', 'Producer_rating', 'Critic_rating', 'Trailer_views',

         'Time_taken', 'Twitter_hastags', 'Avg_age_actors', 'Num_multiplex']]= scaler.fit_transform(X_eval[['Marketing expense', 'Production expense', 'Multiplex coverage',

                                                        'Budget', 'Movie_length', 'Lead_ Actor_Rating', 'Lead_Actress_rating',

                                                        'Director_rating', 'Producer_rating', 'Critic_rating', 'Trailer_views',

                                                        'Time_taken', 'Twitter_hastags', 'Avg_age_actors', 'Num_multiplex']])

X_eval.head()
X_test[['Marketing expense', 'Production expense', 'Multiplex coverage',

         'Budget', 'Movie_length', 'Lead_ Actor_Rating', 'Lead_Actress_rating',

         'Director_rating', 'Producer_rating', 'Critic_rating', 'Trailer_views',

         'Time_taken', 'Twitter_hastags', 'Avg_age_actors', 'Num_multiplex']]= scaler.transform(X_test[['Marketing expense', 'Production expense', 'Multiplex coverage',

                                                        'Budget', 'Movie_length', 'Lead_ Actor_Rating', 'Lead_Actress_rating',

                                                        'Director_rating', 'Producer_rating', 'Critic_rating', 'Trailer_views',

                                                        'Time_taken', 'Twitter_hastags', 'Avg_age_actors', 'Num_multiplex']])

X_test.head()
from catboost import CatBoostRegressor

model=CatBoostRegressor()

categorical_features_indices = np.where(X_train.dtypes != np.float)[0]



model.fit(X_train,y_train,cat_features=([11, 14]),eval_set=(X_eval, y_eval))



score_cbr=model.score(X_test,y_test)

print("Score CatBoostRegressor with keeping Evaluation Data available & using Preprocessing as StandardScaler:", score_cbr)
from sklearn.metrics import mean_squared_error

rmse_cbr=mean_squared_error(y_test,model.predict(X_test))**0.5

print('RMSE CatBoostRegressor with keeping Evaluation Data available & using Preprocessing as StandardScaler:',rmse_cbr)
from sklearn.metrics import mean_squared_log_error

RMSLE_cbr=np.sqrt(mean_squared_log_error( y_test, model.predict(X_test) ))

print("RMSLE for CatBoostRegressor with keeping Evaluation Data available & using Preprocessing as StandardScaler:",RMSLE_cbr)
dftrain, dfeval = train_test_split(df_train1, train_size = 0.8, random_state = 42)
X_train=dftrain.drop('Collection',axis=1)

y_train=dftrain['Collection']

X_test=df_test1.drop('Collection',axis=1)

y_test=df_test1['Collection']

X_eval=dfeval.drop('Collection',axis=1)

y_eval=dfeval['Collection']
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train[['Marketing expense', 'Production expense', 'Multiplex coverage',

         'Budget', 'Movie_length', 'Lead_ Actor_Rating', 'Lead_Actress_rating',

         'Director_rating', 'Producer_rating', 'Critic_rating', 'Trailer_views',

         'Time_taken', 'Twitter_hastags', 'Avg_age_actors', 'Num_multiplex']]= scaler.fit_transform(X_train[['Marketing expense', 'Production expense', 'Multiplex coverage',

                                                        'Budget', 'Movie_length', 'Lead_ Actor_Rating', 'Lead_Actress_rating',

                                                        'Director_rating', 'Producer_rating', 'Critic_rating', 'Trailer_views',

                                                        'Time_taken', 'Twitter_hastags', 'Avg_age_actors', 'Num_multiplex']])

X_train.head()
X_eval[['Marketing expense', 'Production expense', 'Multiplex coverage',

         'Budget', 'Movie_length', 'Lead_ Actor_Rating', 'Lead_Actress_rating',

         'Director_rating', 'Producer_rating', 'Critic_rating', 'Trailer_views',

         'Time_taken', 'Twitter_hastags', 'Avg_age_actors', 'Num_multiplex']]= scaler.fit_transform(X_eval[['Marketing expense', 'Production expense', 'Multiplex coverage',

                                                        'Budget', 'Movie_length', 'Lead_ Actor_Rating', 'Lead_Actress_rating',

                                                        'Director_rating', 'Producer_rating', 'Critic_rating', 'Trailer_views',

                                                        'Time_taken', 'Twitter_hastags', 'Avg_age_actors', 'Num_multiplex']])

X_eval.head()
X_test[['Marketing expense', 'Production expense', 'Multiplex coverage',

         'Budget', 'Movie_length', 'Lead_ Actor_Rating', 'Lead_Actress_rating',

         'Director_rating', 'Producer_rating', 'Critic_rating', 'Trailer_views',

         'Time_taken', 'Twitter_hastags', 'Avg_age_actors', 'Num_multiplex']]= scaler.transform(X_test[['Marketing expense', 'Production expense', 'Multiplex coverage',

                                                        'Budget', 'Movie_length', 'Lead_ Actor_Rating', 'Lead_Actress_rating',

                                                        'Director_rating', 'Producer_rating', 'Critic_rating', 'Trailer_views',

                                                        'Time_taken', 'Twitter_hastags', 'Avg_age_actors', 'Num_multiplex']])

X_test.head()
from catboost import CatBoostRegressor

model=CatBoostRegressor()

categorical_features_indices = np.where(X_train.dtypes != np.float)[0]



model.fit(X_train,y_train,cat_features=([11, 14]),eval_set=(X_eval, y_eval))

score_cbr1=model.score(X_test,y_test)

print("Score CatBoostRegressor with keeping Evaluation Data available & using Preprocessing as MinMaxScaler:", score_cbr1)


from sklearn.metrics import mean_squared_error

rmse_cbr1=mean_squared_error(y_test,model.predict(X_test))**0.5

print('RMSE CatBoostRegressor with keeping Evaluation Data available & using Preprocessing as MinMaxScaler:',rmse_cbr1)
from sklearn.metrics import mean_squared_log_error

RMSLE_cbr1=np.sqrt(mean_squared_log_error( y_test, model.predict(X_test) ))

print("RMSLE for CatBoostRegressor with keeping Evaluation Data available & using Preprocessing as MinMaxScaler:",RMSLE_cbr1)
dftrain, dfeval = train_test_split(df_train2, train_size = 0.8, random_state = 42)
X_train=dftrain.drop('Collection',axis=1)

y_train=dftrain['Collection']

X_test=df_test2.drop('Collection',axis=1)

y_test=df_test2['Collection']

X_eval=dfeval.drop('Collection',axis=1)

y_eval=dfeval['Collection']
from catboost import CatBoostRegressor

model=CatBoostRegressor()

categorical_features_indices = np.where(X_train.dtypes != np.float)[0]

model.fit(X_train,y_train,cat_features=([11, 14]),eval_set=(X_eval, y_eval))

score_cbr2=model.score(X_test,y_test)

print("Score CatBoostRegressor with keeping Evaluation Data available & using No Preprocessing :", score_cbr2)
from sklearn.metrics import mean_squared_error

rmse_cbr2=mean_squared_error(y_test,model.predict(X_test))**0.5

print('RMSE CatBoostRegressor with keeping Evaluation Data available & using no Preprocessing :',rmse_cbr2)
from sklearn.metrics import mean_squared_log_error

RMSLE_cbr2=np.sqrt(mean_squared_log_error( y_test, model.predict(X_test) ))

print("RMSLE for CatBoostRegressor with keeping Evaluation Data available & using No Preprocessing :",RMSLE_cbr2)
print("RMSLE for CatBoostRegressor with keeping Evaluation Data available & using Preprocessing as StandardScaler:",RMSLE_cbr)

print("RMSLE for CatBoostRegressor with keeping Evaluation Data available & using Preprocessing as MinMaxScaler  :",RMSLE_cbr1)

print("RMSLE for CatBoostRegressor with keeping Evaluation Data available & using No Preprocessing               :",RMSLE_cbr2)
print('RMSE CatBoostRegressor with keeping Evaluation Data available & using Preprocessing as StandardScaler:',rmse_cbr)

print('RMSE CatBoostRegressor with keeping Evaluation Data available & using Preprocessing as MinMaxScaler  :',rmse_cbr1)

print('RMSE CatBoostRegressor with keeping Evaluation Data available & using no Preprocessing               :',rmse_cbr2)
print("Score CatBoostRegressor with keeping Evaluation Data available & using Preprocessing as StandardScaler:", score_cbr)

print("Score CatBoostRegressor with keeping Evaluation Data available & using Preprocessing as MinMaxScaler  :", score_cbr1)

print("Score CatBoostRegressor with keeping Evaluation Data available & using no Preprocessing               :", score_cbr2)
df_train3=df_train.copy()

df_train4=df_train.copy()

df_train5=df_train.copy()

df_test3=df_test.copy()

df_test4=df_test.copy()

df_test5=df_test.copy()
X_train=df_train3.drop('Collection',axis=1)

y_train=df_train3['Collection']

X_test=df_test3.drop('Collection',axis=1)

y_test=df_test3['Collection'] ##Let's believe we will get part of evaluation
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
from catboost import CatBoostRegressor

model=CatBoostRegressor()

categorical_features_indices = np.where(X_train.dtypes != np.float)[0]

model.fit(X_train,y_train,cat_features=([11, 14]))

score_cbr4=model.score(X_test,y_test)

print("Score CatBoostRegressor with keeping no Evaluation Data available & using Preprocessing as StandardScaler:", score_cbr4)
from sklearn.metrics import mean_squared_error

rmse_cbr4=mean_squared_error(y_test,model.predict(X_test))**0.5

print('RMSE CatBoostRegressor with keeping no Evaluation Data available & using Preprocessing as StandardScaler:',rmse_cbr4)
from sklearn.metrics import mean_squared_log_error

RMSLE_cbr4=np.sqrt(mean_squared_log_error( y_test, model.predict(X_test) ))

print("RMSLE for CatBoostRegressor with keeping no Evaluation Data available & using Preprocessing as StandardScaler:",RMSLE_cbr4)
X_train=df_train4.drop('Collection',axis=1)

y_train=df_train4['Collection']

X_test=df_test4.drop('Collection',axis=1)

y_test=df_test4['Collection']##Let's believe we will get part of evaluation
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

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
from catboost import CatBoostRegressor

model=CatBoostRegressor()

categorical_features_indices = np.where(X_train.dtypes != np.float)[0]

model.fit(X_train,y_train,cat_features=([11, 14]))

score_cbr5=model.score(X_test,y_test)

print("Score CatBoostRegressor with keeping no Evaluation Data available & using Preprocessing as MinMaxScaler:", score_cbr5)
from sklearn.metrics import mean_squared_error

rmse_cbr5=mean_squared_error(y_test,model.predict(X_test))**0.5

print('RMSE CatBoostRegressor with keeping no Evaluation Data available & using Preprocessing as MinMaxScaler:',rmse_cbr5)
from sklearn.metrics import mean_squared_log_error

RMSLE_cbr5=np.sqrt(mean_squared_log_error( y_test, model.predict(X_test) ))

print("RMSLE for CatBoostRegressor with keeping no Evaluation Data available & using Preprocessing as MinMaxScaler:",RMSLE_cbr5)
X_train=df_train5.drop('Collection',axis=1)

y_train=df_train5['Collection']

X_test=df_test5.drop('Collection',axis=1)

y_test=df_test5['Collection']##Let's believe we will get part of evaluation
X_test.shape,X_train.shape
from catboost import CatBoostRegressor

model=CatBoostRegressor()

categorical_features_indices = np.where(X_train.dtypes != np.float)[0]

model.fit(X_train,y_train,cat_features=([11, 14]))

score_cbr6=model.score(X_test,y_test)

print("Score CatBoostRegressor with keeping no Evaluation Data available & using No Preprocessing :", score_cbr6)
from sklearn.metrics import mean_squared_error

rmse_cbr6=mean_squared_error(y_test,model.predict(X_test))**0.5

print('RMSE CatBoostRegressor with keeping no Evaluation Data available & using no Preprocessing :',rmse_cbr6)
from sklearn.metrics import mean_squared_log_error

RMSLE_cbr6=np.sqrt(mean_squared_log_error( y_test, model.predict(X_test) ))

print("RMSLE for CatBoostRegressor with keeping no Evaluation Data available & using No Preprocessing :",RMSLE_cbr6)
print("RMSLE for CatBoostRegressor with keeping no Evaluation Data available & using Preprocessing as StandardScaler:",RMSLE_cbr4)

print("RMSLE for CatBoostRegressor with keeping no Evaluation Data available & using Preprocessing as MinMaxScaler  :",RMSLE_cbr5)

print("RMSLE for CatBoostRegressor with keeping no Evaluation Data available & using No Preprocessing               :",RMSLE_cbr6)
print('RMSE CatBoostRegressor with keeping no Evaluation Data available & using Preprocessing as StandardScaler:',rmse_cbr4)

print('RMSE CatBoostRegressor with keeping no Evaluation Data available & using Preprocessing as MinMaxScaler  :',rmse_cbr5)

print('RMSE CatBoostRegressor with keeping no Evaluation Data available & using no Preprocessing               :',rmse_cbr6)
print("Score CatBoostRegressor with keeping no Evaluation Data available & using Preprocessing as StandardScaler:", score_cbr4)

print("Score CatBoostRegressor with keeping no Evaluation Data available & using Preprocessing as MinMaxScaler  :", score_cbr5)

print("Score CatBoostRegressor with keeping no Evaluation Data available & using no Preprocessing               :", score_cbr6)
print("RMSLE for CatBoostRegressor with keeping Evaluation Data available & using Preprocessing as StandardScaler   :",RMSLE_cbr)

print("RMSLE for CatBoostRegressor with keeping no Evaluation Data available & using Preprocessing as StandardScaler:",RMSLE_cbr4)
print("RMSLE for CatBoostRegressor with keeping Evaluation Data available & using Preprocessing as MinMaxScaler     :",RMSLE_cbr1)

print("RMSLE for CatBoostRegressor with keeping no Evaluation Data available & using Preprocessing as MinMaxScaler  :",RMSLE_cbr5)
print("RMSLE for CatBoostRegressor with keeping Evaluation Data available & using No Preprocessing                  :",RMSLE_cbr2)

print("RMSLE for CatBoostRegressor with keeping no Evaluation Data available & using No Preprocessing               :",RMSLE_cbr6)
print('RMSE CatBoostRegressor with keeping Evaluation Data available & using Preprocessing as StandardScaler   :',rmse_cbr)

print('RMSE CatBoostRegressor with keeping no Evaluation Data available & using Preprocessing as StandardScaler:',rmse_cbr4)
print('RMSE CatBoostRegressor with keeping Evaluation Data available & using Preprocessing as MinMaxScaler     :',rmse_cbr1)

print('RMSE CatBoostRegressor with keeping no Evaluation Data available & using Preprocessing as MinMaxScaler  :',rmse_cbr5)
print('RMSE CatBoostRegressor with keeping Evaluation Data available & using no Preprocessing                  :',rmse_cbr2)

print('RMSE CatBoostRegressor with keeping no Evaluation Data available & using no Preprocessing               :',rmse_cbr6)
print("Score CatBoostRegressor with keeping Evaluation Data available & using Preprocessing as StandardScaler:   ", score_cbr)

print("Score CatBoostRegressor with keeping no Evaluation Data available & using Preprocessing as StandardScaler:", score_cbr4)
print("Score CatBoostRegressor with keeping Evaluation Data available & using Preprocessing as MinMaxScaler     :", score_cbr1)

print("Score CatBoostRegressor with keeping no Evaluation Data available & using Preprocessing as MinMaxScaler  :", score_cbr5)
print("Score CatBoostRegressor with keeping Evaluation Data available & using no Preprocessing                  :", score_cbr2)

print("Score CatBoostRegressor with keeping no Evaluation Data available & using no Preprocessing               :", score_cbr6)
score=(score_cbr6,score_cbr5,score_cbr4,score_cbr2,score_cbr1,score_cbr)

RMSE =(rmse_cbr6,rmse_cbr5,rmse_cbr4,rmse_cbr2,rmse_cbr1,rmse_cbr)

RMSLE=(RMSLE_cbr6,RMSLE_cbr5,RMSLE_cbr4,RMSLE_cbr2,RMSLE_cbr1,RMSLE_cbr)
acceptedsol=min(RMSLE)

acceptedsol
acceptedsol=max(score)

acceptedsol
acceptedsol=min(RMSE)

acceptedsol