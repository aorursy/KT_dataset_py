# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



Players=pd.read_csv('/kaggle/input/fifa-20-complete-player-dataset/players_20.csv')

Players['main_position']=Players['player_positions'].str.split(pat=',', n=-1, expand=True)[0]

Players.head(5)


Players_grouped=Players.groupby('main_position')['value_eur'].mean()/1e6

Players_grouped=Players_grouped.sort_values()

Players_grouped.plot(kind='barh',figsize=(12,8))

plt.xlabel("Average value, M euro")
Players_grouped_age=Players.groupby('age')['value_eur'].mean()/1e6

Players_grouped_age.plot(grid=True,figsize=(12,8))

plt.ylabel('Average value, M euro')

plt.xlabel('Age')
Players_country=Players.groupby('nationality')['value_eur'].nlargest(25).reset_index(level=1, drop=True)

Players_country=Players_country.groupby('nationality').mean()

Players_country_top10=(Players_country.sort_values()/1e6).tail(10)

Players_country_top10.plot(kind='barh',figsize=(12,8))

plt.xlabel("Average value of TOP25 players, M euro")
Players=Players[Players.main_position!='GK']

Skill_cols=['age', 'height_cm', 'weight_kg','potential',

       'international_reputation', 'weak_foot', 'skill_moves', 'pace',

       'shooting', 'passing', 'dribbling', 'defending', 'physic',

       'attacking_crossing', 'attacking_finishing',

       'attacking_heading_accuracy', 'attacking_short_passing',

       'attacking_volleys', 'skill_dribbling', 'skill_curve',

       'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',

       'movement_acceleration', 'movement_sprint_speed', 'movement_agility',

       'movement_reactions', 'movement_balance', 'power_shot_power',

       'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',

       'mentality_aggression', 'mentality_interceptions',

       'mentality_positioning', 'mentality_vision', 'mentality_penalties',

       'mentality_composure', 'defending_marking', 'defending_standing_tackle',

       'defending_sliding_tackle']

print(len(Skill_cols))
from sklearn.base import BaseEstimator, TransformerMixin



class CustomTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, Columns):

        self.Columns=Columns

        

    def fit(self,X,y=None):

        return self

    

    def transform(self,X):

        New_X=X.copy()

        New_X=New_X[self.Columns].copy()

        return New_X
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer



pipeline=Pipeline([

    ('custom_tr', CustomTransformer(Skill_cols)),

    ('imputer',SimpleImputer(strategy='median')),

    ('std_scaler',StandardScaler())

])
X=pipeline.fit_transform(Players)

y=Players['value_eur'].copy()

y=y.values/1000000
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
from sklearn.linear_model import LinearRegression



lin_reg=LinearRegression()

lin_reg.fit(X_train,y_train)
from sklearn.metrics import mean_squared_error



predictions=lin_reg.predict(X_test)

mse=mean_squared_error(y_test, predictions)

rmse=np.sqrt(mse)

rmse
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV



param_grid=[

    {'n_estimators':[3,10,30], 'max_features':[2,4,6,8,10]},

    {'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4,6]}

]

forest_reg=RandomForestRegressor()



grid_search=GridSearchCV(forest_reg, param_grid, cv=5,

                        scoring='neg_mean_squared_error',

                        return_train_score=True)



grid_search.fit(X_train,y_train)
print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
feature_importances=grid_search.best_estimator_.feature_importances_

features=sorted(zip(feature_importances, Skill_cols),reverse=True)

features_sorted=np.array(features)

features_sorted
plt.pie(features_sorted[:,0], labels=features_sorted[:,1],radius=5,autopct='%1.1f%%')

plt.show()
final_model=grid_search.best_estimator_
def NationalTeamEstimator(nation,N=10):

    Players_National=Players[Players['nationality']==nation].copy()

    Players_National_prepared=pipeline.transform(Players_National)

    National_prediction=final_model.predict(Players_National_prepared)

    Players_National["value_predict"]=National_prediction

    Players_National=Players_National.sort_values(by='value_predict', ascending=False)

    Players_National["Model prediction"]=Players_National["value_predict"].round(2).astype(str)+" M Euro"

    Players_National["actual_value"]=(Players_National["value_eur"]/1e6).round(2).astype(str)+" M Euro"

    return (Players_National[['long_name','nationality','age','club','actual_value','Model prediction']].head(N))
NationalTeamEstimator('France',N=20)
NationalTeamEstimator('Germany',N=20)
NationalTeamEstimator('Russia',N=20)
Y_test_prediction=final_model.predict(X_test)

test_mse = mean_squared_error(y_test, Y_test_prediction)

test_rmse = np.sqrt(test_mse)

test_rmse