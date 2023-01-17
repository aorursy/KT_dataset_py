import math

import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import warnings 

import seaborn as sns

import sklearn

from datetime import datetime

import calendar

%matplotlib inline

import shelve

import warnings

warnings.simplefilter('ignore')
games = pd.read_csv('../input/epl-stats-20192020/epl2020.csv')

print(games.columns)

print(games.isna().sum()) #no missing data
games = games.drop('Unnamed: 0', axis=1) # drop useless column

games = games.drop(['scored', 'missed', 'wins', 'draws', 'loses', 'pts',

                    'tot_points', 'tot_goal', 'tot_con'], axis=1) # drop columns which give away result
games['target']=np.where(games['result']=='l', 0,

                         np.where(games['result']=='d', 1,2))
games = games.drop('result', axis=1) # drop useless column
games[['target', 'h_a', 'teamId', 'Referee.x', 'matchDay']] = games[['target', 'h_a', 'teamId', 'Referee.x', 'matchDay']].astype('category')
EDA_pairplot1=games.filter(items=['xG', 'xGA', 'npxG', 'npxGA', 'npxGD', 'h_a', 'target'])

sns.pairplot(EDA_pairplot1, hue='target')
EDA_pairplot2=games.filter(items=['ppda_cal', 'allowed_ppda', 'HS.x', 'HST.x', 'HF.x', 'HC.x', 'target'])

sns.pairplot(EDA_pairplot2, hue='target')
EDA_pairplot3=games.filter(items=['ppda_cal', 'allowed_ppda', 'AS.x', 'AST.x', 'AF.x', 'AC.x', 'target'])

sns.pairplot(EDA_pairplot3, hue='target')
EDA_plot4=games.groupby('teamId')['target'].agg(counts='value_counts').reset_index()

EDA_plot4['target']=np.where(EDA_plot4['target']==0, 'loss',

                         np.where(EDA_plot4['target']==1, 'draw','win'))

EDA_plot4=EDA_plot4.reset_index() 

sns.catplot(y="teamId",x='counts', hue='target',data=EDA_plot4)
from sklearn.preprocessing import StandardScaler
nums= games.select_dtypes(include=['float', 'int64'])

other= games.select_dtypes(exclude=['float', 'int64']).drop(['date', 'target'], axis=1)
scaler = StandardScaler()

nums_scaled = scaler.fit_transform(nums)

games_scaled=pd.DataFrame(nums_scaled, columns=nums.columns)
dummies=pd.get_dummies(other)
gameday_pred = pd.concat([games_scaled,dummies, games['target']], axis=1)  
X=gameday_pred.drop('target', axis=1)

y=gameday_pred['target']
print(y.dtypes)

print(X.dtypes)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,  QuadraticDiscriminantAnalysis

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, log_loss, precision_score, recall_score, f1_score
classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="rbf", C=0.025, probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis()]
log_cols=["Classifier", "Accuracy", "Log Loss"]

log = pd.DataFrame(columns=log_cols)
for clf in classifiers:

    clf.fit(X_train, y_train)

    name = clf.__class__.__name__

    

    print("="*30)

    print(name)

    

    print('****Results****')

    train_predictions = clf.predict(X_test)

    acc = accuracy_score(y_test, train_predictions)

    

    # calculate score

    precision = precision_score(y_test, train_predictions, average = 'macro') 

    recall = recall_score(y_test, train_predictions, average = 'macro') 

    f_score = f1_score(y_test, train_predictions, average = 'macro')

    

    

    print("Precision: {:.4%}".format(precision))

    print("Recall: {:.4%}".format(recall))

    print("F-score: {:.4%}".format(recall))

    print("Accuracy: {:.4%}".format(acc))

    

    train_predictions = clf.predict_proba(X_test)

    ll = log_loss(y_test, train_predictions)

    print("Log Loss: {}".format(ll))

    

    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)

    log = log.append(log_entry)

    

print("="*30)
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
sns.set_color_codes("muted")

sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")
rf = RandomForestClassifier(n_estimators=500)

rf.fit(X_train, y_train);

feat_importances = pd.Series(rf.feature_importances_, index=X_train.columns)

feat_importances.nlargest(10).plot(kind='barh')
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

import xgboost as xgb
xgb = XGBClassifier(objective='multi:softprob',silent=False)
xgb.fit(X_train,y_train)

y_pred_xgb_basic=xgb.predict(X_test)
precision = precision_score(y_test, y_pred_xgb_basic, average = 'macro') * 100

recall = recall_score(y_test, y_pred_xgb_basic, average = 'macro') * 100

f_score = f1_score(y_test, y_pred_xgb_basic, average = 'macro') * 100

a_score = accuracy_score(y_test, y_pred_xgb_basic) * 100
print('Precision: %.3f' % precision)

print('Recall: %.3f' % recall)

print('F-Measure: %.3f' % f_score)

print('Accuracy: %.3f' % a_score)
parameters_xgb = {

        'learning_rate': [0.05, 0.1, 0.2, 0.3, 0.5], 

        'n_estimators': [200, 300, 400, 500, 600], 

        'max_depth': [1, 5, 10, 15, 20], 

        'gamma' :[0.1, 0.5, 1, 2, 5], 

        'subsample': [0.5, 0.75, 1], 

        'colsample_bytree': [0.01, 0.1, 0.5, 1], 

        }
grid_search_xgb = GridSearchCV(estimator = xgb, param_grid = parameters_xgb, 

                          cv = 3,n_jobs=-1, verbose = 2)
grid_search_xgb.fit(X_train,y_train)

print(grid_search_xgb.best_params_)

    # {'colsample_bytree': 0.1, 'gamma': 0.5, 'learning_rate': 0.3, 'max_depth': 15, 'n_estimators': 300, 'subsample': 1}
best_grid_xgb = grid_search_xgb.best_estimator_

best_grid_xgb.fit(X_train,y_train)
y_pred_xgb = best_grid_xgb.predict(X_test)
precision = precision_score(y_test, y_pred_xgb, average = 'macro') * 100

recall = recall_score(y_test, y_pred_xgb, average = 'macro') * 100

f_score = f1_score(y_test, y_pred_xgb, average = 'macro') * 100

a_score = accuracy_score(y_test, y_pred_xgb) * 100
print('Precision: %.3f' % precision)

print('Recall: %.3f' % recall)

print('F-Measure: %.3f' % f_score)

print('Accuracy: %.3f' % a_score)
rf = RandomForestClassifier(random_state = 1)

rf_model_basic = rf.fit(X_train, y_train)

y_pred_rf_basic = rf_model_basic.predict(X_test)
precision = precision_score(y_test, y_pred_rf_basic, average = 'macro') * 100

recall = recall_score(y_test, y_pred_rf_basic, average = 'macro') * 100

f_score = f1_score(y_test, y_pred_rf_basic, average = 'macro') * 100

a_score = accuracy_score(y_test, y_pred_rf_basic) * 100
print('Precision: %.3f' % precision)

print('Recall: %.3f' % recall)

print('F-Measure: %.3f' % f_score)

print('Accuracy: %.3f' % a_score)
parameters_rf = {

    'bootstrap': [True],

    'n_estimators' : [100, 300, 500, 800, 1200],

    'max_depth' : [5, 8, 15, 25, 30],

    'min_samples_split' : [2, 5, 10, 15, 100],

    'min_samples_leaf' : [1, 2, 5, 10],

    'max_features': [2, 4]

}
grid_search_rf = GridSearchCV(estimator = rf, param_grid = parameters_rf, 

                          cv = 3,n_jobs=-1, verbose = 2)
grid_search_rf.fit(X_train,y_train)

print(grid_search_rf.best_params_)

    # {'bootstrap': True, 'max_depth': 15, 'max_features': 4, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 800}
best_grid_rf = grid_search_rf.best_estimator_

best_grid_rf.fit(X_train,y_train)
y_pred_rf = best_grid_rf.predict(X_test)
precision = precision_score(y_test, y_pred_rf, average = 'macro') * 100

recall = recall_score(y_test, y_pred_rf, average = 'macro') * 100

f_score = f1_score(y_test, y_pred_rf, average = 'macro') * 100

a_score = accuracy_score(y_test, y_pred_rf) * 100
print('Precision: %.3f' % precision)

print('Recall: %.3f' % recall)

print('F-Measure: %.3f' % f_score)

print('Accuracy: %.3f' % a_score)
games2 = pd.read_csv('../input/epl-stats-20192020/epl2020.csv')

table=games2.groupby('teamId')['pts'].agg(table_points='sum').reset_index().sort_values('table_points', ascending=False)

table['position']=range(0,len(table), 1)

table['position']=table['position']+1

print(table)
fc_games=games

fc_games['date'] = pd.to_datetime(fc_games.date , format = '%Y-%m-%d %H:%M:%S')

fc_games.index = fc_games['date']

fc_games = fc_games.drop(['date'], axis=1)
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
categories= fc_games.select_dtypes(include=['category']).drop(['teamId','target'], axis=1).apply(LabelEncoder().fit_transform)

fc_games=fc_games.drop(categories.columns.values, axis=1)

fc_games=pd.concat([fc_games, categories], axis=1)
from statsmodels.tsa.stattools import adfuller
def adfuller_test(series, signif=0.05, name='', verbose=False):

    r = adfuller(series, autolag='AIC')

    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}

    p_value = output['pvalue'] 

    def adjust(val, length= 6): return str(val).ljust(length)



    # Print Summary

    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)

    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')

    print(f' Significance Level    = {signif}')

    print(f' Test Statistic        = {output["test_statistic"]}')

    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():

        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:

        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")

        print(f" => Series is Stationary.")

    else:

        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")

        print(f" => Series is Non-Stationary.")   

    
x=fc_games[fc_games['teamId']=='Liverpool'] #lets take an example and see if we can test/produce stationarity

x=x.drop(['teamId','target'], axis=1) #dont want to forecast either of these.

fc_train = x[:int(0.8*(len(x)))] #split data

fc_valid = x[int(0.8*(len(x))):]
for name, column in fc_train.iteritems(): #run the ADF test of stationarity

    adfuller_test(column, name=column.name)

    print('\n')
fc_train_diff = fc_train.diff().dropna()
from statsmodels.stats.stattools import durbin_watson

from statsmodels.tsa.api import VAR

 

feat_importances_1=feat_importances.nlargest(5).index.values

feat_importances_2=feat_importances.nlargest(10).index.values[5:]

feat_importances_3=feat_importances.nlargest(15).index.values[10:]
def validation_by_team(x):

    x1=x.filter(items=feat_importances_1)

    x2=x.filter(items=feat_importances_2)

    

    fc_train1 = x1[:int(0.8*(len(x1)))] #split data

    fc_valid1 = x1[int(0.8*(len(x1))):]

    

    fc_train2 = x2[:int(0.8*(len(x2)))] #split data

    fc_valid2 = x2[int(0.8*(len(x2))):]

    

    model1 = VAR(endog=fc_train1) #fit VAR model

    model_fit1 = model1.fit()

    

    model2 = VAR(endog=fc_train2) #fit VAR model

    model_fit2 = model2.fit()

    

    prediction1 = model_fit1.forecast(model_fit1.y, steps=len(fc_valid1)) #predict

    prediction1 = pd.DataFrame(data=prediction1, columns=x1.columns)

    

    prediction2 = model_fit2.forecast(model_fit2.y, steps=len(fc_valid2)) #predict

    prediction2 = pd.DataFrame(data=prediction2, columns=x2.columns)

       

    # Check the performance of the by serial correlation of errors using the Durbin Watson statistic.

    

    # The value of this statistic can vary between 0 and 4. The closer it is to the value 2, 

    # then there is no significant serial correlation. The closer to 0, there is a positive 

    # serial correlation, and the closer it is to 4 implies negative serial correlation.

    

    out1 = durbin_watson(model_fit1.resid)

    print(out1)

    out2 = durbin_watson(model_fit2.resid)

    print(out2)

    

    prediction_performance = pd.DataFrame([out1, out2]).T

    return(prediction_performance)
prediction_validations=fc_games.groupby('teamId').apply(validation_by_team)
import random
def forecast_by_team(x):

    target=x['target'].reset_index(drop=True)

    x=x.drop(['teamId','target'], axis=1) #dont want to forecast either of these.

    

    x1=x.filter(items=feat_importances_1)

    x2=x.filter(items=feat_importances_2)

    x3=x.filter(items=feat_importances_3)

    

    model1 = VAR(endog=x1) #fit VAR model1

    model_fit1 = model1.fit()

    

    model2 = VAR(endog=x2) #fit VAR model2

    model_fit2 = model2.fit()

    

    model3 = VAR(endog=x3) #fit VAR model3

    model_fit3 = model3.fit()

    

    prediction1 = model_fit1.forecast(model_fit1.y, steps=6) #predict

    prediction1 = pd.DataFrame(data=prediction1, columns=x1.columns)

    

    prediction2 = model_fit2.forecast(model_fit2.y, steps=6) #predict

    prediction2 = pd.DataFrame(data=prediction2, columns=x2.columns)

    

    prediction3 = model_fit3.forecast(model_fit3.y, steps=6) #predict

    prediction3 = pd.DataFrame(data=prediction3, columns=x3.columns)

    

    predictions=pd.concat([prediction1, prediction2, prediction3], axis=1)

    

    x_forecasted=pd.concat([x, predictions], axis=0).reset_index()

    

    # Lets randomly impute home/away games as 0 or 1

    # I start by generating random 0s and 1s and selecting the index where data is missing

    # Then we fill.

    na_loc =x_forecasted.index[x_forecasted['h_a'].isnull()]

    num_nas = len(na_loc)

    fill_values = pd.DataFrame({'h_a': [random.randint(0,1) for i in range(num_nas)]}, index = na_loc)

    

    x=pd.concat([x, fill_values], axis=0).reset_index(drop=True)

    predictions.index=fill_values.index

    x_forecasted2=x.combine_first(predictions)

    

    # Now lets mean (numeric) or mode (categorical) impute the other missing variables

    # Since these are of less modelling importance, a mean impute shouldn't make much difference/

    # Also, it is logical to assume consistency in Season performance, without any new data

    # to think otherwise.

    

    nums= x_forecasted2.select_dtypes(include=['float', 'int64']).apply(lambda x: x.fillna(x.mean()),axis=0).round()

    x_forecasted3=x_forecasted2.combine_first(nums) #join this imputation back in to fill missingness

    x_forecasted4=pd.concat([x_forecasted3, target], axis=1)

    

    return(x_forecasted4)

    

forecasted_data=fc_games.groupby('teamId').apply(forecast_by_team).reset_index().drop('level_1', axis=1) # run seperately for all teams

print(forecasted_data.isna().sum()) #no missing data but the target
forecasted_data[[

    'target', 'h_a', 'teamId', 'Referee.x', 'matchDay']] = forecasted_data[[

        'target', 'h_a', 'teamId', 'Referee.x', 'matchDay']].astype('category')

        

# Scale numeric data

from sklearn.preprocessing import StandardScaler
nums= forecasted_data.select_dtypes(include=['float', 'int64'])

other= forecasted_data.select_dtypes(exclude=['float', 'int64']).drop('target', axis=1)
scaler = StandardScaler()

nums_scaled = scaler.fit_transform(nums)

fc_scaled=pd.DataFrame(nums_scaled, columns=nums.columns)
dummies=pd.get_dummies(other)
fc_pred = pd.concat([fc_scaled,dummies, forecasted_data['target']], axis=1)  
train_data = fc_pred[fc_pred['target'].notnull()]

X_train=train_data.drop('target', axis=1)

y_train=train_data['target']
X_test=fc_pred[fc_pred['target'].isnull()].drop('target', axis=1).reset_index(drop=True)
xgb = XGBClassifier(objective='multi:softprob',

                    colsample_bytree = 0.5, gamma = 2, learning_rate= 0.2, max_depth= 10, 

                    n_estimators= 500, subsample = 1)

xgb.fit(X_train,y_train)

final_predictions=pd.DataFrame({'target':xgb.predict(X_test)})
final_test=pd.concat([X_test, final_predictions], axis=1)

complete_season=pd.concat([train_data, final_test], axis=0).reset_index(drop=True)
team_cols=complete_season[complete_season.columns[complete_season.columns.to_series().str.contains('teamId')]]#[col==1].stack().reset_index().drop(0,1)

team_cols_long=team_cols[team_cols==1].stack().reset_index().drop(0,1).drop('level_0', axis=1)

team_cols_long['team']=team_cols_long['level_1'].str.rpartition('_')[2]

team_cols_long=team_cols_long.drop('level_1', axis=1)
complete_season=pd.concat([complete_season, team_cols_long], axis=1)
new_points=complete_season.filter(items=['team', 'target'])

new_points['pts_new']=np.where(new_points['target']==0, 0,

                           np.where(new_points['target']==1,1, 3))

table_final=new_points.groupby('team')['pts_new'].agg(table_points_new='sum').reset_index().sort_values('table_points_new', ascending=False)

table_final['position']=range(0,len(table_final), 1)

table_final['position']=table_final['position']+1
print(table_final)
table_final.sort_values('team', inplace=True)

table.sort_values('teamId', inplace=True)

table_final['positition_change'] = table['position'] -  table_final['position']

table_final.sort_values('position', inplace=True, ascending=True)
print(table_final)
os.chdir(r"C:/Users/jaket")

!jupyter nbconvert --to html EPL_2020_predictions.ipynb