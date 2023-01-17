# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb

from sklearn.metrics import precision_score, recall_score, fbeta_score, confusion_matrix, precision_recall_curve, accuracy_score,mean_absolute_error,roc_auc_score



from sklearn import preprocessing

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.manifold import TSNE

import warnings

warnings.filterwarnings('ignore')



from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score, cross_val_predict , GridSearchCV

from sklearn.pipeline import make_pipeline

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

from imblearn.under_sampling import NearMiss



# Algorithms

from sklearn import ensemble, tree, svm, naive_bayes, neighbors, linear_model, gaussian_process, neural_network

import xgboost as xgb

from xgboost.sklearn import XGBClassifier



# Evaluation

from sklearn.metrics import f1_score, accuracy_score, roc_curve, roc_auc_score

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix, recall_score, precision_score, precision_recall_curve

train = pd.read_csv('../input/data.csv')

sample = pd.read_csv('../input/sample_submission.csv')
def define_data(data , info=True ,shape = True, percentage =True,describe = True , sample=True , columns = False):

    if columns == True:

        print('\nColumns of Data...')

        print(data.columns)

        return 

    if shape ==True:

        print('Shape of Data is...')

        print(data.shape)

    if info==True:

        print('\nInfo of Data...')

        print(data.info())

    if percentage ==True:

        print('\nPercentage of Data Missing ...')

        print((data.isnull().sum()/data.shape[0])*100)

    if describe == True:

        print('\nDescription of data...')

        display(data.describe())

    if sample == True:

        print('\nSample of Data...')

        display(data.sample(10).T)

# define_data(df , info=False ,shape = True, percentage =False,describe = False , sample=True , columns = False)

# define_data(sample , info=False ,shape = True, percentage =False,describe = False , sample=True , columns = False)
train.sample(5)
len(train.match_id.unique())
train = train.drop(columns =['match_event_id' , 'date_of_game','team_name' , 

                       'match_id' , 'team_id' , 'shot_id_number' ] )
def display_unique_data(data):

    for i in data.columns:

        unique_cols_data = data[i].unique()

        if len(unique_cols_data)<20:

            print('Correct Type on Column -> ',i)

            print('Unique data in this Column is -> ',unique_cols_data)

            print('\n')
def time_combine(minute  ,second):

    if np.isnan(minute) and np.isnan(second):

        return np.nan

    else:

        if np.isnan(minute):

            return second

        if np.isnan(second):

            return minute*60

    return (minute*60+second)

def team_name(data , col , string ,col_name , string_replace=''):

    data[col_name] = data[col].str.replace(string , string_replace)

    return data

def sum_both(shot1 , shot2):

    if np.isnan(shot1):

        if not np.isnan(shot2):

            return shot2

    if np.isnan(shot2):

        if not np.isnan(shot1):

            return shot1

    return np.nan

def new_col_categorical(data , columns = [] , remove_original = True):

    for i in columns:

        unique_cols = data[i].unique()

        if len(unique_cols) < 40:

            print('\nCorrect Type on Column -> ',i)

            print('Unique data in this Column is -> ',unique_cols)

        else:

            return data

    if remove_original == False:

        original_data = data[columns]

    data = pd.get_dummies(data , columns = columns)

    if remove_original == False:

        data = pd.concat([data,original_data] , axis=1)

    return data

def check_data_train_test(train_new_x , test_x):

    flag=1

    for f in train_new_x.columns:

        if f not in test_x.columns:

            print('Column from Train -> ',f,' not present in Test Columns')

            if flag==1:

                flag=0

    for f in test_x.columns:

        if f not in train_new_x.columns:

            print('Column from Test -> ',f,' not present in Train Columns')

            if flag==1:

                flag=0

    if flag==1:

        print('No Error Found ... Checking for Mismatch..')

        print('Done')

        for i,j in zip(train_new_x.columns,test_x.columns):

            if not i==j:

                print('Possible MisMatch ' , i , ' Not Same as ' , j)

                flag=0

        if flag==0:

            print('Solving The MisMacthg Problem ')

            train_data_columns = train_new_x.columns

            test_x = test_x.reindex(train_data_columns , axis=1)

    return train_new_x , test_x

def average_groupby(train , column_group):

    new_data = pd.DataFrame()

    unique_data = train[column_group].unique()

    for i in unique_data:

        if (i!=i):

            group_data = train[train[column_group].isnull()]

        else:

            group_data = train[train[column_group]==i]

        average_0 , average_1 , average_nan = group_data['is_goal'].value_counts(dropna = False).values

        average_1 , average_0 , average_nan = average_1/group_data.shape[0] , average_0/group_data.shape[0] , average_nan/group_data.shape[0]

        group_data['average_1'] = average_1

        group_data['average_0'] = average_0

        group_data['average_nan'] = average_nan

        new_data = pd.concat([new_data , group_data] , axis=0)

    return new_data

train = average_groupby(train , column_group='game_season')
train.columns
temp_data = train[train['game_season'] ==train['game_season'].unique()[3]].is_goal.value_counts(dropna=False)
train['total_time'] = train.apply(lambda x: time_combine(x.remaining_min , x.remaining_sec), axis=1)

train = team_name(train , col='type_of_shot' , string = r'(shot - )', col_name= 'type_of_shot' )

train = team_name(train , col='type_of_combined_shot' , string = r'(shot - )', col_name= 'type_of_combined_shot' )

train['type_of_shot'] = train['type_of_shot'].astype('float64')

train['type_of_combined_shot'] = train['type_of_combined_shot'].astype('float64')

train['team_of_shot_both'] = train.apply(lambda x: sum_both(x.type_of_shot , x.type_of_combined_shot), axis=1)

train['vs_match'] = train['home/away'].str.contains('vs.')

train['@_match'] = train['home/away'].str.contains('@')

train = team_name(train , col='home/away' , string = r'(MANU @ |MANU vs. )', col_name= 'Team_against' )

train['vs_match'] = train['vs_match'].astype('bool')

train['@_match'] = train['@_match'].astype('bool')
# #preparing test data

# test['total_time'] = test.apply(lambda x: time_combine(x.remaining_min , x.remaining_sec), axis=1)

# test = team_name(test , col='type_of_shot' , string = r'(shot - )', col_name= 'type_of_shot' )

# test = team_name(test , col='type_of_combined_shot' , string = r'(shot - )', col_name= 'type_of_combined_shot' )

# test['type_of_shot'] = test['type_of_shot'].astype('float64')

# test['type_of_combined_shot'] = test['type_of_combined_shot'].astype('float64')

# test['team_of_shot_both'] = test.apply(lambda x: sum_both(x.type_of_shot , x.type_of_combined_shot), axis=1)

# test['vs_match'] = test['home/away'].str.contains('vs.')

# test['@_match'] = test['home/away'].str.contains('@')

# test['vs_match'] = test['vs_match'].astype('bool')

# test['@_match'] = test['@_match'].astype('bool')



# test = team_name(test , col='home/away' , string = r'(MANU @ |MANU vs. )', col_name= 'Team_against' )

# test = new_col_categorical(test,columns=[ 'area_of_shot' , 'game_season', 'lat/lng', 'shot_basics' , 'range_of_shot'], remove_original = False)
encode_col = [ 'area_of_shot' , 'shot_basics', 'range_of_shot'  ,'game_season','lat/lng','Team_against']

for f in encode_col:

    lbl = preprocessing.LabelEncoder()

    lbl.fit(list(train[f].values))

    train['encoded_'+f]  = lbl.transform(list(train[f].values))

#     test['encoded_'+f] = lbl.transform(list(test[f].values))

# train = new_col_categorical(train,columns=[ 'area_of_shot','lat/lng' , 'game_season','shot_basics' , 'range_of_shot'], remove_original = False)

columns = ['location_x', 'location_y', 

       'distance_of_shot', 

       'total_time', 'team_of_shot_both', 'remaining_min.1', 'power_of_shot.1',

       'knockout_match.1', 'remaining_sec.1', 'distance_of_shot.1']

for f in columns:

    scaler = preprocessing.StandardScaler()

    train[f] = scaler.fit_transform(train[[f]])
train.columns
# FillNa to fill it with 

# fillna_columns_mode = 
print('Shape of train is .. ' ,train.shape)

# print('Shape of test is .. ' ,test.shape)
def unique_count(data , columns = []):

    for col in columns :

        print('Unique Data Percentage in ',col)

        print((data[col].value_counts()/data.shape[0])*100)

        print('\n')

# unique_count(train , columns=['area_of_shot' , 'shot_basics', 'range_of_shot' , 'team_name' ])
df = train.copy()

train = df.dropna(subset=['is_goal'])

test = df[df.is_goal.isnull()]
df.sample(5)
%%time

test_x = test.drop(columns = ['Unnamed: 0','remaining_min' ,'remaining_sec' , 'game_season',

                                  'home/away','Team_against','area_of_shot' , 'shot_basics' ,

                                  'range_of_shot','is_goal','lat/lng','type_of_shot' , 'type_of_combined_shot'])

train_new_x = train.drop(columns = ['Unnamed: 0','remaining_min' ,'remaining_sec' , 'game_season' ,

                                  'home/away','Team_against','area_of_shot' , 'shot_basics' ,

                                  'range_of_shot','is_goal','lat/lng','type_of_shot' , 'type_of_combined_shot'])

train_new_y = train['is_goal']
missing_Data_percentage = pd.DataFrame(train_new_x.isnull().sum()/train.shape[0]*100)

missing_Data_percentage.sort_values(by=[0] ,  ascending = False , inplace = True)

display(missing_Data_percentage.head())
mean_fill=['location_x' , 'location_y', 'total_time','distance_of_shot' , 'remaining_min.1' , 'remaining_sec.1' ,'distance_of_shot.1'] 

mode_fill=['power_of_shot','knockout_match', 'power_of_shot.1' ,'knockout_match.1' ]

for f in  mean_fill:

    train_new_x[f] = train_new_x[f].fillna((train_new_x[f].mean()))

    test_x[f] = test_x[f].fillna((test_x[f].mean()))

for f in  mode_fill:

    train_new_x[f] = train_new_x[f].fillna((train_new_x[f].mode()[0]))

    test_x[f] = test_x[f].fillna((test_x[f].mode()[0]))

train_new_x , test_x = check_data_train_test(train_new_x  , test_x)

X_train , X_test , Y_train , Y_test = train_test_split(train_new_x , train_new_y , test_size = .10 , random_state = 65 )
# %%time

# tsne = TSNE(n_components = 2 , random_state = 32).fit_transform(train_new_x.values)

# f , (ax1) = plt.subplots(1,1, figsize = (16,8))

# ax1.scatter(tsne[:,0] , tsne[:,1] , c = (train_new_y==0) , cmap = 'coolwarm' , label = 'No Fraud')

# ax1.scatter(tsne[:,0] , tsne[:,1] , c = (train_new_y==1) , cmap = 'coolwarm' , label = 'Fraud')

# ax1.set_title('T-SNE' , fontsize = 14)

# ax1.grid(True)
%%time

MLA = [

    ensemble.AdaBoostClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),

#     gaussian_process.GaussianProcessClassifier(),

    linear_model.LogisticRegressionCV(),

    linear_model.RidgeClassifierCV(),

    linear_model.Perceptron(),

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    neighbors.KNeighborsClassifier(),

#     svm.SVC(probability=True),

#     svm.NuSVC(probability=True),

    svm.LinearSVC(),

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    xgb.XGBClassifier()

    ]





col = []

algorithms = pd.DataFrame(columns = col)

ind= 0

for a in MLA:

    

    a.fit(X_train , Y_train)

    pred = a.predict(X_test)

    acc = accuracy_score(Y_test , pred)

    f1 = f1_score(Y_test , pred)

    cross_score = cross_val_score(a, X_train , Y_train).mean()

    Alg = a.__class__.__name__

    print('Prediction Done for - ',Alg)

    algorithms.loc[ind , 'Algorithm'] = Alg

    algorithms.loc[ind , 'Accuracy'] =round( acc * 100 , 2)

    algorithms.loc[ind , 'F1_score'] = round( f1 * 100 , 2)

    algorithms.loc[ind , 'Cross_val_score'] = round( cross_score * 100 , 2)

    ind+=1



algorithms.sort_values(by=['Accuracy'] ,  ascending = False , inplace = True)

display(algorithms.head(14))
parameters = {'n_estimators': [500], 

              'max_depth':[3,4] ,

              'learning_rate' :[ 0.01 ] , 

              'loss' : ['deviance' , 'exponential'],

              'subsample' : [0.2 ,0.5 ,  0.9],

              'min_samples_leaf' : [1,2,3],

#               'max_features' : [None , 'auto' , 'log2']

             }
%%time

gbc = ensemble.GradientBoostingClassifier()

# n_jobes = use -1 to run on all processors

grid_search = GridSearchCV(estimator=gbc, param_grid=parameters, cv=3, n_jobs=-1 , verbose = 6)
%%time

grid_search.fit(train_new_x , train_new_y)

print("Best score: %0.3f" % grid_search.best_score_)

print("Best parameters set:")

best_parameters=grid_search.best_estimator_.get_params()

for param_name in sorted(parameters.keys()):

    print("\t%s: %r" % (param_name, best_parameters[param_name]))
%%time

n_estimators=500

xgbr = xgb.XGBClassifier(n_estimators=n_estimators,max_depth=8,learning_rate =0.001 , booster = 'gbtree')

xgbr.fit(X_train ,Y_train ,eval_set=[(X_train, Y_train), (X_test, Y_test)] , verbose = 3)

xgb_pred = xgbr.predict(X_test)

xgb_test_pred = xgbr.predict(test_x)

print(1/(1+mean_absolute_error(xgb_pred , Y_test)))
plt.figure(figsize=(18,18))

# xgb.plot_importance(xgbr )

# plt.show()

plt.bar(X_train.columns, xgbr.feature_importances_)

plt.xticks(rotation=90)

plt.show()
gbc_model = ensemble.GradientBoostingClassifier(learning_rate = 0.01 , max_features = 'auto' ,loss = 'exponential' , max_depth=3 , subsample = 0.5 , n_estimators = 500 )

gbc_model.fit(X_train ,Y_train)

gbc_model_test = gbc_model.predict(X_test)

gbc_model_pred = gbc_model.predict(test_x)

print(1/(1+mean_absolute_error(gbc_model_test , Y_test)))
train_set = lgb.Dataset(X_train , label = Y_train)

val_set = lgb.Dataset( X_test, label = Y_test)
output = pd.DataFrame({'shot_id_number': test_x.index,'is_goal': xgb_test_pred})

output.head()

output.to_csv('xgb_pred_normal.csv', index=False)