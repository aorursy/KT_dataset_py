import pandas as pd

import numpy as np



import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

%matplotlib inline



#pd.set_option('display.max_rows', 100)

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import  mean_absolute_error

from sklearn.model_selection import train_test_split, KFold, cross_val_score

from datetime import datetime

from catboost import Pool, CatBoostClassifier, cv



from scipy.stats import pearsonr

from sklearn.feature_selection import RFE, RFECV

from math import sqrt



import xgboost as xgb

import seaborn as sns



from IPython.display import HTML

import base64



import re

org_train_data = pd.read_csv('../input/data.csv')

train_data=org_train_data.drop(['team_id','team_name'],axis=1)

# test_data=org_test_data.drop(['New_Price'],axis=1)
org_train_data.info()
org_train_data.head()
train_data.head()
## Distance of Range

distance_range_data = org_train_data[(org_train_data.distance_of_shot.notnull())&(org_train_data.range_of_shot.notnull())]

print(distance_range_data.groupby('range_of_shot').distance_of_shot.mean())

## replacing null values of distance_of_shot with respectice mean of range_of shot

train_data.distance_of_shot.fillna(-999,inplace=True)

train_data.range_of_shot.fillna(-999,inplace=True)



def full_null_distance(row):

    if row.distance_of_shot ==-999:

        if row.range_of_shot=='16-24 ft.':

            return 38.5

        if row.range_of_shot=='24+ ft.':

            return 45.2

        if row.range_of_shot=='8-16 ft.':

            return 31.9

        if row.range_of_shot=='Back Court Shot':

            return 74.8

        if row.range_of_shot=='Less Than 8 ft.':

            return 21.6

    else:

        return row.distance_of_shot

    

# def full_null_range(row):

#     if row.range_of_shot ==-999:

#         if row.distance_of_shot<=25:

#             return 'Less Than 8 ft.'

#         if 25<row.distance_of_shot<=35:

#             return '8-16 ft.'

#         if 35<row.distance_of_shot<=45:

#             return '16-24 ft.'

#         if 45<row.distance_of_shot<=65:

#             return '24+ ft.'

#         if row.distance_of_shot>65:

#             return 'Back Court Shot'

#     else:

#         return row.range_of_shot



train_data['distance_of_shot'] = train_data.apply(full_null_distance,axis=1)

# train_data['range_of_shot'] = train_data.apply(full_null_range,axis=1)
## Game Season



# def year(row):

#     if type(row.game_season)==str:

#         yr = re.match('.*([1-3][0-9]{3})', row.game_season)

#         return (yr.group(1))

#     else:

#         return -999



# train_data['game_season'] = train_data.apply(year, axis=1)

# train_data['game_season'] = train_data.game_season.astype('int64')

# train_data.date_of_game.fillna(-999,inplace=True)



train_data['date_of_game'] = pd.to_datetime(train_data.date_of_game)



# def game_season(row):

#     if row.game_season==-999 and row.date_of_game!=-999:

#         if row.date_of_game.month>=7:

#             return float(row.date_of_game.year)

#         elif row.date_of_game.month<=6:

#             return float(row.date_of_game.year-1)        

#     else:

#         return float(row.game_season)

    

# train_data['game_season']=train_data.apply(game_season,axis=1)

train_data['y_of_game'] = train_data['date_of_game'].dt.year

train_data['m_of_game'] = train_data['date_of_game'].dt.month

train_data['d_of_game'] = train_data['date_of_game'].dt.day

train_data.drop(['date_of_game'],axis=1, inplace=True)

train_data['game_season'] = train_data['game_season'].astype('object')



train_data.y_of_game.fillna(-999,inplace=True)

train_data.y_of_game = train_data.y_of_game.astype('int64')

train_data.info()
## Latitude and Longitude



train_data['lat/lng'].fillna(-999, inplace=True)

def lat_lng(row):

    if row['lat/lng'] != -999:

        train_data.at[row.name,'lat'] = row['lat/lng'].split(',')[0]

        train_data.at[row.name,'lng'] = row['lat/lng'].split(',')[1]

    else:

        train_data.at[row.name,'lat'] = -999

        train_data.at[row.name,'lng'] = -999

t = train_data.apply(lat_lng,axis=1)

train_data.lat = train_data.lat.astype('float64')

train_data.lng = train_data.lng.astype('float64')

train_data.drop(['lat/lng'],axis=1,inplace=True)
## Remaining Minute



train_data['remaining_min'].fillna(-999, inplace=True)

train_data['remaining_min.1'].fillna(-999, inplace=True)



def remaining_min(row):

#     if row.remaining_min!=-999 and row['remaining_min.1']!=-999:

#         return (row.remaining_min+row['remaining_min.1'])/2

    if row.remaining_min==-999 and row['remaining_min.1']!=-999:

        return row['remaining_min.1']

#     elif row.remaining_min!=-999 and row['remaining_min.1']==-999:

#         return row['remaining_min']

    else:

        return row['remaining_min']

train_data['new_remaining_min'] = train_data.apply(remaining_min,axis=1)

train_data.drop(['remaining_min','remaining_min.1'],axis=1,inplace=True)

    
## Remaining Second



train_data['remaining_sec'].fillna(-999, inplace=True)

train_data['remaining_sec.1'].fillna(-999, inplace=True)



def remaining_sec(row):

#     if row.remaining_sec!=-999 and row['remaining_sec.1']!=-999:

#         return (row.remaining_sec+row['remaining_sec.1'])/2

    if row.remaining_sec==-999 and row['remaining_sec.1']!=-999:

        return row['remaining_sec.1']

#     elif row.remaining_sec!=-999 and row['remaining_sec.1']==-999:

#         return row['remaining_sec']

    else:

        return row['remaining_sec']

train_data['new_remaining_sec'] = train_data.apply(remaining_sec,axis=1)

train_data.drop(['remaining_sec','remaining_sec.1'],axis=1,inplace=True)

    
## Distance of Shot



train_data['distance_of_shot'].fillna(-999, inplace=True)

train_data['distance_of_shot.1'].fillna(-999, inplace=True)



def remaining_sec(row):

    if row.distance_of_shot==-999 and row['distance_of_shot.1']!=-999:

        return row['distance_of_shot.1']

    else:

        return row['distance_of_shot']

train_data['new_distance_of_shot'] = train_data.apply(remaining_sec,axis=1)

train_data.drop(['distance_of_shot','distance_of_shot.1'],axis=1,inplace=True)

    
## Knockout Match



train_data['knockout_match'].fillna(-999, inplace=True)

train_data['knockout_match.1'].fillna(-999, inplace=True)



def remaining_sec(row):

    if row.knockout_match==-999 and row['knockout_match.1']<=2:

        return row['knockout_match.1']

    else:

        return row['knockout_match']

train_data['knockout_match'] = train_data.apply(remaining_sec,axis=1)

train_data.drop(['knockout_match.1'],axis=1,inplace=True)

    
## Power of Shot



train_data['power_of_shot'].fillna(-999, inplace=True)

train_data['power_of_shot.1'].fillna(-999, inplace=True)



def remaining_sec(row):

    if row.knockout_match==-999 and row['power_of_shot.1']<=7:

        return row['power_of_shot.1']

    else:

        return row['power_of_shot']

train_data['power_of_shot'] = train_data.apply(remaining_sec,axis=1)

train_data.drop(['power_of_shot.1'],axis=1,inplace=True)

    
# train_data['home/away'].fillna(-999,inplace=True)

# def home_away(row):

#     if row['home/away']!=-999:

#         if '@' in row['home/away']:

#             return 'home'

#         elif 'vs.' in row['home/away']:

#             return 'away'

#     else:

#         return -999

    

# train_data['home_away_sign'] = train_data.apply(home_away,axis=1)
fig, ax = plt.subplots(figsize=(15,7))

train_data.groupby('game_season').is_goal.value_counts().unstack().plot.barh(ax=ax)

plt.xlabel('Number')

plt.show()
def plot_correlation_map( df ):

    corr = df.corr()

    _ , ax = plt.subplots( figsize =( 20 , 15 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    _ = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

    )

plot_correlation_map(train_data)

train_data.corr()
train_data.is_goal = train_data.is_goal.astype('object')
target = train_data['is_goal']

train_data.drop(['is_goal'],axis=1,inplace=True)

train_data.fillna(-999, inplace=True)

train_data['is_goal'] = target

train_data['game_season'] = train_data['game_season'].astype('object')
temp_train_data = train_data[train_data.is_goal.notnull()]

temp_test_data = train_data[train_data.is_goal.isnull()]

target = temp_train_data['is_goal']
cat = pd.Categorical(train_data.range_of_shot, categories=['Less Than 8 ft.','8-16 ft.','16-24 ft.','24+ ft.','Back Court Shot'],ordered=True)

labels, unique = pd.factorize(cat, sort=True)

train_data_range = labels



dummy_train_data = pd.get_dummies(train_data[['area_of_shot','shot_basics','home/away','type_of_shot','type_of_combined_shot','game_season']])

# dummy_train_data['game_season'] = train_data.game_season



dummy_train_data['is_goal'] = train_data.is_goal

temp_data = train_data.select_dtypes(exclude=['object'])

temp_data.drop(['Unnamed: 0','match_id','d_of_game','m_of_game'],axis=1, inplace=True)

dummy_train_data = pd.concat([pd.Series(train_data_range),temp_data, dummy_train_data],axis=1)

dummy_train_data.rename(columns={0:'range_of_shot'},inplace=True)
dummy_test_data = dummy_train_data[dummy_train_data['is_goal'].isnull()]

dummy_train_data.drop(dummy_train_data[dummy_train_data.is_goal.isnull()].index,inplace=True)

dummy_train_data.drop(['is_goal'],axis=1,inplace=True)

dummy_test_data.drop(['is_goal'],axis=1,inplace=True)

#temp_data = train_data.iloc[np.where(train_data.dtypes == object)[0]]

temp_data = temp_train_data.select_dtypes(include=['object'])

temp_data1 = temp_test_data.select_dtypes(include=['object'])

# Catboost



from sklearn.metrics import accuracy_score

sc = StandardScaler()

X = dummy_train_data#[imp_features1[:75]]

X = sc.fit_transform(X)



test_data = dummy_test_data#[imp_features1[:75]]

test_data = sc.fit_transform(test_data)



#cate_features_index = np.where(X.dtypes == object)[0]

#X = temp_train_data.select_dtypes(include=object)

X_train, X_test, y_train, y_test = train_test_split(X,target,train_size=.85,random_state=1234)



model = CatBoostClassifier(eval_metric='Accuracy',use_best_model=True,random_seed=42)

#model.fit(X_train,y_train,cat_features=cate_features_index,eval_set=(X_test,y_test))



# model = CatBoostClassifier(task_type = "GPU",verbose=True, random_state=42, use_best_model=True,eval_metric='MAE')

model.fit(X_train,y_train,eval_set=(X_test,y_test))

y_pred = model.predict(X_test)

print(1-mean_absolute_error(y_test, y_pred))





final_pred = model.predict_proba(test_data)



feature_importances = pd.DataFrame(model.feature_importances_,

                                    index=dummy_train_data.columns,

                                    columns=['importance']).sort_values('importance',ascending=False)



imp_features1 = feature_importances.head(500).index.tolist()

feature_importances
## XGBClassifier



sc = StandardScaler()

X = dummy_train_data

X = sc.fit_transform(X)

X = pd.DataFrame(X)



test_data = dummy_test_data

test_data = sc.fit_transform(test_data)

test_data = pd.DataFrame(test_data)





y = target.astype('int64')

X_train, X_val, y_train, y_val = train_test_split(X,y,train_size=.85,random_state=1234)



                      

def train_and_predict(model,X_train,y_train,X_val,y_val,test_data):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    print(1-mean_absolute_error(y_val, y_pred))

    return model.predict_proba(test_data)[:,0]





kf = KFold(n_splits=5, shuffle=True, random_state=3)

pred_arr = []

model = XGBClassifier(n_jobs=-1, random_state=42, n_estimators=300, max_depth=3,min_child_weight=2, subsample=1 ,colsample_bytree=0.6, colsample_bynode=0.8,

                     colsample_bylevel=1, learning_rate=0.1)

for train_index, test_index in kf.split(X):    

    pred =train_and_predict(model,X.iloc[train_index],y.iloc[train_index],X.iloc[test_index],y.iloc[test_index],test_data)

    pred_arr.append(pred)
final_y_pred = (pred_arr[0]+pred_arr[1]+pred_arr[2]+pred_arr[3]+pred_arr[4]+final_pred[:,0])/6

final_y_pred
output = pd.DataFrame({'shot_id_number':temp_data1.index.tolist(),'is_goal':final_y_pred.reshape(-1)})

output.to_csv('submission.csv',index=False)

output['shot_id_number'] = output['shot_id_number']+1

output = output.iloc[:5000,:]



# To create csv file which can be downloaded from the kernel without being commited on kaggle



def create_download_link(df, title = "Download CSV file", filename = "submission6.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(output)
feature_importances = pd.DataFrame(model.feature_importances_,

                                    index=dummy_train_data.columns,

                                    columns=['importance']).sort_values('importance',ascending=False)



imp_features = feature_importances.head(500).index.tolist()

feature_importances
# sc = StandardScaler()

# X = dummy_train_data.drop(['d_of_game','m_of_game'],axis=1)#[imp_features[:120]]

# X = sc.fit_transform(X)



# target = target.astype('int64')

# X_train, X_val, y_train, y_val = train_test_split(X,target,train_size=.85,random_state=1234)



# scores = []

# n_estimators = [100,400,500,600,700]



# for nes in n_estimators:

#     xgb = XGBClassifier(learning_rate =0.1, n_estimators=nes,seed=42,silent=1, n_jobs=-1)

#     xgb.fit(X_train, y_train)

#     y_pred = xgb.predict(X_val)

#     error = mean_absolute_error(y_val,y_pred)

#     scores.append(1-error)

#     print("test data MAE eval : {}".format(1-error))

# print("best n_estimator {}".format(n_estimators[np.argmax(scores)]))

    

# plt.plot(n_estimators,scores,'o-')

# plt.ylabel('MAE')

# plt.xlabel("n_estimator")



# ############################################################################



# scores_md = []

# max_depths = [3,4,5,10,11,12,13]



# for md in max_depths:

#     xgb = XGBClassifier(learning_rate =0.1, n_estimators=n_estimators[np.argmax(scores)], 

#                         max_depth=md, seed=42, n_jobs=-1, silent=1)

#     xgb = RandomForestClassifier(n_estimators=600, n_jobs=-1, random_state=42, max_depth=md)



#     xgb.fit(X_train, y_train)

#     y_pred = xgb.predict(X_val)

#     error = mean_absolute_error(y_val,y_pred)

#     scores_md.append(1-error)



#     print("test data MAE eval : {}".format(1-error))

# print("best max_depth {}".format(max_depths[np.argmax(scores_md)]))

    

# # plt.plot(max_depths,scores_md,'o-')

# # plt.ylabel('MAE')

# # plt.xlabel("max_depth")





# # # #####################################################################



# scores_mcw = []

# min_child_weights = [0.001,0.1,0.5,1,2,3,]



# for mcw in min_child_weights:

#     xgb = XGBClassifier(learning_rate =0.1, n_estimators=n_estimators[np.argmax(scores)], max_depth = max_depths[np.argmax(scores_md)],

#                         min_child_weight=mcw, seed=42, n_jobs=-1, silent=1)

#     xgb.fit(X_train, y_train)

#     y_pred = xgb.predict(X_val)

#     error = mean_absolute_error(y_val,y_pred)

#     scores_mcw.append(1-error)



#     print("test data MAE eval : {}".format(1-error))

# print("best min_child_weight {}".format(min_child_weights[np.argmax(scores_mcw)]))

    

# # ##############################################################################

# scores_ss = []

# subsamples = [0.5,0.6,0.7,0.8,0.9,1]



# for ss in subsamples:

#     xgb = XGBClassifier(learning_rate =0.1, n_estimators=n_estimators[np.argmax(scores)], max_depth = max_depths[np.argmax(scores_md)],

#                        min_child_weight=min_child_weights[np.argmax(scores_mcw)],

#                        subsample= ss,seed=42, n_jobs=-1, silent=1)

#     xgb.fit(X_train, y_train)

#     y_pred = xgb.predict(X_val)

#     error = mean_absolute_error(y_val,y_pred)

#     scores_ss.append(1-error)



#     print("test data MAE eval : {}".format(1-error))

# print("best subsample {}".format(subsamples[np.argmax(scores_ss)]))

    

# # plt.plot(subsamples,scores_ss,'o-')

# # plt.ylabel('RMSLE')

# # plt.xlabel('subsamples')



# # ###############################################################################





# scores_csbt = []

# col_sample_tree = [0.5,0.6,0.7,0.8,0.9,1]



# for csbt in col_sample_tree:

#     xgb = XGBClassifier(learning_rate =0.1, n_estimators=n_estimators[np.argmax(scores)], max_depth = max_depths[np.argmax(scores_md)],min_child_weight=min_child_weights[np.argmax(scores_mcw)],

#                        subsample=subsamples[np.argmax(scores_ss)], colsample_bytree=csbt ,seed=42, n_jobs=-1, silent=1)

#     xgb.fit(X_train, y_train)

#     y_pred = xgb.predict(X_val)

#     error = mean_absolute_error(y_val,y_pred)

#     scores_csbt.append(1-error)



#     print("test data MAE eval : {}".format(1-error))

# print("best col sample tree {}".format(col_sample_tree[np.argmax(scores_csbt)]))



# # plt.plot(col_sample_tree,scores_csbt,'o-')

# # plt.ylabel('RMSLE')

# # plt.xlabel("col sample tree")



# # ################################################################################



# scores_csbl = []

# col_sample_level = [0.5,0.6,0.7,0.8,0.9,1]



# for csbl in col_sample_level:

#     xgb = XGBClassifier(learning_rate =0.1, n_estimators=n_estimators[np.argmax(scores)], max_depth = max_depths[np.argmax(scores_md)],min_child_weight=min_child_weights[np.argmax(scores_mcw)],

#                        subsample=subsamples[np.argmax(scores_ss)], colsample_bytree=col_sample_tree[np.argmax(scores_csbt)] ,

#                        colsample_bylevel=csbl, seed=42, n_jobs=-1, silent=1)

#     xgb.fit(X_train, y_train)

#     y_pred = xgb.predict(X_val)

#     error = mean_absolute_error(y_val,y_pred)

#     scores_csbl.append(1-error)



#     print("test data MAE eval : {}".format(1-error))



# print("best col sample level {}".format(col_sample_level[np.argmax(scores_csbl)]))



# # ####################################################################################    

# scores_csbn = []

# col_sample_node = [0.5,0.6,0.7,0.8,0.9,1]



# for csbn in col_sample_node:

#     xgb = XGBClassifier(learning_rate =0.1, n_estimators=n_estimators[np.argmax(scores)], max_depth = max_depths[np.argmax(scores_md)],min_child_weight=min_child_weights[np.argmax(scores_mcw)],

#                        subsample=subsamples[np.argmax(scores_ss)], colsample_bytree=col_sample_tree[np.argmax(scores_csbt)] ,

#                        colsample_bylevel=col_sample_level[np.argmax(scores_csbl)], colsample_bynode=csbn,seed=42, n_jobs=-1, silent=1)

#     xgb.fit(X_train, y_train)

#     y_pred = xgb.predict(X_val)

#     error = mean_absolute_error(y_val,y_pred)

#     scores_csbn.append(1-error)



#     print("test data MAE eval : {}".format(1-error))

# print("best col sample node {}".format(col_sample_node[np.argmax(scores_csbn)]))

    

# # #################################################################################   

# scores_eta = []

# etas = [0.01,0.05,0.07,0.08,0.09,0.1,]



# for eta in etas:

#     xgb = XGBClassifier(learning_rate =eta, n_estimators=n_estimators[np.argmax(scores)], max_depth = max_depths[np.argmax(scores_md)],min_child_weight=min_child_weights[np.argmax(scores_mcw)],

#                        subsample=subsamples[np.argmax(scores_ss)], colsample_bytree=col_sample_tree[np.argmax(scores_csbt)] ,seed=42, n_jobs=-1, silent=1)

#     xgb.fit(X_train, y_train)

#     y_pred = xgb.predict(X_val)

#     error = mean_absolute_error(y_val,y_pred)

#     scores_eta.append(1-error)



#     print("test data MAE eval : {}".format(1-error))

# print("best learning rate {}".format(etas[np.argmax(scores_eta)]))

    

# # plt.plot(etas,scores_eta,'o-')

# # plt.ylabel('RMSLE')

# # plt.xlabel("learning rate")
