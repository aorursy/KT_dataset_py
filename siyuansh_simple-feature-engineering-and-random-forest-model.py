import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

pd.set_option('display.max_colwidth', -1)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_y_test_sample = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')

df_specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')

df_x_train = pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv")

df_y_train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

df_x_train = df_x_train.loc[df_x_train['installation_id'].isin(df_y_train['installation_id'].unique())]

df_x_test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')

df_x_train = pd.concat([df_x_train,df_x_test])
print(df_x_train.loc[df_x_train['type']=='Assessment'].groupby(

        by=['world','title'], as_index=False

        ).count())
df_train_duration = df_x_train.groupby(

    by=['type','installation_id','game_session','world','title'], as_index=False

        ).aggregate({'timestamp': ['min', 'max']}

)

df_train_duration['duration'] = pd.to_datetime(df_train_duration["timestamp"]["max"]) -pd.to_datetime(df_train_duration["timestamp"]["min"])

df_train_duration.columns=['type','installation_id','game_session','world','title','timestamp_min','timestamp_max','duration']

df_train_duration['timestamp_min']=pd.to_datetime(df_train_duration['timestamp_min'])

df_train_duration['timestamp_max']=pd.to_datetime(df_train_duration['timestamp_max'])

df_train_duration.sort_values(by=["installation_id","world","timestamp_min"],inplace=True)

df_train_duration = df_train_duration.loc[df_train_duration['world'].isin(['MAGMAPEAK', 'TREETOPCITY', 'CRYSTALCAVES'])]

df_train_duration.reset_index(inplace=True)

df_train_duration.head()
df_train_duration['clip_count']=0

df_train_duration['cum_duration'] = df_train_duration.loc[0].duration

df_train_duration['pre_assess'] = 0

for i,row in df_train_duration.iterrows():

    if(i==0 or (row['installation_id'] != df_train_duration.loc[i-1].installation_id) or (row['world'] != df_train_duration.loc[i-1].world)):

        if(row.type == 'Clip'):

            df_train_duration.at[i,'clip_count'] = 1

    else: 

        if(row.type == 'Clip'):

            df_train_duration.at[i,'clip_count'] = df_train_duration.loc[i-1]['clip_count'] +1

        else:

            df_train_duration.at[i,'clip_count'] = df_train_duration.loc[i-1]['clip_count'] 

        df_train_duration.at[i,'cum_duration'] = row.duration + df_train_duration.loc[i-1]['cum_duration']

        if(row.type=='Assessment'):

            df_train_duration.at[i,'pre_assess'] = df_train_duration.loc[i-1]['pre_assess'] + 1

        else:

            df_train_duration.at[i,'pre_assess'] = df_train_duration.loc[i-1]['pre_assess']

df_train_duration= df_train_duration.loc[df_train_duration['type']=='Assessment']

df_train_duration.head()
df_train_event = df_x_train.loc[df_x_train['type']=='Assessment']

df_train_event = df_train_event[df_train_event['event_code'].isin(['4100','4110'])].reset_index()

df_train_event['correct'] = pd.io.json.json_normalize(df_train_event.event_data.apply(json.loads))["correct"]



df_train_event=df_train_event.groupby(

    by=['installation_id','game_session','world','title','event_code','correct'], as_index=False

).count()



df_train_event=df_train_event.pivot_table(index=['installation_id','game_session','world','title'], columns='correct', values='event_id'

                             ).reset_index()

df_train_event=df_train_event.fillna(0)

df_train_event.columns=['installation_id','game_session','world','title','nbr_false','nbr_true']

df_train_event['total'] = df_train_event['nbr_true'] + df_train_event['nbr_false']

df_train_event['accuracy'] = df_train_event['nbr_true']/df_train_event['total']

bins = [-0.01,0, 0.49, 0.5, 1]

group_names = [0,1,2,3]

df_train_event['group'] = pd.cut(df_train_event['accuracy'], bins, labels=group_names)

result = pd.merge(df_train_duration, df_train_event, how='left', on=['installation_id','game_session','world','title'])

result = result[['installation_id','game_session','world','title','duration','clip_count','cum_duration','pre_assess','group']]

result['pre_assess'] = result['pre_assess']+1

result.head()
result['nbr_duration'] = [x.seconds for x in result['duration']]

result['nbr_cumduration'] = [x.seconds for x in result['cum_duration']]

train = result[result['group']>=0]

pred =  result[result['group'].isnull()]
from sklearn.metrics import r2_score

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDClassifier

import eli5

from eli5.sklearn import PermutationImportance
train_TREE= train[train['world']=='TREETOPCITY']

train_MAGM=train[train['world']=='MAGMAPEAK']

train_CRYS=train[train['world']=='CRYSTALCAVES']

pred_TREE= pred[pred['world']=='TREETOPCITY']

pred_MAGM=pred[pred['world']=='MAGMAPEAK']

pred_CRYS=pred[pred['world']=='CRYSTALCAVES']
# Create target object and call it y

def model_train(train,pred):

    y = train.group

    # Create X

    features = ['nbr_duration','clip_count','nbr_cumduration','pre_assess']

    X = train[features]

    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

    forest_model = RandomForestClassifier(n_estimators=10, random_state=1)

    forest_model.fit(train_X, train_y)

    val_predictions = forest_model.predict(val_X)



    val_mae = mean_absolute_error(val_predictions, val_y)

    val_accuracy = accuracy_score(val_predictions, val_y)

    pred_y = forest_model.predict(pred[features])

    print(val_mae, val_accuracy)

    return pred_y
pred_TREE_y= model_train(train_TREE,pred_TREE)

pred_MAGM_y=model_train(train_MAGM,pred_MAGM)

pred_CRYS_y=model_train(train_CRYS,pred_CRYS)
pred_TREE['accuracy_group'] = pred_TREE_y.astype(int)

pred_MAGM['accuracy_group'] = pred_MAGM_y.astype(int)

pred_CRYS['accuracy_group'] = pred_CRYS_y.astype(int)
pred_final = pd.concat([pred_TREE,pred_MAGM,pred_CRYS])
submit = pd.merge(df_y_test_sample[['installation_id']],pred_final[['installation_id','accuracy_group']],how='left',on=['installation_id'])

submit.drop_duplicates(subset=None, keep='first', inplace=True)

submit.drop_duplicates(subset=['installation_id'], keep='first', inplace=True)

submit.info()
submit.to_csv('submission.csv', index=False)