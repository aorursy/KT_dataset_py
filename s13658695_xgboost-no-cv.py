# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import xgboost as xgb

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold, StratifiedKFold

from time import time

import pickle

import datetime as dt

from imblearn.over_sampling import RandomOverSampler

from scipy.sparse import csc_matrix

import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

train_label = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

sample_df =  pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')

specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')

test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')



# Data Merging for train data

train_df = pd.merge(

                    train, train_label, how = 'right',

                    on = ['game_session','installation_id']

                    )



#train_df = train_df.merge(specs, left_on = 'event_id',  right_on = 'event_id', how = "left")



# merge test data

#test_df = test.merge(specs, left_on = 'event_id', right_on = 'event_id', how = "left")



test_df = test.copy()





keep_columns_train = [

                      'timestamp',

                      'event_count',

                      'event_code',

                      'event_data',

                      'game_time',

                      'world',

                      'title_y',

                   

                     ]
lbl = preprocessing.LabelEncoder()

sc = preprocessing.StandardScaler()





def preprocess_train(data, keep_columns_train):

    

    data= data[keep_columns_train]

    

    

    data["timestamp"] = pd.to_datetime(data["timestamp"])



    data["month"] = data["timestamp"].dt.month

    data["dayOfMonth"] = data["timestamp"].dt.day

    #data["week"] = data["timestamp"].dt.week

    

    #data["weekday"] = data["timestamp"].dt.weekday

    data["hour"] = data["timestamp"].dt.hour

    data["minute"] = data["timestamp"].dt.minute

    

    

    data = data.drop(columns = ['timestamp'])

    

    

    # standardisation

    #data[['game_time']] = sc.fit_transform(data[['game_time']])

    

    '''

    for f in data.columns:

        if data[f].dtype=='object':

            print(f)

            lbl.fit(data[f].values)

            data[f] = lbl.transform(data[f].values)

            print('pass')

            

    '''

    # Label Encoder

    lbl.fit(data['world'].values)

    data['world'] = lbl.transform(data['world'].values)

    

    

    

    lbl.fit(data['title_y'].values)

    data['title_y'] = lbl.transform(data['title_y'].values)

    



    data['data_str_length'] = data['event_data'].str.len()

    data = data.drop(columns=['event_data'])

    



    data = data.rename(columns={"title_y": "title"})

    

    #data = data.fillna(-999)



    return data
y = train_df['accuracy_group']

train_df_processed = preprocess_train(train_df, keep_columns_train)
X_train, X_valid, y_train, y_valid = train_test_split(train_df_processed, y, test_size = 0.15,random_state = 999)

clf = xgb.XGBClassifier(

                            

                        n_estimators=500,

                        min_child_weight = 1,

                        max_depth=6,

                        verbosity = 1,

                        n_jobs=8,                                              

                        scale_pos_weight=1.025,

                        tree_method='hist',

                        objective = 'multi:softmax',

                        num_class = 4,

                        predictor='cpu_predictor',

                        colsample_bytree = 0.66,

                        subsample = 1,

                        gamma = 0,

                        learning_rate=0.15,

                        num_parallel_tree = 1,    

                       )





clf.fit(X_train, y_train, eval_metric="merror", early_stopping_rounds=100,

                    eval_set=[(X_train, y_train), (X_valid, y_valid)],verbose=True)
# Feature Importance



print(clf.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(clf.feature_importances_, index=X_train.columns)

feat_importances.nlargest(len(X_train.columns)).plot(kind='bar',fontsize = 15)

plt.rcParams["figure.figsize"] = (20,20)

plt.xlabel("Attributes", fontsize=20)

plt.ylabel("Relative Importance", fontsize=20)

plt.show()
keep_columns_test = [

    

                     'timestamp',

                     'event_count',

                     'event_code',

                     'event_data',

                     'game_time',

                     'world',

                     'title'

                   

                    ]



def preprocess_test(data, keep_columns_test):

    

    

    data = data[keep_columns_test]

    

    data["timestamp"] = pd.to_datetime(data["timestamp"])



    data["month"] = data["timestamp"].dt.month

    data["dayOfMonth"] = data["timestamp"].dt.day

    #data["week"] = data["timestamp"].dt.week

    

    #data["weekday"] = data["timestamp"].dt.weekday

    data["hour"] = data["timestamp"].dt.hour

    data["minute"] = data["timestamp"].dt.minute

    

    

    data = data.drop(columns = ['timestamp'])

    

    # standardisation

    data[['game_time']] = sc.fit_transform(data[['game_time']])

    

    

    # Label Encoder

    lbl.fit(data['world'].values)

    data['world'] = lbl.transform(data['world'].values)

    

    lbl.fit(data['title'].values)

    data['title'] = lbl.transform(data['title'].values)  

    

    

    '''

    

    for f in data.columns:

        if data[f].dtype=='object':

            print(f)

            lbl.fit(data[f].values)

            data[f] = lbl.transform(data[f].values)

            

    '''

    data['data_str_length'] = data['event_data'].str.len()

    data = data.drop(columns=['event_data'])

    

    #data = data = data.fillna(-999)



    return data
test_df_processed = preprocess_test(test_df, keep_columns_test)



preds = clf.predict(test_df_processed)



submission = test[['installation_id']]

submission['accuracy_group'] = preds



submission = submission.groupby(['installation_id']).agg(lambda x: x.iloc[-1]).reset_index()

submission = submission.astype({"accuracy_group": int})



submission.to_csv('submission.csv', index=False)