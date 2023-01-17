import numpy as np

import datetime



import matplotlib.pyplot as plt 

%matplotlib inline



import pandas as pd

pd.set_option('display.max_columns', 150)



import warnings

warnings.filterwarnings('ignore')



from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import ExtraTreesClassifier
df_train = pd.read_csv('../input/competicao-dsa-machine-learning-dec-2019/dataset_treino.csv')

df_test = pd.read_csv('../input/competicao-dsa-machine-learning-dec-2019/dataset_teste.csv')



###########################################################################################



df_train = df_train.fillna(-1)

df_test = df_test.fillna(-1)



###########################################################################################



for col in ['v3', 'v22', 'v24', 'v30', 'v31', 'v47', 'v52', 'v56', 'v66', 'v71', 'v74', 'v75', 

            'v79', 'v91', 'v107', 'v110', 'v112', 'v113', 'v125']:

    df_train[col] = LabelEncoder().fit_transform(df_train[col].astype(str))

    df_test[col] = LabelEncoder().fit_transform(df_test[col].astype(str))



###########################################################################################



features = list(df_train.columns)

features.remove('ID')

features.remove('target')



###########################################################################################



clf = ExtraTreesClassifier(

            n_estimators=700,

            max_features= 50,

            criterion= 'entropy',

            min_samples_split= 5,

            max_depth= 50, 

            min_samples_leaf= 5,   

            n_jobs=3,

            random_state=777) 



clf.fit(df_train[features],df_train['target'])



###########################################################################################



y_pred = clf.predict_proba(df_test[features])



submission1 = pd.read_csv('../input/competicao-dsa-machine-learning-dec-2019/dataset_teste.csv')

submission1['PredictedProb'] = list(y_pred[:,1])

#submission1[['ID', 'PredictedProb']].to_csv("model1_final.csv", index=False)
df_train = pd.read_csv('../input/competicao-dsa-machine-learning-dec-2019/dataset_treino.csv')

df_test = pd.read_csv('../input/competicao-dsa-machine-learning-dec-2019/dataset_teste.csv')



###########################################################################################



df_train = df_train.fillna(-1)

df_test = df_test.fillna(-1)



###########################################################################################



for col in ['v3', 'v22', 'v24', 'v30', 'v31', 'v47', 'v52', 'v56', 'v66', 'v71', 'v74', 'v75', 

            'v79', 'v91', 'v107', 'v110', 'v112', 'v113', 'v125']:

    df_train[col] = LabelEncoder().fit_transform(df_train[col].astype(str))

    df_test[col] = LabelEncoder().fit_transform(df_test[col].astype(str))



###########################################################################################



features = list(df_train.columns)



# drop columns by pandas profile

drop_list=['v12','v53','v104','v32','v86','v64','v76','v65','v55','v83','v121','v114','v25','v46','v54',

          'v63','v89','v105','v43','v60','v41','v49','v67','v77','v96','v73','v95','v118','v128']

for i in drop_list:

    features.remove(i)

    

features.remove('ID')

features.remove('target')



###########################################################################################



clf = ExtraTreesClassifier(

            n_estimators=700,

            max_features= 50,

            criterion= 'entropy',

            min_samples_split= 5,

            max_depth= 50, 

            min_samples_leaf= 5,

            n_jobs=3,

            random_state=777) 



clf.fit(df_train[features],df_train['target'])



###########################################################################################



y_pred = clf.predict_proba(df_test[features])



submission2 = pd.read_csv('../input/competicao-dsa-machine-learning-dec-2019/dataset_teste.csv')

submission2['PredictedProb'] = list(y_pred[:,1])

#submission2[['ID', 'PredictedProb']].to_csv("model2_final.csv", index=False)
perc1=0.7

perc2=0.3



df_merge = submission1[['ID', 'PredictedProb']].merge(submission2[['ID', 'PredictedProb']], on='ID', how='left')

df_merge['PredictedProb'] = df_merge['PredictedProb_x']*perc1 + df_merge['PredictedProb_y']*perc2



submission = pd.read_csv('../input/competicao-dsa-machine-learning-dec-2019/dataset_teste.csv')

submission['PredictedProb'] = df_merge['PredictedProb']

submission[['ID', 'PredictedProb']].to_csv("submission.csv", index=False)