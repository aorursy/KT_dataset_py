import os

import numpy as np

import pandas as pd

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import MinMaxScaler



train = pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')

test = pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')



train_x =train.loc[:,'V1':'V16']

test_x =test.loc[:,'V1':'V16']

train_y=train.loc[:,'Class']



test_index=test['Unnamed: 0'] #copying test index for

df_test = test.loc[:, 'V1':'V16']
#Pre processing the continuous data

tr_conti_data=pd.DataFrame()

te_conti_data=pd.DataFrame()

conti_index=['V1','V6','V10','V12','V13','V14','V15']

for i in conti_index:

    tr_conti_data[i]=train_x.loc[:,i]

    te_conti_data[i]=df_test.loc[:,i]



tr_continuous = MinMaxScaler().fit_transform(tr_conti_data[:])

te_continuous = MinMaxScaler().fit_transform(te_conti_data[:])

"""

print(len(tr_continuous[0]))

print(len(tr_continuous))

print(len(tr_continuous[0]))

print(len(tr_continuous))



"""
#Pre processing the discrete data



tr_disr_data=pd.DataFrame()

te_disr_data=pd.DataFrame()

disr_index=['V2','V3','V4','V5','V7','V8','V9','V16']

for i in disr_index:

    tr_disr_data[i]=train_x.loc[:,i]

    te_disr_data[i]=df_test.loc[:,i]



tr_disr_data_missing=train_x.loc[:,'V11']

te_disr_data_missing=df_test.loc[:,'V11']



tr_disr_data.join(tr_disr_data_missing)

te_disr_data.join(te_disr_data_missing)



feature_diff = set(te_disr_data_missing) - set(tr_disr_data_missing)

feature_diff_df = pd.DataFrame(data=np.zeros((train.shape[0], len(feature_diff))),

                                     columns=list(feature_diff))
tr_disr_data_onehot = tr_disr_data.copy()

te_disr_data_onehot = te_disr_data.copy()

for i in disr_index:

    tr_disr_data_onehot = pd.get_dummies(tr_disr_data_onehot, columns=[i], prefix = [i])

    te_disr_data_onehot = pd.get_dummies(te_disr_data_onehot, columns=[i], prefix = [i])



tr_disr_data_onehot.join(feature_diff_df)

tr_disr_data_arr = np.array(tr_disr_data_onehot)

te_disr_data_arr = np.array(te_disr_data_onehot)

"""

#print(te_disr_data_onehot)

print(len(tr_disr_data_arr[0]))

print(len(tr_disr_data_arr))

print(len(te_disr_data_arr[0]))

print(len(te_disr_data_arr))"""
train_data = np.concatenate((tr_disr_data_arr,tr_continuous),axis=1)

test_data = np.concatenate((te_disr_data_arr,te_continuous),axis=1)




rf =SGDClassifier(alpha=0.0001, average=False, class_weight=None,

       early_stopping=False, epsilon=0.1, eta0=0.1, fit_intercept=True,

       l1_ratio=0.15, learning_rate='adaptive', loss='log', max_iter=1000,

       n_iter_no_change=5, n_jobs=None, penalty='l2', power_t=0.5,

       random_state=0, shuffle=True, tol=0.001, validation_fraction=0.1,

       verbose=2, warm_start=False)

rf.fit(train_data, train_y)

pred = rf.predict_proba(test_data)


result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred[:,1])

result.to_csv('output.csv', index=False)

print(result)