import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/armanik-patient-drugswitch/Drug_Switch_Prediction_ParticipantsData/train_data.csv')

test = pd.read_csv('../input/armanik-patient-drugswitch/Drug_Switch_Prediction_ParticipantsData/test_data.csv')

train_labels = pd.read_csv('../input/armanik-patient-drugswitch/Drug_Switch_Prediction_ParticipantsData/train_labels.csv')
all_fitness = pd.read_csv("/kaggle/input/fitness/fitness_values_2.csv")
train_copy = train.copy()

test_copy = test.copy()

train_labels_copy = train_labels.copy()

fitness_copy = all_fitness.copy()
train = pd.merge(train, train_labels, on='patient_id', how='left')

train.head()
combine = train.append(test)
combine['event_time'].fillna(-1, inplace=True)

combine['patient_payment'].fillna(-1, inplace=True)

grouping_pid = combine.groupby('patient_id')

patient_ids = combine.drop_duplicates('patient_id')['patient_id'].tolist()



event_name_list = []

specialty_list = []

plan_type_list = []

event_time_list = []

patient_payment_list = []

outcome_flag_list = []



# .drop_duplicates('event_name') .drop_duplicates('event_name') .drop_duplicates('specialty').drop_duplicates('plan_type')



for patient_id in patient_ids:

    event_name_list.append(grouping_pid.get_group(patient_id)['event_name'].tolist())

    specialty_list.append(grouping_pid.get_group(patient_id)['specialty'].tolist())

    plan_type_list.append(grouping_pid.get_group(patient_id)['plan_type'].tolist())

    event_time_list.append(grouping_pid.get_group(patient_id)['event_time'].agg('std'))

    patient_payment_list.append(grouping_pid.get_group(patient_id)['patient_payment'].agg('mean'))

    outcome_flag_list.append(grouping_pid.get_group(patient_id)['outcome_flag'].agg('mean'))

    

dicts = {'patient_id':patient_ids, 'event_name':event_name_list,'speciality':specialty_list,'plan_type':plan_type_list,

        'event_time':event_time_list,'patient_payment':patient_payment_list}



data = pd.DataFrame(data=dicts)

data.shape
data['outcome_flag'] = outcome_flag_list

data.head()
data['event_name'] = data['event_name'].astype('str').str.replace("'","")

data['event_name'] = data['event_name'].astype('str').str.replace("]","")

data['event_name'] = data['event_name'].astype('str').str.replace("[","")



data['speciality'] = data['speciality'].astype('str').str.replace("'","")

data['speciality'] = data['speciality'].astype('str').str.replace("]","")

data['speciality'] = data['speciality'].astype('str').str.replace("[","")



data['plan_type'] = data['plan_type'].astype('str').str.replace("'","")

data['plan_type'] = data['plan_type'].astype('str').str.replace("]","")

data['plan_type'] = data['plan_type'].astype('str').str.replace("[","")



data.head()
data = pd.concat([data, 

          data.event_name.apply(lambda x: pd.Series(x.split(', ')).value_counts()).fillna(0)], 

          axis = 1)

data.shape
data = data.drop('event_name', axis=1)



data = pd.concat([data, 

          data.speciality.apply(lambda x: pd.Series(x.split(', ')).value_counts()).fillna(0)], 

          axis = 1)

data = data.drop('speciality', axis=1)

data = pd.concat([data, 

          data.plan_type.apply(lambda x: pd.Series(x.split(', ')).value_counts()).fillna(0)], 

          axis = 1)

data = data.drop('plan_type', axis=1)



data.shape
X = data[data['outcome_flag'].isnull()!=True].drop(['patient_id','outcome_flag'], axis=1)

y = data[data['outcome_flag'].isnull()!=True]['outcome_flag']



X_test = data[data['outcome_flag'].isnull()==True].drop(['patient_id','outcome_flag'], axis=1)



X.shape, y.shape, X_test.shape
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
from lightgbm import LGBMClassifier

model = LGBMClassifier(n_estimator=5000,

                       random_state=1994,

                       learning_rate=0.05,

                       reg_alpha=0.2,

                       colsample_bytree=0.5,

                       bagging_fraction=0.9)



model.fit(x_train,y_train,

          eval_set=[(x_train,y_train),(x_val, y_val.values)],

          eval_metric='auc',

          early_stopping_rounds=100,

          verbose=200)



pred_y = model.predict_proba(x_val)[:,1]
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score

print(roc_auc_score(y_val, pred_y))

confusion_matrix(y_val,pred_y>0.5)
err = []

y_pred_tot_lgm = []



from sklearn.model_selection import StratifiedKFold



fold = StratifiedKFold(n_splits=10,shuffle=True,random_state=1994)

i = 1

for train_index, test_index in fold.split(X, y):

    x_train, x_val = X.iloc[train_index], X.iloc[test_index]

    y_train, y_val = y[train_index], y[test_index]

    m = LGBMClassifier(boosting_type='gbdt',

                       max_depth=5,

                       learning_rate=0.08,

                       n_estimators=5000,

                       min_child_weight=0.01,

                       colsample_bytree=0.5,

                       random_state=1994)

    m.fit(x_train, y_train,

          eval_set=[(x_train,y_train),(x_val, y_val)],

          early_stopping_rounds=200,

          eval_metric='auc',

          verbose=200)

    pred_y = m.predict_proba(x_val)[:,1]

    print("err_lgm: ",roc_auc_score(y_val,pred_y))

    err.append(roc_auc_score(y_val, pred_y))

    pred_test = m.predict_proba(X_test)[:,1]

    i = i + 1

    y_pred_tot_lgm.append(pred_test)
np.mean(err,0)
from xgboost import XGBClassifier



errxgb = []

y_pred_tot_xgb = []



from sklearn.model_selection import KFold,StratifiedKFold



fold = StratifiedKFold(n_splits=10,shuffle=True,random_state=1994)

i = 1

for train_index, test_index in fold.split(X,y):

    x_train, x_val = X.iloc[train_index], X.iloc[test_index]

    y_train, y_val = y[train_index], y[test_index]

    m = XGBClassifier(boosting_type='gbdt',

                      max_depth=5,

                      learning_rate=0.07,

                      n_estimators=5000,

                      random_state=1994)

    m.fit(x_train, y_train,

          eval_set=[(x_train,y_train),(x_val, y_val)],

          early_stopping_rounds=200,

          eval_metric='auc',

          verbose=200)

    pred_y = m.predict_proba(x_val)[:,-1]

    print("err_xgb: ",roc_auc_score(y_val,pred_y))

    errxgb.append(roc_auc_score(y_val, pred_y))

    pred_test = m.predict_proba(X_test)[:,-1]

    i = i + 1

    y_pred_tot_xgb.append(pred_test)
from catboost import CatBoostClassifier,Pool, cv

errCB = []

y_pred_tot_cb = []

from sklearn.model_selection import KFold,StratifiedKFold



fold = StratifiedKFold(n_splits=15,shuffle=True,random_state=1994)

i = 1

for train_index, test_index in fold.split(X,y):

    x_train, x_val = X.iloc[train_index], X.iloc[test_index]

    y_train, y_val = y[train_index], y[test_index]

    m = CatBoostClassifier(n_estimators=5000,

                           random_state=1994,

                           eval_metric='AUC',

                           learning_rate=0.03)

    m.fit(x_train, y_train,

          eval_set=[(x_train,y_train),(x_val, y_val)],

          early_stopping_rounds=200,

          verbose=200)

    pred_y = m.predict_proba(x_val)[:,-1]

    print("err_cb: ",roc_auc_score(y_val,pred_y))

    errCB.append(roc_auc_score(y_val,pred_y))

    pred_test = m.predict_proba(X_test)[:,-1]

    i = i + 1

    y_pred_tot_cb.append(pred_test)
(np.mean(errxgb, 0) + np.mean(err, 0) + np.mean(errCB, 0))/3
submission = pd.DataFrame()

submission['patient_id'] = data[data['outcome_flag'].isnull()==True]['patient_id']

submission['outcome_flag'] = (np.mean(y_pred_tot_lgm, 0) + np.mean(y_pred_tot_cb, 0) + np.mean(y_pred_tot_xgb, 0))/3

submission.to_excel('rfr_lrg.xlsx',sheet_name='Sheet1', index=False)

submission.shape
submission['outcome_flag'] = (np.mean(y_pred_tot_lgm, 0) >= 0.25).astype(int)

submission['outcome_flag'].value_counts(normalize=True)

submission.shape
test_new = pd.merge(test, submission, on=['patient_id'], how='left')

test_new.head()
test_new.shape
def findClosest(arr, n, target):

    if (target <= arr[0]): 

        return arr[0] 

    if (target >= arr[n - 1]): 

        return arr[n - 1]

    i = 0; j = n; mid = 0

    while (i < j):  

        mid = int((i + j) / 2)

        if (arr[mid] == target):

            return arr[mid]

        if (target < arr[mid]) :

            if (mid > 0 and target > arr[mid - 1]): 

                return getClosest(arr[mid - 1], arr[mid], target)

            j = mid

        else : 

            if (mid < n - 1 and target < arr[mid + 1]): 

                return getClosest(arr[mid], arr[mid + 1], target)

            i = mid + 1

    return arr[mid]



def getClosest(val1, val2, target): 

    if (target - val1 >= val2 - target): 

        return val2 

    else: 

        return val1
time0 = [i*30 for i in range(1,37)]

time1 = [i*30 for i in range(1,19)]
test_new['event_time_range'] = test_new['event_time'].apply(lambda x: findClosest(time0, len(time0), x))

test_new.head()
train.shape, test_new.shape
df = train.append(test_new)

df = df.drop('event_time_range', axis=1)

df.shape
event_name = df.drop_duplicates('event_name')['event_name']

event_name = 'recency__event_name__' + event_name



specialty = df.drop_duplicates('specialty')['specialty']

specialty = 'recency__specialty__' + specialty



plan_type = df.drop_duplicates('plan_type')['plan_type']

plan_type = 'recency__event_name__' + plan_type
new_fitness = pd.DataFrame()

new_fitness['feature_name'] = (event_name).append(specialty).append(plan_type)

new_fitness.shape
new_fitness = new_fitness.drop_duplicates('feature_name').reset_index()

new_fitness = new_fitness.drop('index', axis=1)

new_fitness.shape
def get_recency_attributes(feature_name):

    column = feature_name.split('__')[1]

    value = feature_name.split('__')[2]



    patient_level_feature = pd.DataFrame(df[df[column]==value][['patient_id', 'outcome_flag', 'event_time']]

                                         .groupby(['patient_id', 'outcome_flag'])['event_time'].min(). reset_index())

    patient_level_feature.columns = ['patient_id', 'outcome_flag', 'feature_value']





    avg1 = patient_level_feature[(patient_level_feature['outcome_flag']==1) & (patient_level_feature['feature_value']!=9999999999)]['feature_value'].mean()

    sd1 = patient_level_feature[(patient_level_feature['outcome_flag']==1) & (patient_level_feature['feature_value']!=9999999999)]['feature_value'].std()

    avg0 = patient_level_feature[(patient_level_feature['outcome_flag']==0) & (patient_level_feature['feature_value']!=9999999999)]['feature_value'].mean()

    sd0 = patient_level_feature[(patient_level_feature['outcome_flag']==0) & (patient_level_feature['feature_value']!=9999999999)]['feature_value'].std()

    

    return avg1, avg0, sd1, sd0
new_fitness['numerics'] = new_fitness['feature_name'].apply(lambda x: get_recency_attributes(x))

new_fitness.head()
temp = new_fitness.copy()
temp.numerics = temp.numerics.astype('str').str.replace('(','')

temp.numerics = temp.numerics.str.replace(')','')

temp = pd.concat([temp, temp.numerics.str.split(',', expand=True)], axis=1)

temp = temp.drop(['numerics'], axis=1)

temp.columns = ['feature_name','avg_1','avg_0','sd_1','sd_0']

temp.head()
print(temp['avg_1'].str.contains('nan').sum())

print(temp['avg_0'].str.contains('nan').sum())

print(temp['sd_1'].str.contains('nan').sum())

print(temp['sd_0'].str.contains('nan').sum())



temp['avg_1'] = temp['avg_1'].str.replace('nan','0')

temp['avg_0'] = temp['avg_0'].str.replace('nan','0')

temp['sd_1'] = temp['sd_1'].str.replace('nan','0')

temp['sd_0'] = temp['sd_0'].str.replace('nan','0')



print(temp['avg_1'].str.contains('nan').sum())

print(temp['avg_0'].str.contains('nan').sum())

print(temp['sd_1'].str.contains('nan').sum())

print(temp['sd_0'].str.contains('nan').sum())



temp['avg_1'] = pd.to_numeric(temp['avg_1'])

temp['avg_0'] = pd.to_numeric(temp['avg_0'])

temp['sd_1'] = pd.to_numeric(temp['sd_1'])

temp['sd_0'] = pd.to_numeric(temp['sd_0'])

temp.shape
def get_frequency_attribute(event, value, times, data):

    df = pd.DataFrame()

    for time in times:

        _data = data[(data[time_var]<=int(time))].reset_index(drop=True)

        _freq = _data[[id_var, event, time_var]].groupby([id_var, event]).agg({time_var: len}).reset_index()

        _freq.columns = [id_var, 'feature_name', 'feature_value']

        _freq['feature_name'] = 'frequency__' + str(time) + '__' + event + '__' + _freq['feature_name'].astype(str)

        _freq = _freq.reset_index(drop=True)

        _df1 = pd.DataFrame(_freq['feature_name'].unique().tolist(), columns=['feature_name'])

        _df2 = pd.DataFrame(_freq[id_var].unique().tolist(), columns=[id_var])

        _df1['key'] = 1

        _df2['key'] = 1

        _freqTotal = pd.merge(_df2, _df1, on='key')

        _freqTotal.drop(['key'], axis=1, inplace=True)

        _freqTotal = pd.merge(_freqTotal, _freq, on=[id_var, 'feature_name'], how='left')

        _freqTotal.fillna(0, inplace=True)

        _df3 = data[[id_var,y_var]].drop_duplicates().reset_index(drop=True)

        _freqTotal = _freqTotal.merge(_df3, on=id_var, how='left')

        freqTotal = _freqTotal.copy()



        group_1 = freqTotal.loc[freqTotal[y_var]==1,['feature_name', 'feature_value']].groupby('feature_name')

        _avg1 = group_1.mean().reset_index()

        _avg1.columns = ['feature_name', 'avg_1']

        _sd1 = group_1.agg(np.std).reset_index()

        _sd1.columns = ['feature_name', 'sd_1']

        group_0 = freqTotal.loc[freqTotal[y_var]==0,['feature_name', 'feature_value']].groupby('feature_name')

        _avg0 = group_0.mean().reset_index()

        _avg0.columns = ['feature_name', 'avg_0']

        _sd0 = group_0.agg(np.std).reset_index()

        _sd0.columns = ['feature_name', 'sd_0']



        _fitness_value = pd.merge(_avg1, _avg0, on='feature_name', how='left')

        _fitness_value = pd.merge(_fitness_value, _sd1, on='feature_name', how='left')

        _fitness_value = pd.merge(_fitness_value, _sd0, on='feature_name', how='left')

        df = df.append(_fitness_value)

    return(df)
time_var = 'event_time'

id_var = 'patient_id'

y_var = 'outcome_flag'



event = 'event_name'

evalue = 'event_1'



temp = temp.append(get_frequency_attribute(event, evalue, time0, df))



temp.shape
event = 'specialty'

evalue = 'spec_1'



temp = temp.append(get_frequency_attribute(event, evalue, time0, df))



event = 'plan_type'

evalue = 'plan_type_1'



temp = temp.append(get_frequency_attribute(event, evalue, time0, df))



temp.shape
temp[['feature_name','avg_1','avg_0','sd_1','sd_0']].to_csv('fitness_values_train.csv',index=False, header=True)
def get_norm_attributes(event, value, times, data):

    df = pd.DataFrame()

    for time in times:

        _data_post = data[data[time_var]<=int(time)].reset_index(drop=True)

        _data_pre = data[data[time_var]>int(time)].reset_index(drop=True)

        _freq_post = _data_post[[id_var, event, time_var]].groupby([id_var, event]).agg({time_var: len}).reset_index()

        _freq_pre = _data_pre[[id_var, event, time_var]].groupby([id_var, event]).agg({time_var: len}).reset_index()

        _freq_post.columns = [id_var, 'feature_name', 'feature_value_post']

        _freq_pre.columns = [id_var, 'feature_name', 'feature_value_pre']

        _freq_post['feature_value_post'] = _freq_post['feature_value_post']/int(time)

        _freq_pre['feature_value_pre'] = _freq_pre['feature_value_pre']/((data[time_var].max()) - int(time))

        _normChange = pd.merge(_freq_post, _freq_pre, on=[id_var, 'feature_name'], how='outer')

        _normChange.fillna(0, inplace=True)

        _normChange['feature_value'] = np.where(_normChange['feature_value_post']>_normChange['feature_value_pre'], 1, 0)

        _normChange.drop(['feature_value_post', 'feature_value_pre'], axis=1, inplace=True)

        _normChange['feature_name'] = 'normChange__' + str(time) + '__' + event + '__' + _normChange['feature_name'].astype(str)



        _normChange = _normChange.reset_index(drop=True)

        _df1 = pd.DataFrame(_normChange['feature_name'].unique().tolist(), columns=['feature_name'])

        _df2 = pd.DataFrame(_normChange[id_var].unique().tolist(), columns=[id_var])

        _df1['key'] = 1

        _df2['key'] = 1

        _normTotal = pd.merge(_df2, _df1, on='key')

        _normTotal.drop(['key'], axis=1, inplace=True)

        _normTotal = pd.merge(_normTotal, _normChange, on=[id_var, 'feature_name'], how='left')

        _normTotal.fillna(0, inplace=True)

        _df3 = data[[id_var,y_var]].drop_duplicates().reset_index(drop=True)

        _normTotal = _normTotal.merge(_df3, on=id_var, how='left')

        normTotal = _normTotal.copy()



        group_1 = normTotal.loc[normTotal[y_var]==1,['feature_name', 'feature_value']].groupby('feature_name')

        _avg1 = group_1.mean().reset_index()

        _avg1.columns = ['feature_name', 'avg_1']

        _sd1 = group_1.agg(np.std).reset_index()

        _sd1.columns = ['feature_name', 'sd_1']

        group_0 = normTotal.loc[normTotal[y_var]==0,['feature_name', 'feature_value']].groupby('feature_name')

        _avg0 = group_0.mean().reset_index()

        _avg0.columns = ['feature_name', 'avg_0']

        _sd0 = group_0.agg(np.std).reset_index()

        _sd0.columns = ['feature_name', 'sd_0']



        _fitness_value = pd.merge(_avg1, _avg0, on='feature_name', how='left')

        _fitness_value = pd.merge(_fitness_value, _sd1, on='feature_name', how='left')

        _fitness_value = pd.merge(_fitness_value, _sd0, on='feature_name', how='left')

        df = df.append(_fitness_value)

    return (df)
event = 'event_name'

evalue = 'event_1'



temp = temp.append(get_norm_attributes(event, evalue, time1, df))



event = 'specialty'

evalue = 'spec_1'



temp = temp.append(get_norm_attributes(event, evalue, time1, df))



event = 'plan_type'

evalue = 'plan_type_1'



temp = temp.append(get_norm_attributes(event, evalue, time1, df))



temp.shape
def fitness_calculation(data):

    if ((data['sd_0'] == 0 ) and (data['sd_1'] == 0)) and (((data['avg_0'] == 0) and (data['avg_1'] != 0)) or ((data['avg_0'] != 0) and (data['avg_1'] == 0))):

        return 9999999999

    elif (((data['sd_0'] == 0 ) and (data['sd_1'] != 0)) or ((data['sd_0'] != 0) and (data['sd_1'] == 0))) and (data['avg_0'] == data['avg_1']):

        return 1

    elif ((data['sd_0'] != 0 ) and (data['sd_1'] != 0)) and (data['avg_0'] != 0):

        return ((data['avg_1']/data['sd_1'])/(data['avg_0']/data['sd_0']))

    elif ((data['sd_0'] != 0 ) and (data['sd_1'] != 0)) and ((data['avg_0'] == 0) and (data['avg_1'] != 0)):

        return 9999999999

    else:

        return 1
temp['fitness_value'] = temp.apply(fitness_calculation, axis=1)

temp.head()
temp[['feature_name','avg_0','avg_1','sd_0','sd_1','fitness_value']].to_csv('fitness_values.csv',index=False, header=True)