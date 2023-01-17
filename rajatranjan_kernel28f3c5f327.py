import time

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

train = pd.read_csv('/kaggle/input/mh-patient-drug-switch-prediction/DS_ML_Recruitment_V2.0/train_data.csv')

# test = pd.read_csv('/kaggle/input/mh-patient-drug-switch-prediction/Drug_Switch_Prediction_ParticipantsData/test_data.csv')

ftns = pd.read_csv('/kaggle/input/mh-patient-drug-switch-prediction/DS_ML_Recruitment_V2.0/fitness_values_2.csv')

labels = pd.read_csv('/kaggle/input/mh-patient-drug-switch-prediction/DS_ML_Recruitment_V2.0/train_labels.csv')

# s = pd.read_csv('/kaggle/input/mh-patient-drug-switch-prediction/Drug_Switch_Prediction_ParticipantsData/Sample Submission.csv')

# Any results you write to the current directory are saved as output.



start=time.time()



import pandas as pd

import numpy as np

start
train_data = pd.merge(train, labels, on='patient_id', how='left')

del train, labels
time_var = 'event_time'

id_var = 'patient_id'

y_var = 'outcome_flag'

from tqdm import tqdm_notebook as tqdm
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
column = ['event_name','specialty','plan_type']
lavg1=[]

lsd1=[]

lavg0=[]

lsd0=[]

fn=[]

feature_name = 'recency__event_name__event_10'



values = train_data['event_name'].unique()





for k in tqdm(column):

    _data=train_data[[k,'patient_id', 'outcome_flag', 'event_time']].groupby(['patient_id',k, 'outcome_flag'])['event_time'].min().reset_index()

    for j in tqdm(train_data[k].unique()):

        patient_level_feature=_data[_data[k]==j].drop([k],axis=1)

        patient_level_feature.columns = ['patient_id', 'outcome_flag', 'feature_value']

        x1=patient_level_feature[(patient_level_feature['outcome_flag']==1) & (patient_level_feature['feature_value']!=9999999999)]['feature_value']

        x0=patient_level_feature[(patient_level_feature['outcome_flag']==0) & (patient_level_feature['feature_value']!=9999999999)]['feature_value']

        lavg1.append(x1.mean())

        lsd1.append(x1.std())

        lavg0.append(x0.mean())

        lsd0.append( x0.std())

        fn.append('recency_'+str(k)+'__'+str(j))
## Record all the above stats for using the below naming convention.

fitness = pd.DataFrame([fn, lavg1, lavg0, lsd1, lsd0]).transpose().fillna(0)

fitness.columns = ['feature_name', 'avg_1', 'avg_0', 'sd_1', 'sd_0']

fitness['fitness_value'] = fitness.apply(fitness_calculation, axis=1)

fitness ## calculated Fitness Score..




# for k in ['avg_1', 'avg_0', 'sd_1', 'sd_0']:

    
rec=ftns[ftns.feature_name.str.startswith("recency")]

print(rec.shape,fitness.shape)

rec.sort_values('feature_name',inplace=True)

fitness.sort_values('feature_name',inplace=True)



from sklearn.metrics import mean_squared_error

for k in ['avg_1', 'avg_0', 'sd_1', 'sd_0']:

    print(mean_squared_error(rec[k],fitness[k]))
# train_data[train_data['event_time']<60].groupby([id_var, 'plan_type']).agg({time_var: len}).reset_index().sort_values(['patient_id','plan_type'])
# train_data.groupby([id_var, 'specialty']).agg({'event_time':lambda x : (x<60).count()}).reset_index()
# s=train_data.groupby([id_var, 'plan_type']).agg({'event_time':lambda x : sum(x<60)}).reset_index()

# # s[s['event_time']<30]

# s
# s[s.duplicated()==True]


# np.dot(['event_name'],['1','2'])


# for t in tqdm(([x for x in range(30,0,-30)][::-1])):

    
# ft
ftns[ftns.feature_name.str.startswith('frequency_30_plan_type')]
from itertools import product

# list(product(train_data['plan_type'].unique().tolist(),train_data[id_var].unique().tolist()))
# pd.DataFrame(list(product(train_data['plan_type'].unique().tolist(),train_data[id_var].unique().tolist())),columns=['s1','p1'])
# _df1 = pd.DataFrame(_freq['feature_name'].unique().tolist(), columns=['feature_name'])

# _df2 = pd.DataFrame(train_data[id_var].unique().tolist(), columns=[id_var])

# _df1['key'] = 1

# _df2['key'] = 1

# _freqTotal = pd.merge(_df2, _df1, on='key')
# list(product(train_data['specialty'].unique(),[x for x in range(1080,0,-30)]))
# event_time_list=list(product(column,[x for x in range(1080,0,-30)]))

# _df3=train_data[[id_var,y_var]].drop_duplicates().reset_index(drop=True)

# ft=pd.DataFrame()

# ft1=pd.DataFrame()

# column=['plan_type']

# for kk in tqdm(column):

#     for t in tqdm(([x for x in range(1080,0,-30)][::-1])):

        

# for kk,t in tqdm(event_time_list):

#     _data = train_data[(train_data[time_var]<=int(t))].reset_index(drop=True)

#     _freq = _data[[id_var, kk, time_var]].groupby([id_var, kk]).agg({time_var: len}).reset_index()

#     _freq.columns = [id_var, 'feature_name', 'feature_value']

#     _freq['feature_name'] = 'frequency_' + str(t) + '_' +str(kk)+'__'+ _freq['feature_name'].astype(str)

#     _freq = _freq.reset_index(drop=True)

#     freqTotal=pd.DataFrame(product(train_data[id_var].unique(),_freq['feature_name'].unique()),columns=['patient_id','feature_name'])

#     freqTotal = pd.merge(freqTotal, _freq, on=[id_var, 'feature_name'], how='left')

#     freqTotal.fillna(0, inplace=True)

#     freqTotal = freqTotal.merge(_df3, on=id_var, how='left')

#     xx1=freqTotal.loc[freqTotal[y_var]==1,['feature_name', 'feature_value']].groupby('feature_name').agg({'feature_value':['mean','std']}).reset_index()

#     xx1.columns=['feature_name','avg_1','sd_1']

#     xx0=freqTotal.loc[freqTotal[y_var]==0,['feature_name', 'feature_value']].groupby('feature_name').agg({'feature_value':['mean','std']}).reset_index()

#     xx0.columns=['feature_name','avg_0','sd_0']

#     _fitness_value = pd.merge(xx1, xx0, on='feature_name', how='left')

#     ft=ft.append(_fitness_value,ignore_index=True)



_df3=train_data[[id_var,y_var]].drop_duplicates().reset_index(drop=True)

ft=pd.DataFrame()

ft1=pd.DataFrame()

for kk in tqdm(column):

    for t in tqdm(([x for x in range(1080,0,-30)][::-1])):

#         _data = train_data[(train_data[time_var]<=int(t))].reset_index(drop=True)

        _freq = train_data[(train_data[time_var]<=int(t))][[id_var, kk, time_var]].groupby([id_var, kk]).agg({time_var: len}).reset_index()

        _freq.columns = [id_var, 'feature_name', 'feature_value']

        _freq['feature_name'] = 'frequency_' + str(t) + '_' +str(kk)+'__'+ _freq['feature_name'].astype(str)

        _freq = _freq.reset_index(drop=True)

        freqTotal=pd.DataFrame(product(train_data[id_var].unique(),_freq['feature_name'].unique()),columns=['patient_id','feature_name'])

        freqTotal = pd.merge(freqTotal, _freq, on=[id_var, 'feature_name'], how='left')

        freqTotal.fillna(0, inplace=True)

        freqTotal = freqTotal.merge(_df3, on=id_var, how='left')

        xx1=freqTotal.loc[freqTotal[y_var]==1,['feature_name', 'feature_value']].groupby('feature_name').agg({'feature_value':['mean','std']}).reset_index()

        xx1.columns=['feature_name','avg_1','sd_1']

        xx0=freqTotal.loc[freqTotal[y_var]==0,['feature_name', 'feature_value']].groupby('feature_name').agg({'feature_value':['mean','std']}).reset_index()

        xx0.columns=['feature_name','avg_0','sd_0']

        _fitness_value = pd.merge(xx1, xx0, on='feature_name', how='left')

        ft=ft.append(_fitness_value,ignore_index=True)
all_specs=[]

for k,j in list(product(train_data['specialty'].unique(),[x for x in range(1080,0,-30)])):

    if 'frequency_' + str(j) + '_specialty' +'__'+ k not in ft.feature_name.values:

        ft=ft.append({'feature_name':'frequency_' + str(j) + '_specialty' +'__'+ k,

                   'avg_1':0,'avg_0':0,'sd_0':0,'sd_1':0},ignore_index=True)

                 

#         all_specs.append('frequency_' + str(j) + '_specialty' +'__'+ k)

# len(all_specs)

ft



# ft['feature_name'].append(np.setdiff1d(all_specs,ft.feature_name.values))
ft
ftns[(ftns.feature_name.str.contains("specialty")) &(ftns.feature_name.str.contains("frequency"))]
frq=ftns[ftns.feature_name.str.startswith("frequency")]

print(frq.shape,ft.shape)

frq.sort_values('feature_name',inplace=True)

ft.sort_values('feature_name',inplace=True)



from sklearn.metrics import mean_squared_error

for k in ['avg_1', 'avg_0', 'sd_1', 'sd_0']:

    print(mean_squared_error(frq[k],ft[k]))
# ft=pd.DataFrame()

# ft1=pd.DataFrame()

# column=['plan_type']

# for kk in tqdm(column):

#     for t in tqdm(([x for x in range(540,0,-30)][::-1])):



#         _data = train_data[(train_data[time_var]<=int(t))].reset_index(drop=True)

# #         print(_data.nunique())

#         _freq = _data[[id_var, kk, time_var]].groupby([id_var, kk]).agg({time_var: len}).reset_index()

#         _freq.columns = [id_var, 'feature_name', 'feature_value']

#         _freq['feature_name'] = 'frequency__' + kk + '__' + _freq['feature_name'].astype(str) + '__' + str(t)

#         _freq = _freq.reset_index(drop=True)

# #         print(_freq.nunique())

#         _df1 = pd.DataFrame(_freq['feature_name'].unique().tolist(), columns=['feature_name'])

#         _df2 = pd.DataFrame(_freq[id_var].unique().tolist(), columns=[id_var])

#         _df1['key'] = 1

#         _df2['key'] = 1

#         _freqTotal = pd.merge(_df2, _df1, on='key')

#         _freqTotal.drop(['key'], axis=1, inplace=True)

#         # _freqTotal=pd.DataFrame({id_var:_freq[id_var].unique().tolist(),'feature_name':_freq['feature_name'].unique().tolist()})

# #         _freqTotal = pd.merge(pd.DataFrame({id_var:_freq[id_var].unique().tolist(),'feature_name':_freq['feature_name'].unique().tolist()}), _freq, on=[id_var, 'feature_name'], how='left')

#         _freqTotal.fillna(0, inplace=True)

#         _df3 = train_data[[id_var,y_var]].drop_duplicates().reset_index(drop=True)

#         _freqTotal = _freqTotal.merge(_df3, on=id_var, how='left')

#         freqTotal = _freqTotal.copy()

#         x1=freqTotal.loc[freqTotal[y_var]==1,['feature_name', 'feature_value']].groupby('feature_name')

#         x0=freqTotal.loc[freqTotal[y_var]==0,['feature_name', 'feature_value']].groupby('feature_name')

#         _avg1 = x1.mean().reset_index()

#         _avg1.columns = ['feature_name', 'avg_1']

#         _sd1 = x1.agg(np.std).reset_index()

#         _sd1.columns = ['feature_name', 'sd_1']

#         _avg0 = x0.mean().reset_index()

#         _avg0.columns = ['feature_name', 'avg_0']

#         _sd0 = x0.agg(np.std).reset_index()

#         _sd0.columns = ['feature_name', 'sd_0']



#         _fitness_value = pd.merge(_avg1, _avg0, on='feature_name', how='left')

#         _fitness_value = pd.merge(_fitness_value, _sd1, on='feature_name', how='left')

#         _fitness_value = pd.merge(_fitness_value, _sd0, on='feature_name', how='left')

# #         print(_fitness_value.shape)

#         ft=ft.append(_fitness_value,ignore_index=True)
# ft=ft.append(_fitness_value)

# ft

# ft[ft.feature_name.isin(ftns.feature_name)]

# ftns[(ftns.feature_name.str.startswith('frequency__specialty')) & (ftns.feature_name.str.endswith('_30'))]

ftns[ftns.feature_name.str.startswith('frequency_')][~ftns[ftns.feature_name.str.startswith('frequency_')].feature_name.isin(ft.feature_name)]
ft.shape
ft['fitness_value'] = ft.apply(fitness_calculation, axis=1)

# ft
# ft=ft.append(ftns[ftns.feature_name.str.startswith('frequency_')][~ftns[ftns.feature_name.str.startswith('frequency_')].feature_name.isin(ft.feature_name)])
ft.shape
ft[ft.feature_name=='frequency__plan_type__plan_6__30']
# ftns[755:]

ftns[ftns.feature_name=='frequency__plan_type__plan_6__30']
ftns[ftns.feature_name.str.startswith('normChange__specialty')]
column
# event = feature_name.split('__')[1]

# value = feature_name.split('__')[2]

# time = feature_name.split('__')[3]

# __df3 = train_data[[id_var,y_var]].drop_duplicates().reset_index(drop=True)

ftnorm=pd.DataFrame()

# column=['plan_type']

for kk in tqdm(column):

    for t in tqdm(([x for x in range(540,0,-30)][::-1])):

        _data_post = train_data[train_data[time_var]<=int(t)].reset_index(drop=True)

        _data_pre = train_data[train_data[time_var]>int(t)].reset_index(drop=True)

        _freq_post = _data_post[[id_var, kk, time_var]].groupby([id_var, kk]).agg({time_var: len}).reset_index()

        _freq_pre = _data_pre[[id_var, kk, time_var]].groupby([id_var, kk]).agg({time_var: len}).reset_index()

#         print(_freq_post.nunique(),_freq_pre.nunique())

        _freq_post.columns = [id_var, 'feature_name', 'feature_value_post']

        _freq_pre.columns = [id_var, 'feature_name', 'feature_value_pre']

        _freq_post['feature_value_post'] = _freq_post['feature_value_post']/int(t)

        _freq_pre['feature_value_pre'] = _freq_pre['feature_value_pre']/((train_data[time_var].max()) - int(t))

        _normChange = pd.merge(_freq_post, _freq_pre, on=[id_var, 'feature_name'], how='outer')

        _normChange.fillna(0, inplace=True)

        _normChange['feature_value'] = np.where(_normChange['feature_value_post']>_normChange['feature_value_pre'], 1, 0)

        _normChange.drop(['feature_value_post', 'feature_value_pre'], axis=1, inplace=True)

#         _normChange['feature_name'] = 'normChange__' + kk + '__' + _normChange['feature_name'].astype(str) + '__' + str(t)

        _normChange['feature_name'] = 'normChange_' + str(t)+'_'+ kk + '__' + _normChange['feature_name'].astype(str)



        _normChange = _normChange.reset_index(drop=True)

        _df1 = pd.DataFrame(_normChange['feature_name'].unique().tolist(), columns=['feature_name'])

        _df2 = pd.DataFrame(_normChange[id_var].unique().tolist(), columns=[id_var])

        _df1['key'] = 1

        _df2['key'] = 1

        _normTotal = pd.merge(_df2, _df1, on='key')

        _normTotal.drop(['key'], axis=1, inplace=True)

        _normTotal = pd.merge(_normTotal, _normChange, on=[id_var, 'feature_name'], how='left')

        _normTotal.fillna(0, inplace=True)

        

        _normTotal = _normTotal.merge(_df3, on=id_var, how='left')

        normTotal = _normTotal.copy()



#         _avg1 = normTotal.loc[normTotal[y_var]==1,['feature_name', 'feature_value']].groupby('feature_name').mean().reset_index()

#         _avg1.columns = ['feature_name', 'avg_1']

#         _sd1 = normTotal.loc[normTotal[y_var]==1,['feature_name', 'feature_value']].groupby('feature_name').agg(np.std).reset_index()

#         _sd1.columns = ['feature_name', 'sd_1']

#         _avg0 = normTotal.loc[normTotal[y_var]==0,['feature_name', 'feature_value']].groupby('feature_name').mean().reset_index()

#         _avg0.columns = ['feature_name', 'avg_0']

#         _sd0 = normTotal.loc[normTotal[y_var]==0,['feature_name', 'feature_value']].groupby('feature_name').agg(np.std).reset_index()

#         _sd0.columns = ['feature_name', 'sd_0']

        

        xx1=normTotal.loc[normTotal[y_var]==1,['feature_name', 'feature_value']].groupby('feature_name').agg({'feature_value':['mean','std']}).reset_index()

        xx1.columns=['feature_name','avg_1','sd_1']

        xx0=normTotal.loc[normTotal[y_var]==0,['feature_name', 'feature_value']].groupby('feature_name').agg({'feature_value':['mean','std']}).reset_index()

        xx0.columns=['feature_name','avg_0','sd_0']

        _fitness_value = pd.merge(xx1, xx0, on='feature_name', how='left')



#         _fitness_value = pd.merge(_avg1, _avg0, on='feature_name', how='left')

#         _fitness_value = pd.merge(_fitness_value, _sd1, on='feature_name', how='left')

#         _fitness_value = pd.merge(_fitness_value, _sd0, on='feature_name', how='left')

        ftnorm=ftnorm.append(_fitness_value)

ftns[(ftns.feature_name.str.startswith('normChange__specialty')) & (ftns.feature_name.str.endswith('_30'))]
# fitness = _fitness_value[_fitness_value.feature_name==feature_name]

ftnorm['fitness_value'] = ftnorm.apply(fitness_calculation, axis=1)

ftnorm ## Calculated Fitness Scores for Frequency..
ftnorm[ftnorm.feature_name=='normChange__event_name__event_10__30']
ftns[ftns.feature_name=='normChange__event_name__event_10__30']
ft.shape,fitness.shape,ftnorm.shape
frq=ftns[ftns.feature_name.str.startswith("norm")]

print(frq.shape,ftnorm.shape)

ftnorm.sort_values('feature_name',inplace=True)

frq.sort_values('feature_name',inplace=True)



from sklearn.metrics import mean_squared_error

for k in ['avg_1', 'avg_0', 'sd_1', 'sd_0']:

    print(mean_squared_error(frq[k],ftnorm[k]))
ft=ft.append(fitness,ignore_index=True)

ft=ft.append(ftnorm,ignore_index=True)

ft.shape
ft
ftns[ftns.feature_name=='normChange__specialty__spec_96__30']
ft.to_csv('final_fitness_v2.csv',index=False)
print(time.time()-start)