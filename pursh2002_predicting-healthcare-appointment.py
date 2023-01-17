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

import matplotlib as plt
df = pd.read_csv('/kaggle/input/noshowappointments/KaggleV2-May-2016.csv')
df.shape
df.head(5)
df.columns
df.describe
df['No-show'].value_counts()
df['OUTPUT_LABEL'] = (df['No-show'] == 'Yes').astype(int)
# check the prevalence of our OUTPUT_LABEL:

def calc_prevalence(y):

    return (sum(y)/len(y))
calc_prevalence(df.OUTPUT_LABEL.values) 
df.ScheduledDay.head()
df.AppointmentDay.head()
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'], 

 format = '%Y-%m-%dT%H:%M:%SZ',errors = 'coerce')

df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'],format = '%Y-%m-%dT%H:%M:%SZ',errors = 'coerce')
df['ScheduledDay']
assert df.ScheduledDay.isnull().sum() == 0,'missing ScheduledDay Dates'

assert df.AppointmentDay.isnull().sum() == 0, 'missing AppointmentDay dates'
df['AppointmentDay']
df.ScheduledDay.isnull().sum()
df.AppointmentDay.isnull().sum()
(df['ScheduledDay'] > df['AppointmentDay']).sum() # there are ~40k appointments that were scheduled after the appointment datetime.
df['AppointmentDay'] = df['AppointmentDay'] + pd.Timedelta('1d') - pd.Timedelta('1s')
df['AppointmentDay']
(df['ScheduledDay'] > df['AppointmentDay']).sum()# With this change there are only 1 rows where scheduledDay time is after appointmentDay. Let’s just drop those rows.
df.loc[(df['ScheduledDay'] <= df['AppointmentDay'])].copy()
df['ScheduledDay_year'] = df['ScheduledDay'].dt.year

df['SheduledDay_month'] = df['ScheduledDay'].dt.month

df['ScheduledDay_week'] = df['ScheduledDay'].dt.week

df['ScheduledDay_day'] = df['ScheduledDay'].dt.day

df['ScheduledDay_hour'] = df['ScheduledDay'].dt.hour

df['ScheduledDay_minute'] = df['ScheduledDay'].dt.minute

df['ScheduledDay_dayofweek'] = df['ScheduledDay'].dt.dayofweek



df['AppointmentDay_year'] = df['AppointmentDay'].dt.year

df['AppointmentDay_month'] = df['AppointmentDay'].dt.month

df['AppointmentDay_week'] = df['AppointmentDay'].dt.week

df['AppointmentDay_day'] = df['AppointmentDay'].dt.day

df['AppointmentDay_hour'] = df['AppointmentDay'].dt.hour

df['AppointmentDay_minute'] = df['AppointmentDay'].dt.minute

df['AppointmentDay_dayofweek'] = df['AppointmentDay'].dt.dayofweek
df[['ScheduledDay_year','SheduledDay_month','ScheduledDay_week','ScheduledDay_day','ScheduledDay_hour','ScheduledDay_minute','ScheduledDay_dayofweek']]
df[['AppointmentDay_year','AppointmentDay_month','AppointmentDay_week','AppointmentDay_day','AppointmentDay_hour','AppointmentDay_minute','AppointmentDay_dayofweek']]
df.groupby('AppointmentDay_year').size()
df.groupby('AppointmentDay_month').size()
df.groupby('AppointmentDay_dayofweek').size()
df.groupby('AppointmentDay_dayofweek').apply(lambda g:calc_prevalence(g.OUTPUT_LABEL.values))
df['delta_days'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.total_seconds()/(60*60*24) # There is a function dt.day u can use that 1) dt.days rounds to the nearest day, 2) dt.days used to take much longer than total_seconds. 
import matplotlib.pyplot as plt



plt.hist(df.loc[df.OUTPUT_LABEL == 1,'delta_days'], 

 label = 'Missed',bins = range(0,60,1), normed = True)

plt.hist(df.loc[df.OUTPUT_LABEL == 0,'delta_days'], 

 label = 'Not Missed',bins = range(0,60,1), normed = True,alpha =0.5)

plt.legend()

plt.xlabel('days until appointment')

plt.ylabel('normed distribution')

plt.xlim(0,40)

plt.show()
df = df.sample(n= len(df),random_state = 42)

df = df.reset_index(drop = True)



df_valid = df.sample(frac = 0.3,random_state = 42)

df_train = df.drop(df_valid.index)
print('Valid prevalence(n = %d):%.3f'%(len(df_valid),calc_prevalence(df_valid.OUTPUT_LABEL.values)))

print('Train prevalence(n = %d):%.3f'%(len(df_train), calc_prevalence(df_train.OUTPUT_LABEL.values)))
col2use = ['ScheduledDay_day', 'ScheduledDay_hour',

 'ScheduledDay_minute', 'ScheduledDay_dayofweek', 

 'AppointmentDay_day',

 'AppointmentDay_dayofweek', 'delta_days']
X_train = df_train[col2use].values

X_valid = df_valid[col2use].values

y_train = df_train['OUTPUT_LABEL'].values

y_valid = df_valid['OUTPUT_LABEL'].values

print('Training shapes:',X_train.shape, y_train.shape)

print('Validation shapes:',X_valid.shape, y_valid.shape)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth = 5, n_estimators = 100, random_state = 42)

rf.fit(X_train,y_train)
y_train_preds = rf.predict_proba(X_train)[:,1]

y_valid_preds = rf.predict_proba(X_valid)[:,1]
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

def calc_specificity(y_actual, y_pred, thresh):

    return sum((y_pred < thresh) & (y_actual == 0)) /sum(y_actual ==0)

def print_report(y_actual, y_pred, thresh):

    auc = roc_auc_score(y_actual, y_pred)

    accuracy = accuracy_score(y_actual, (y_pred > thresh))

    recall = recall_score(y_actual, (y_pred > thresh))

    precision = precision_score(y_actual, (y_pred > thresh))

    specificity = calc_specificity(y_actual, y_pred, thresh)

    print('AUC:%.3f'%auc)

    print('accuracy:%.3f'%accuracy)

    print('recall:%.3f'%recall)

    print('precision:%.3f'%precision)

    print('specificity:%.3f'%specificity)

    print('prevalence:%.3f'%calc_prevalence(y_actual))

    print(' ')

    return auc, accuracy, recall, precision, specificity

    
thresh = 0.201

print('Training:')

print_report(y_train,y_train_preds, thresh)

print("Validation:")

print_report(y_valid,y_valid_preds, thresh)
from sklearn.metrics import roc_curve

fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_preds)

auc_train = roc_auc_score(y_train, y_train_preds)

fpr_valid, tpr_valid, thresholds_valid = roc_curve(y_valid, y_valid_preds)

auc_valid = roc_auc_score(y_valid, y_valid_preds)

plt.plot(fpr_train, tpr_train, 'r-',label ='Train AUC:%.3f'%auc_train)

plt.plot(fpr_valid, tpr_valid, 'b-',label ='Valid AUC:%.3f'%auc_valid)

plt.plot([0,1],[0,1],'k')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend()

plt.show()