
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


%matplotlib inline
df = pd.read_csv('../input/noshowappointments/KaggleV2-May-2016.csv')
df['OUTPUT_LABEL'] = (df['No-show'] == 'Yes').astype('int')
def calc_prevalence(y):
#     print(sum(y))
#     print(len(y))
    return (sum(y)/len(y))
print(calc_prevalence(df['OUTPUT_LABEL'].values))
df['OUTPUT_LABEL'].value_counts().plot(kind='bar')
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'], format = '%Y-%m-%dT%H:%M:%SZ', errors = 'coerce')
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'], format = '%Y-%m-%dT%H:%M:%SZ', errors = 'coerce')
assert df.ScheduledDay.isnull().sum() == 0, 'missing ScheduledDay dates'
assert df.AppointmentDay.isnull().sum() == 0, 'missing AppointmentDay dates'
df['ScheduledDay'].head()
(df['ScheduledDay'] > df['AppointmentDay']).sum()
df['AppointmentDay'].head()
df['AppointmentDay'] = df['AppointmentDay'] + pd.Timedelta('1d') - pd.Timedelta('1s')
df['AppointmentDay'].head()
(df['ScheduledDay'] > df['AppointmentDay']).sum()
df = df.loc[(df['ScheduledDay'] <= df['AppointmentDay'])].copy()
(df['ScheduledDay'] > df['AppointmentDay']).sum()
df['ScheduledDay_year'] = df['ScheduledDay'].dt.year
df['ScheduledDay_month'] = df['ScheduledDay'].dt.month
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
df[['ScheduledDay','ScheduledDay_year','ScheduledDay_month','ScheduledDay_week','ScheduledDay_day',
   'ScheduledDay_hour','ScheduledDay_minute','ScheduledDay_dayofweek']].head()
print(df.groupby('AppointmentDay_year').size(), end="\n\n")
print(df.groupby('ScheduledDay_year').size())
print(df.groupby('AppointmentDay_month').size(), end='\n\n')
print(df.groupby('ScheduledDay_month').size())
print(df.groupby('AppointmentDay_dayofweek').size(), end='\n\n')
print(df.groupby('ScheduledDay_dayofweek').size())
df.groupby('AppointmentDay_dayofweek').apply(lambda g: calc_prevalence(g.OUTPUT_LABEL.values))
df.groupby('ScheduledDay_dayofweek').apply(lambda g: calc_prevalence(g.OUTPUT_LABEL.values))
df['delta_days'] = (df['AppointmentDay']-df['ScheduledDay']).dt.total_seconds()/(60*60*24)
print(df[['ScheduledDay', 'AppointmentDay', 'delta_days']].head())
print(df[['ScheduledDay', 'AppointmentDay', 'delta_days']].tail())
plt.hist(df.loc[df.OUTPUT_LABEL == 1,'delta_days'], color='red',
         label = 'Missed',bins = range(0,60,1), density = True)
plt.hist(df.loc[df.OUTPUT_LABEL == 0,'delta_days'], color='blue',
         label = 'Not Missed',bins = range(0,60,1), density = True,alpha =0.5)
plt.legend()
plt.xlabel('days until appointment')
plt.ylabel('normed distribution')
plt.xlim(0,40)
# shuffle the samples
df = df.sample(n = len(df), random_state = 42)
df = df.reset_index(drop = True)
df_valid = df.sample(frac = 0.3, random_state = 42)
df_train = df.drop(df_valid.index)
print('Valid prevalence(n = %d):%.3f'%(len(df_valid),calc_prevalence(df_valid.OUTPUT_LABEL.values)))
print('Train prevalence(n = %d):%.3f'%(len(df_train),calc_prevalence(df_train.OUTPUT_LABEL.values)))
col2use = ['ScheduledDay_day','ScheduledDay_hour',
           'ScheduledDay_minute','ScheduledDay_dayofweek',
           'AppointmentDay_day','AppointmentDay_dayofweek',
           'delta_days']
X_train = df_train[col2use].values
X_valid = df_valid[col2use].values
y_train = df_train['OUTPUT_LABEL'].values
y_valid = df_valid['OUTPUT_LABEL'].values
print('Training shapes:',X_train.shape, y_train.shape)
print('Validation shapes:',X_valid.shape, y_valid.shape)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth = 5, n_estimators=100, random_state = 42)
rf.fit(X_train, y_train)
y_train_preds = rf.predict_proba(X_train)[:,1]
y_valid_preds = rf.predict_proba(X_valid)[:,1]
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
def calc_specificity(y_actual, y_pred, thresh):
 # calculates specificity
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
print('Traing: ')
print_report(y_train, y_train_preds, 0.201)
print('Validation')
print_report(y_valid, y_valid_preds, 0.201)
from sklearn.metrics import roc_curve

fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_preds)
auc_train = roc_auc_score(y_train, y_train_preds)
fpr_valid, tpr_valid, thresholds_valid = roc_curve(y_valid, y_valid_preds)
auc_valid = roc_auc_score(y_valid, y_valid_preds)

plt.plot(fpr_train, tpr_train, 'r-',label ='Train AUC:%.3f'%auc_train)
plt.plot(fpr_valid, tpr_valid, 'b-',label ='Valid AUC:%.3f'%auc_valid)
plt.plot([0,1],[0,1],'k—')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()