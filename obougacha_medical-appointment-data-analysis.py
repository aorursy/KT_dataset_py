import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns 



base_color = sns.color_palette()[0]
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data_df = pd.read_csv("/kaggle/input/noshowappointments/KaggleV2-May-2016.csv")

data_df.head(2)
data_df.isnull().sum()
data_df.dtypes
data_df['Age'].describe()
data_df['Neighbourhood'].unique()
data_df.duplicated().sum()
cleaned_data = data_df.copy()
cleaned_data.rename(columns={'Hipertension': 'Hypertension'}, inplace=True)
cleaned_data.head(2)
neg_age_idx = cleaned_data[cleaned_data['Age']<0].index.tolist()

cleaned_data.drop(neg_age_idx, inplace = True)

cleaned_data.reset_index(drop=True, inplace=True)
cleaned_data[cleaned_data['Age']<0]
cleaned_data['PatientId'] = cleaned_data['PatientId'].astype('int64')
cleaned_data.dtypes
cleaned_data['ScheduledDay'] = pd.to_datetime(cleaned_data['ScheduledDay'])

cleaned_data['AppointmentDay'] = pd.to_datetime(cleaned_data['AppointmentDay'])
cleaned_data.dtypes
cleaned_data['Scholarship'] = cleaned_data['Scholarship'].astype('bool')

cleaned_data['Hypertension'] = cleaned_data['Hypertension'].astype('bool')

cleaned_data['Diabetes'] = cleaned_data['Diabetes'].astype('bool')

cleaned_data['Alcoholism'] = cleaned_data['Alcoholism'].astype('bool')

cleaned_data["Handcap"] = cleaned_data['Handcap'].apply(lambda x: True if x>=1 else False)

cleaned_data['No-show'] = cleaned_data['No-show'].apply(lambda x : True if x=='Yes' else False)
cleaned_data.dtypes
appointment_df = cleaned_data[['AppointmentID', 'PatientId', 'ScheduledDay','AppointmentDay', 

                               'SMS_received', 'No-show']]

patient_df = cleaned_data[['PatientId', 'Gender','Age', 'Neighbourhood', 'Scholarship', 'Hypertension',

                             'Diabetes', 'Alcoholism', 'Handcap']]
patient_df['PatientId'].duplicated().sum()
idx = patient_df[patient_df['PatientId'].duplicated()].index.tolist()

patient_df.drop(idx,inplace=True)

patient_df.reset_index(drop=True, inplace=True)
patient_df.duplicated().sum()
appointment_df.to_csv('Appointment_Data.csv', index=False)

patient_df.to_csv('Patient_Data.csv', index=False)
patient_df['Age'].hist(bins=100)

plt.xlabel('Patient Age (years)')

plt.ylabel('Count')

plt.title("Histogram of Patients' Age")

plt.show()
sns.catplot(x='Gender', data = patient_df, kind='count', color=base_color)

plt.xlabel("Patient's Gender")

plt.ylabel("Count")

plt.title("Patient's Gender Distribution")

plt.show()
sns.catplot(y='Neighbourhood', data = patient_df, kind='count', color=base_color, height=12)

plt.ylabel("Patient's Neighbourhood")

plt.xlabel("Count")

plt.title("Patient's Neighbourhood Distribution")

plt.show()
sns.catplot(x='Scholarship', data = patient_df, kind='count', color=base_color)

plt.xlabel("Patient's Scholarship")

plt.ylabel("Count")

plt.title("Patient's Scholarship Distribution")

plt.show()
fig = plt.figure(figsize=(10,10))

ax1 = fig.add_subplot(2,2,1)

ax2 = fig.add_subplot(2,2,2)

ax3 = fig.add_subplot(2,2,3)

ax4 = fig.add_subplot(2,2,4)

g = sns.countplot(x='Hypertension', data = patient_df, color=base_color, ax=ax1)

ax1.set_xlabel("Patient's Hypertension Status")

ax1.set_ylabel("Count")

ax1.set_title("Patient's Hypertension Status Distribution")

g = sns.countplot(x='Diabetes', data = patient_df, color=base_color, ax=ax2)

ax2.set_xlabel("Patient's Diabetes Status")

ax2.set_ylabel("Count")

ax2.set_title("Patient's Diabetes Status Distribution")

g = sns.countplot(x='Alcoholism', data = patient_df, color=base_color, ax=ax3)

ax3.set_xlabel("Patient's Alcoholism Status")

ax3.set_ylabel("Count")

ax3.set_title("Patient's Alcoholism Status Distribution")

g = sns.countplot(x='Handcap', data = patient_df, color=base_color, ax=ax4)

ax4.set_xlabel("Patient's Handcap Status")

ax4.set_ylabel("Count")

ax4.set_title("Patient's Handcap Status Distribution")

plt.tight_layout()

plt.show()
sns.catplot(x='SMS_received', data = appointment_df, kind='count', color=base_color)

plt.xlabel("How Many SMS are Send to / Received by the Patient")

plt.ylabel("Count")

plt.title("Number of Received SMS")

plt.show()
sns.catplot(x='No-show', data = appointment_df, kind='count', color=base_color)

plt.xlabel("Did the Patient Miss the Appointment?")

plt.ylabel("Count")

plt.title("Number of No Show")

plt.show()
weekDays = ("Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday")
appointment_df['Appointment_w_d'] = appointment_df['AppointmentDay'].apply(lambda x: weekDays[x.weekday()])
appointment_df['DaysToAppointment'] = appointment_df.apply(lambda x: 0 if x['AppointmentDay']<x['ScheduledDay'] else (x['AppointmentDay']-x['ScheduledDay']).days, axis = 1)
appointment_df['CallHour'] = appointment_df['ScheduledDay'].apply(lambda x: x.hour)
appointment_df.head(2)
appointment_df['DaysToAppointment'].hist(bins=100)

plt.xlabel('Number of Days Between the Call and the Appointment (days)')

plt.ylabel('Count')

plt.title('Histogram of Number of Days to Appointment')

plt.show()
sns.catplot(x='Appointment_w_d', data = appointment_df, kind='count', order = weekDays, color=base_color, aspect=1.5)

plt.xticks(rotation=45)

plt.xlabel("Week Day of the Appointment")

plt.ylabel("Count")

plt.title("Number of Appointment per Day of Week")

plt.show()
sns.catplot(x='CallHour', data = appointment_df, kind='count', order = range(24), color=base_color, aspect=1.6)

plt.xlabel("Hour of Call")

plt.ylabel("Count")

plt.title("Hour of Call for Making an Appointment")

plt.show()
d_m = patient_df[patient_df['Gender']=='M']

sns.distplot(d_m['Age'],kde=False, label='Male')

d_f = patient_df[patient_df['Gender']=='F']

sns.distplot(d_f['Age'],kde=False, label='Female')

plt.legend(prop={'size': 12})

plt.title('Distrubtion of Age per Gender')

plt.xlabel('Patient Age')

plt.ylabel('Count')

plt.show()
d_m = patient_df[patient_df['Scholarship']]

sns.distplot(d_m['Age'],kde=False, label='Scholarship Owners')

d_f = patient_df[patient_df['Scholarship']==False]

sns.distplot(d_f['Age'],kde=False, label='No Scholarship')

plt.legend(prop={'size': 12})

plt.title('Distrubtion of Age per Scholarship Ownship')

plt.xlabel('Patient Age')

plt.ylabel('Count')

plt.show()
d_m = patient_df[patient_df['Hypertension']]

sns.distplot(d_m['Age'],kde=False, label='Suffers From Hypertension')

d_f = patient_df[patient_df['Hypertension']==False]

sns.distplot(d_f['Age'],kde=False, label='No Hypertension')

plt.legend(prop={'size': 12})

plt.title('Distrubtion of Age per Hypertension')

plt.xlabel('Patient Age')

plt.ylabel('Count')

plt.show()
d_m = patient_df[patient_df['Diabetes']]

sns.distplot(d_m['Age'],kde=False, label='Suffers From Diabetes')

d_f = patient_df[patient_df['Diabetes']==False]

sns.distplot(d_f['Age'],kde=False, label='No Diabetes')

plt.legend(prop={'size': 12})

plt.title('Distrubtion of Age per Diabetes')

plt.xlabel('Patient Age')

plt.ylabel('Count')

plt.show()
d_m = patient_df[patient_df['Alcoholism']]

sns.distplot(d_m['Age'],kde=False, label='Alcoholic')

d_f = patient_df[patient_df['Alcoholism']==False]

sns.distplot(d_f['Age'],kde=False, label='Not Alcoholic')

plt.legend(prop={'size': 12})

plt.title('Distrubtion of Age per Alcoholism')

plt.xlabel('Patient Age')

plt.ylabel('Count')

plt.show()
d_m = patient_df[patient_df['Handcap']]

sns.distplot(d_m['Age'],kde=False, label='Challenged Patients')

d_f = patient_df[patient_df['Handcap']==False]

sns.distplot(d_f['Age'],kde=False, label='Normal Patients')

plt.legend(prop={'size': 12})

plt.title('Distrubtion of Age per Status')

plt.xlabel('Patient Age')

plt.ylabel('Count')

plt.show()
sns.catplot(y='Neighbourhood', data = patient_df, hue = 'Gender',

            kind='count', height=12)

plt.ylabel("Patient's Neighbourhood")

plt.xlabel("Count")

plt.title("Patient's Neighbourhood Distribution per Gender")

plt.show()
sns.catplot(y='Neighbourhood', data = patient_df, hue = 'Scholarship',

            kind='count', height=12)

plt.ylabel("Patient's Neighbourhood")

plt.xlabel("Count")

plt.title("Patient's Neighbourhood Distribution per Scholarship")

plt.show()
sns.catplot(y='Neighbourhood', data = patient_df, hue = 'Hypertension',

            kind='count', height=12)

plt.ylabel("Patient's Neighbourhood")

plt.xlabel("Count")

plt.title("Patient's Neighbourhood Distribution per Hypertension")

plt.show()
sns.catplot(y='Neighbourhood', data = patient_df, hue = 'Alcoholism',

            kind='count', height=12)

plt.ylabel("Patient's Neighbourhood")

plt.xlabel("Count")

plt.title("Patient's Neighbourhood Distribution per Alcoholism")

plt.show()
sns.catplot(x='SMS_received', data = appointment_df, kind='count', hue='No-show')

plt.xlabel("How Many SMS are Send to / Received by the Patient")

plt.ylabel("Count")

plt.title("Number of Missed Appointement to Received SMS")

plt.show()
d_m = appointment_df[appointment_df['No-show']]

sns.distplot(d_m['DaysToAppointment'],kde=False, label='No-show Patients')

d_f = appointment_df[appointment_df['No-show']==False]

sns.distplot(d_f['DaysToAppointment'],kde=False, label='Patients that showed up')

plt.legend(prop={'size': 12})

plt.title('Distrubtion of the Number of No-show per Number of Days to Appointment')

plt.xlabel('Number of Days to Appointment')

plt.ylabel('Count')

plt.show()
d_m = appointment_df[appointment_df['SMS_received']==1]

sns.distplot(d_m['DaysToAppointment'],kde=False, label='Patients that received SMS')

d_f = appointment_df[appointment_df['SMS_received']==0]

sns.distplot(d_f['DaysToAppointment'],kde=False, label='No SMS reception')

plt.legend(prop={'size': 12})

plt.title('Distrubtion of SMS reception per Number of Days to Appointment')

plt.xlabel('Number of Days To Appointment')

plt.ylabel('Count')

plt.show()
sns.catplot(x='SMS_received', data = appointment_df[appointment_df['DaysToAppointment']>0],

            kind='count', hue='No-show')

plt.xlabel("How Many SMS are Send to / Received by the Patient")

plt.ylabel("Count")

plt.title("Number of Missed Appointement to Received SMS")

plt.show()
import statsmodels.api as sm

d_f1 = appointment_df[appointment_df['DaysToAppointment']>0]

sms_1 = d_f1[d_f1['SMS_received']==1]

sms_0 = d_f1[d_f1['SMS_received']==0]

counts = np.array([sms_0.shape[0]-sms_0['No-show'].sum(), 

                   sms_1.shape[0]-sms_1['No-show'].sum()])

nobs = np.array([sms_0.shape[0], sms_1.shape[0]])

zstat, pval = sm.stats.proportions_ztest(counts, nobs, alternative='smaller')

zstat, pval
sns.catplot(x='CallHour', data = appointment_df, kind='count', order = range(24), hue='No-show', aspect=2)

plt.xlabel("Hour of Call")

plt.ylabel("Count")

plt.title("Hour of Call for Making an Appointment vs Show or No-Show")

plt.show()
d_f1 = appointment_df[appointment_df['DaysToAppointment']==0]

sns.catplot(x='CallHour', data = d_f1, kind='count', order = range(24), hue='No-show', aspect=2)

plt.xlabel("Hour of Call")

plt.ylabel("Count")

plt.title("Hour of Call for Making an Appointment vs Show or No-Show")

plt.show()
sns.catplot(x='Appointment_w_d', data = appointment_df, kind='count', order = weekDays, hue='No-show',

            aspect=1.5)

plt.xticks(rotation=45)

plt.xlabel("Week Day of the Appointment")

plt.ylabel("Count")

plt.title("Number of Appointment Show/No-Show per Day of Week")

plt.show()
combined = appointment_df.merge(patient_df, on='PatientId')

combined.shape, appointment_df.shape, patient_df.shape
sns.catplot(x='Gender', data = combined, kind='count', hue='No-show')

plt.xlabel("Gender")

plt.ylabel("Count")

plt.title("Patient Gender vs Show or No-Show")

plt.show()
counts = np.array([combined[combined['Gender']=='F'].shape[0]-combined[combined['Gender']=='F']['No-show'].sum(), 

                   combined[combined['Gender']=='M'].shape[0]-combined[combined['Gender']=='M']['No-show'].sum()])

nobs = np.array([combined[combined['Gender']=='F'].shape[0], combined[combined['Gender']=='M'].shape[0]])

zstat, pval = sm.stats.proportions_ztest(counts, nobs, alternative='two-sided')

zstat, pval
d_m = combined[combined['No-show']]

sns.distplot(d_m['Age'],kde=False, label='No-Show Patients')

d_f = combined[combined['No-show']==False]

sns.distplot(d_f['Age'],kde=False, label='Showed Up Patients')

plt.legend(prop={'size': 12})

plt.title('Distrubtion of Age per No-Show/Show')

plt.xlabel('Patient Age')

plt.ylabel('Count')

plt.show()
sns.catplot(x='Scholarship', data = combined, kind='count', hue='No-show')

plt.xlabel("Do Patient Receive Aid?")

plt.ylabel("Count")

plt.title("Aids Receive vs Show or No-Show")

plt.show()
sns.catplot(x='Hypertension', data = combined, kind='count', hue='No-show')

plt.xlabel("Do Patient Have Hypertension")

plt.ylabel("Count")

plt.title("Hypertension vs Show or No-Show")

plt.show()
sns.catplot(x='Diabetes', data = combined, kind='count', hue='No-show')

plt.xlabel("Do Patient Have Diabetes")

plt.ylabel("Count")

plt.title("Diabetes vs Show or No-Show")

plt.show()
sns.catplot(x='Alcoholism', data = combined, kind='count', hue='No-show')

plt.xlabel("Do Patient Are Alcoholics?")

plt.ylabel("Count")

plt.title("Alcoholism vs Show or No-Show")

plt.show()
sns.catplot(x='Handcap', data = combined, kind='count', hue='No-show')

plt.xlabel("Do Patient Are Challenged?")

plt.ylabel("Count")

plt.title("Handcap vs Show or No-Show")

plt.show()
sns.scatterplot(x='Age', y='DaysToAppointment', data=combined)

plt.show()
sns.scatterplot(x='Age', y='DaysToAppointment', hue = 'No-show', data=combined)

plt.show()
sns.catplot(x='CallHour', y='Age', hue='No-show', data = combined, 

            order = range(24), kind='violin', aspect=2)

plt.xlabel("Hour of Call")

plt.ylabel("Age")

plt.title("Hour of Call for Making an Appointment vs Show or No-Show")

plt.show()
sns.catplot(x='CallHour', y='DaysToAppointment', hue='No-show', data = combined, 

            order = range(24), kind='violin', aspect=2)

plt.xlabel("Hour of Call")

plt.ylabel("Days To Appointment")

plt.title("Hour of Call for Making an Appointment vs Show or No-Show")

plt.show()
X = combined[['SMS_received', 'Appointment_w_d', 'DaysToAppointment',

              'CallHour', 'Gender', 'Age', 'Neighbourhood', 'Scholarship',

              'Hypertension', 'Diabetes', 'Alcoholism', 'Handcap']]

y = combined['No-show']
X.dtypes
X[['Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'Handcap']]=X[['Scholarship', 'Hypertension', 'Diabetes', 

                                                                           'Alcoholism', 'Handcap']].astype('int32')

X.dtypes
X['Gender'] = X['Gender'].apply(lambda x : 1 if x=='M' else 0)
X = pd.get_dummies(X, columns=['Neighbourhood', 'Appointment_w_d'])

X.drop(['Neighbourhood_JARDIM CAMBURI', 'Appointment_w_d_Monday'], axis=1, inplace=True)

X.head(2)
y = y.astype('int32')
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.metrics import roc_auc_score, roc_curve, auc

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size=0.7)
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(n_estimators=150, random_state=0)

RFC.fit(X_train, y_train)
print(confusion_matrix(y_test, RFC.predict(X_test)))
print(classification_report(y_test, RFC.predict(X_test)))
rfc_features = pd.DataFrame()

rfc_features['Feature'] = X_train.columns.tolist()

rfc_features['Importance'] = RFC.feature_importances_
sns.catplot(y='Feature', x='Importance', data=rfc_features, kind='bar', height=15, color=base_color)

plt.show()
cols_to_drop =[x for x in X_train.columns.tolist() if 'Neighb' in x]

X_train_rf = X_train.drop(cols_to_drop, axis=1)

X_test_rf = X_test.drop(cols_to_drop, axis=1)
RFC2 = RandomForestClassifier(n_estimators=150, random_state=0)

RFC2.fit(X_train_rf, y_train)
print(confusion_matrix(y_test, RFC2.predict(X_test_rf)))
print(classification_report(y_test, RFC2.predict(X_test_rf)))
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state=0, solver='saga')

LR.fit(X_train, y_train)
print(confusion_matrix(y_test, LR.predict(X_test)))
print(classification_report(y_test, LR.predict(X_test)))
from xgboost import XGBClassifier
XGB = XGBClassifier(n_estimators=1000)

XGB.fit(X_train, y_train)
print(confusion_matrix(y_test, XGB.predict(X_test)))
print(classification_report(y_test, XGB.predict(X_test)))
from catboost import CatBoostClassifier
CAT = CatBoostClassifier(iterations=2500)

CAT.fit(X_train,y_train)
print(confusion_matrix(y_test, CAT.predict(X_test)))
print(classification_report(y_test, CAT.predict(X_test)))
fpr_RFC, tpr_RFC, _ = roc_curve(y_test, RFC.predict_proba(X_test)[:,1])

roc_auc_RFC = auc(fpr_RFC, tpr_RFC)

fpr_LR, tpr_LR, _ = roc_curve(y_test, LR.predict_proba(X_test)[:,1])

roc_auc_LR = auc(fpr_LR, tpr_LR)

fpr_XGB, tpr_XGB, _ = roc_curve(y_test, XGB.predict_proba(X_test)[:,1])

roc_auc_XGB = auc(fpr_XGB, tpr_XGB)

fpr_CAT, tpr_CAT, _ = roc_curve(y_test, CAT.predict_proba(X_test)[:,1])

roc_auc_CAT = auc(fpr_CAT, tpr_CAT)
plt.figure(figsize=(8,8))

lw = 2

plt.plot(fpr_LR, tpr_LR, 

         lw=lw, label='ROC curve Logistic Regression (area = %0.2f)' % roc_auc_LR)

plt.plot(fpr_RFC, tpr_RFC, 

         lw=lw, label='ROC curve Random Forest (area = %0.2f)' % roc_auc_RFC)

plt.plot(fpr_XGB, tpr_XGB, 

         lw=lw, label='ROC curve XGBoost (area = %0.2f)' % roc_auc_XGB)

plt.plot(fpr_CAT, tpr_CAT, 

         lw=lw, label='ROC curve CATBoost (area = %0.2f)' % roc_auc_CAT)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Using All Features')

plt.legend(loc="lower right")

plt.show()