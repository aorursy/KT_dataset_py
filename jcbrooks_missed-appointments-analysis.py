import pandas as pd

import numpy as np



df = pd.read_csv('../input/KaggleV2-May-2016.csv')

print(len(df))

df.head(10)
import matplotlib.pyplot as plt



fig, ax = plt.subplots(figsize = (12,6))



#patientid_log = np.log10(df['PatientId'])



ax.hist(df['PatientId'], bins=100)

#ax.hist([df['CRSDepTime'], cancelled['CRSDepTime']], normed=1, bins=20, label=['All', 'Cancelled'])





#ax.set_xlim(0,2400)



ax.set_xlabel('Patient ID')

ax.set_title('Histogram of Patient IDs')



#plt.legend()

plt.show()
import matplotlib.pyplot as plt



fig, ax = plt.subplots(figsize = (12,6))



patientid_log = np.log10(df['PatientId'])



ax.hist(patientid_log, bins=100)

#ax.hist([df['CRSDepTime'], cancelled['CRSDepTime']], normed=1, bins=20, label=['All', 'Cancelled'])





#ax.set_xlim(0,2400)



ax.set_xlabel('log10(Patient ID)')

ax.set_title('Histogram of Patient IDs')



#plt.legend()

plt.show()
import matplotlib.pyplot as plt



fig, ax = plt.subplots(figsize = (12,6))



show = df[df['No-show']=='No']

noshow = df[df['No-show']=='Yes']



patientid_show_log = np.log10(show['PatientId'])

patientid_noshow_log = np.log10(noshow['PatientId'])



ax.hist([patientid_show_log,patientid_noshow_log], bins=50, normed=1,label=['Show', 'No-show'])

#ax.hist([df['CRSDepTime'], cancelled['CRSDepTime']], normed=1, bins=20, label=['All', 'Cancelled'])





#ax.set_xlim(0,2400)



ax.set_xlabel('log10(Patient ID)')

ax.set_title('Histogram of Patient IDs')



plt.legend()

plt.show()
import matplotlib.pyplot as plt



fig, ax = plt.subplots(figsize = (12,6))



ax.hist(df['AppointmentID'], bins=100)

#ax.hist([df['CRSDepTime'], cancelled['CRSDepTime']], normed=1, bins=20, label=['All', 'Cancelled'])





#ax.set_xlim(0,2400)



ax.set_xlabel('Patient ID')

ax.set_title('Histogram of Appointment IDs')



#plt.legend()

plt.show()
import matplotlib.pyplot as plt



fig, ax = plt.subplots(figsize = (12,6))



patientid_log = np.log10(df['AppointmentID'])



ax.hist(patientid_log, bins=100)

#ax.hist([df['CRSDepTime'], cancelled['CRSDepTime']], normed=1, bins=20, label=['All', 'Cancelled'])





#ax.set_xlim(0,2400)



ax.set_xlabel('log10(Patient ID)')

ax.set_title('Histogram of Appointment IDs')



#plt.legend()

plt.show()
import matplotlib.pyplot as plt



fig, ax1 = plt.subplots(figsize = (12,6))



ax1.scatter(df['PatientId'],df['AppointmentID'])



ax1.set_xlabel('Patient ID')

ax1.set_ylabel('Appointment ID')



plt.show()
import matplotlib.pyplot as plt

import datetime as dt

#import matplotlib.dates as mdates



df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])



fig, ax1 = plt.subplots(figsize = (12,6))



ax1.scatter(df['ScheduledDay'].tolist(),df['AppointmentID'])



ax1.set_xlabel('Scheduled Datetime')

ax1.set_ylabel('Appointment ID')



plt.show()
import matplotlib.pyplot as plt

import datetime as dt



df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])



show = df[df['No-show']=='No']

noshow = df[df['No-show']=='Yes']



show.dtypes



fig, ax1 = plt.subplots(figsize = (12,6))



show_dates = np.asarray(show['ScheduledDay'].tolist())

noshow_dates = np.asarray(noshow['ScheduledDay'].tolist())

print(type(show_dates))



ax1.hist([show_dates, noshow_dates], bins=50, normed=1, label=['Show', 'No-show'])

# ax1.hist(show['ScheduledDay'].tolist(), bins=50, normed=1)

# ax1.hist(noshow['ScheduledDay'].tolist(), bins=50, normed=1,alpha=0.5)



ax1.set_xlabel('Scheduled Datetime')

ax1.set_title('Histogram of Scheduled Datetime')



plt.legend()

plt.show()
import matplotlib.pyplot as plt

import datetime as dt



df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])



fig, ax1 = plt.subplots(figsize = (12,6))



ax1.scatter(df['ScheduledDay'].dt.hour,df['AppointmentID'])



ax1.set_xlabel('Appointment hour')

ax1.set_ylabel('Appointment ID')



plt.show()
import matplotlib.pyplot as plt

import datetime as dt



df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])



fig, ax1 = plt.subplots(figsize = (12,6))



ax1.scatter(df['ScheduledDay'].dt.day,df['AppointmentID'])



ax1.set_xlabel('Appointment Day of the Month')

ax1.set_ylabel('Appointment ID')



plt.show()
import matplotlib.pyplot as plt

import datetime as dt



df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])



fig, ax1 = plt.subplots(figsize = (12,6))



ax1.scatter(df['ScheduledDay'].tolist(),df['PatientId'])



ax1.set_xlabel('Scheduled Datetime')

ax1.set_ylabel('Patient ID')



plt.show()
import matplotlib.pyplot as plt

import datetime as dt

import seaborn as sns



df1 = df.sample(n=5000, random_state = 47)



df1['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])

df1['DayofWeek'] = df['ScheduledDay'].dt.dayofweek



patientid_log = np.log10(df1['PatientId'])



fig, ax1 = plt.subplots(figsize = (12,6))



#ax1.scatter(df['ScheduledDay'].dt.month,df['PatientId'])

ax1 = sns.swarmplot(df1['DayofWeek'],patientid_log)



ax1.set_xlabel('Appointment day')

ax1.set_ylabel('log10(Patient ID)')



plt.show()
df['No-show'] = df['No-show'].astype('category').cat.codes

#print(df['No-show'].head(10))

patient_visits = df['PatientId'].groupby(df['PatientId']).count()

top_patients = patient_visits.sort_values(ascending=False).head(10)

top_patients.index.values

for x in top_patients.index.values:

    #print("{:f}".format(x))

    print(int(x))

    

patient_1 = df[df['PatientId']==822145925426128]

patient_2 = df[df['PatientId']==99637671331]

patient_3 = df[df['PatientId']==26886125921145]

patient_4 = df[df['PatientId']==33534783483176]

patient_5 = df[df['PatientId']==75797461494159]

patient_6 = df[df['PatientId']==258424392677]

patient_7 = df[df['PatientId']==871374938638855]

patient_8 = df[df['PatientId']==6264198675331]

patient_9 = df[df['PatientId']==66844879846766]

patient_10 = df[df['PatientId']==872278549442]
x1 = patient_1['No-show'].sum()

print(x1)

print(len(patient_1))



y = [patient_1['No-show'].sum()/len(patient_1),

     patient_2['No-show'].sum()/len(patient_2),

     patient_3['No-show'].sum()/len(patient_3),

     patient_4['No-show'].sum()/len(patient_4),

     patient_5['No-show'].sum()/len(patient_5),

     patient_6['No-show'].sum()/len(patient_6),

     patient_7['No-show'].sum()/len(patient_7),

     patient_8['No-show'].sum()/len(patient_8),

     patient_9['No-show'].sum()/len(patient_9),

     patient_10['No-show'].sum()/len(patient_10)]



x = np.arange(1,11)

print(x)



import matplotlib.pyplot as plt



fig, ax = plt.subplots(figsize = (12,6))



ax.bar(x,y)



ax.set_xlabel('top patients')

ax.set_title('Fraction of Missed Appointments for Top Patients')



#plt.legend()

plt.show()
import datetime as dt



df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])

df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

df['Timedelta'] = df['AppointmentDay'] - df['ScheduledDay']

df['Timedelta'] = df['Timedelta'].dt.days



# df.dtypes



df['Month'] = df['ScheduledDay'].dt.month

df['DayofMonth'] = df['ScheduledDay'].dt.day

df['DayofWeek'] = df['ScheduledDay'].dt.dayofweek

df['Hour'] = df['ScheduledDay'].dt.hour



df['Gender'] = df['Gender'].astype('category').cat.codes

df['Neighbourhood'] = df['Neighbourhood'].astype('category').cat.codes

df['No-show'] = df['No-show'].astype('category').cat.codes



df.dtypes
from sklearn.model_selection import train_test_split



# X = df[['PatientId', 'AppointmentID', 'Month', 'DayofMonth', 'DayofWeek', 'Hour', 

#         'Age', 'Neighbourhood', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap',

#         'SMS_received']]

X = df[['PatientId', 'AppointmentID', 'Timedelta', 'DayofMonth', 'DayofWeek', 'Hour', 

        'Age', 'Neighbourhood', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap',

        'SMS_received']]

y = df['No-show']



# df1 = df.sample(n=10000, random_state = 47)

# X = df1[['PatientId', 'AppointmentID', 'DayofMonth', 'DayofWeek', 'Hour', 

#         'Age', 'Neighbourhood', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap',

#         'SMS_received']]

# y = df1['No-show']



print(sum(y)/len(y))



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 87)



from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

# we must apply the scaling to the test set that we computed for the training set

X_test_scaled = scaler.transform(X_test)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



clf = RandomForestClassifier(n_estimators=100, max_depth = 6, random_state=47).fit(X_train, y_train)



y_predicted = clf.predict(X_test)



print('Recall: {:.3f}'.format(recall_score(y_test, y_predicted)))

print('Precision: {:.3f}'.format(precision_score(y_test, y_predicted)))

print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_predicted)))

print('F1: {:.3f}'.format(f1_score(y_test, y_predicted)))

confusion = confusion_matrix(y_test, y_predicted)

print(confusion)

print('Feature importances: {}'.format(clf.feature_importances_))
from sklearn.ensemble import GradientBoostingClassifier



clf = GradientBoostingClassifier(n_estimators=1000, learning_rate = 0.5, 

                                 max_depth = 6, random_state=37).fit(X_train, y_train)



y_predicted = clf.predict(X_test)

confusion = confusion_matrix(y_test, y_predicted)



print('Recall: {:.3f}'.format(recall_score(y_test, y_predicted)))

print('Precision: {:.3f}'.format(precision_score(y_test, y_predicted)))

print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_predicted)))

print('F1: {:.3f}'.format(f1_score(y_test, y_predicted)))

confusion = confusion_matrix(y_test, y_predicted)

print(confusion)

print('Feature importances: {}'.format(clf.feature_importances_))



y_gbc = y_predicted
from sklearn.ensemble import AdaBoostClassifier



clf = AdaBoostClassifier(n_estimators=1000, learning_rate = 1.5, 

                                 random_state=37).fit(X_train, y_train)



y_predicted = clf.predict(X_test)

confusion = confusion_matrix(y_test, y_predicted)



print('Recall: {:.3f}'.format(recall_score(y_test, y_predicted)))

print('Precision: {:.3f}'.format(precision_score(y_test, y_predicted)))

print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_predicted)))

print('F1: {:.3f}'.format(f1_score(y_test, y_predicted)))

confusion = confusion_matrix(y_test, y_predicted)

print(confusion)

print('Feature importances: {}'.format(clf.feature_importances_))
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV



np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})



lr = LogisticRegression()

#grid_values = {'penalty': ['l1', 'l2'], 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}

grid_values = {'penalty': ['l1', 'l2'], 'C': [10, 100, 1000, 10000, 100000, 1e6]}

grid_lr = GridSearchCV(lr, param_grid = grid_values, scoring = 'recall').fit(X_train_scaled, y_train)

print(grid_lr.cv_results_['mean_test_score'].reshape(6,2))
from sklearn.neural_network import MLPClassifier



nnclf = MLPClassifier(hidden_layer_sizes = [50,50], solver='lbfgs', alpha=0.3, activation='relu',

                     max_iter = 100, random_state = 47).fit(X_train_scaled, y_train)



y_predicted = nnclf.predict(X_test_scaled)

confusion = confusion_matrix(y_test, y_predicted)



print('Recall: {:.3f}'.format(recall_score(y_test, y_predicted)))

print('Precision: {:.3f}'.format(precision_score(y_test, y_predicted)))

print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_predicted)))

print('F1: {:.3f}'.format(f1_score(y_test, y_predicted)))

confusion = confusion_matrix(y_test, y_predicted)

print(confusion)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 1)

knn.fit(X_train_scaled, y_train)



y_predicted = knn.predict(X_test_scaled)



confusion = confusion_matrix(y_test, y_predicted)



print('Recall: {:.3f}'.format(recall_score(y_test, y_predicted)))

print('Precision: {:.3f}'.format(precision_score(y_test, y_predicted)))

print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_predicted)))

print('F1: {:.3f}'.format(f1_score(y_test, y_predicted)))

confusion = confusion_matrix(y_test, y_predicted)

print(confusion)



y_knn = y_predicted



y_scores = knn.predict_proba(X_test_scaled)

y_score_list = list(zip(y_test[0:20], y_scores[0:20]))
from sklearn.naive_bayes import GaussianNB



nbclf = GaussianNB().fit(X_train_scaled, y_train)



y_predicted = nbclf.predict(X_test_scaled)



confusion = confusion_matrix(y_test, y_predicted)



print('Recall: {:.3f}'.format(recall_score(y_test, y_predicted)))

print('Precision: {:.3f}'.format(precision_score(y_test, y_predicted)))

print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_predicted)))

print('F1: {:.3f}'.format(f1_score(y_test, y_predicted)))

confusion = confusion_matrix(y_test, y_predicted)

print(confusion)
# from sklearn.gaussian_process import GaussianProcessClassifier



# gpc = GaussianProcessClassifier(max_iter_predict=100, random_state=56).fit(X_train_scaled, y_train)



# y_predicted = gpc.predict(X_test_scaled)



# confusion = confusion_matrix(y_test, y_predicted)



# print('Recall: {:.3f}'.format(recall_score(y_test, y_predicted)))

# print('Precision: {:.3f}'.format(precision_score(y_test, y_predicted)))

# print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_predicted)))

# print('F1: {:.3f}'.format(f1_score(y_test, y_predicted)))

# confusion = confusion_matrix(y_test, y_predicted)

# print(confusion)
y_votes = 0.5*(y_gbc + y_knn)



threshold = 0.4



print(y_votes)



low_values_flags = y_votes < threshold

y_votes[low_values_flags] = 0.0



high_values_flags = y_votes >= threshold

y_votes[high_values_flags] = 1.0



y_predicted = y_votes.astype(int)



print('Recall: {:.3f}'.format(recall_score(y_test, y_predicted)))

print('Precision: {:.3f}'.format(precision_score(y_test, y_predicted)))

print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_predicted)))

print('F1: {:.3f}'.format(f1_score(y_test, y_predicted)))

confusion = confusion_matrix(y_test, y_predicted)

print(confusion)
from sklearn.model_selection import train_test_split

#df1 = df.sample(n=10000, random_state = 47)

X = df[['PatientId', 'AppointmentID', 'Timedelta', 'DayofMonth', 'DayofWeek', 'Hour', 

        'Age', 'Neighbourhood',

        'SMS_received']]

y = df['No-show']



print(sum(y)/len(y))

print(len(y))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state = 87)

print(len(y_train))



from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

# we must apply the scaling to the test set that we computed for the training set

X_test_scaled = scaler.transform(X_test)
# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

# from sklearn.svm import SVC

# from sklearn.model_selection import GridSearchCV



# clf = SVC(kernel='rbf')

# #grid_values = {'gamma': [0.001, 0.01, 0.05, 0.1, 1, 10, 100], 'C': [0.01, 0.1, 1, 10, 100]}

# #grid_values = {'gamma': [1, 3, 10, 30, 100, 300, 1000], 'C': [10, 100, 1000, 3000, 10000]}

# grid_values = {'gamma': [1, 1.5, 2, 3, 4, 6, 9], 'C': [1000, 10000, 30000, 100000, 300000]}



# grid_clf = GridSearchCV(clf, param_grid = grid_values, scoring = 'recall')

# grid_clf.fit(X_train_scaled, y_train)

# grid_clf.cv_results_['mean_test_score'].reshape(5,7)
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})



#svm = SVC(kernel='rbf', C=7000, gamma=0.6).fit(X_train_scaled, y_train)

svm = SVC(kernel='rbf', C=10000, gamma=2, random_state=37).fit(X_train_scaled, y_train)

y_pred = svm.predict(X_test_scaled)



print('Recall: {:.3f}'.format(recall_score(y_test, y_pred)))

print('Precision: {:.3f}'.format(precision_score(y_test, y_pred)))

print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)))

print('F1: {:.3f}'.format(f1_score(y_test, y_pred)))

confusion = confusion_matrix(y_test, y_pred)

print(confusion)



y_scores = svm.decision_function(X_test_scaled)

y_score_list = list(zip(y_test[0:20], y_scores[0:20]))
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, roc_curve, auc



precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

closest_zero = np.argmin(np.abs(thresholds))

closest_zero_p = precision[closest_zero]

closest_zero_r = recall[closest_zero]



plt.figure(figsize=(8,8))

plt.plot(precision, recall, label='Precision-Recall Curve')

plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)

plt.xlabel('Precision', fontsize=16)

plt.ylabel('Recall', fontsize=16)

plt.axes().set_aspect('equal')

plt.show()
fpr, tpr, _ = roc_curve(y_test, y_scores)

roc_auc = auc(fpr, tpr)

print('AUC: {:.3f}'.format(roc_auc))



plt.figure(figsize=(8,8))

plt.xlim([-0.01, 1.00])

plt.ylim([-0.01, 1.01])

plt.plot(fpr, tpr, lw=3, label='SVC ROC curve (area = {:0.2f})'.format(roc_auc))

plt.xlabel('False Positive Rate', fontsize=16)

plt.ylabel('True Positive Rate', fontsize=16)

plt.title('ROC curve', fontsize=16)

plt.legend(loc='lower right', fontsize=13)

plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')

plt.axes().set_aspect('equal')

plt.show()