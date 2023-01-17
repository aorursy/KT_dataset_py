import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns 

from sklearn.linear_model import LogisticRegression,LogisticRegressionCV

from sklearn.model_selection import KFold,cross_val_score

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,f1_score,precision_recall_curve,recall_score,precision_score,auc,roc_curve

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures, StandardScaler







app = pd.read_csv('../input/KaggleV2-May-2016.csv',encoding='latin1')

app.head(3)
print(app.shape)
print("Number of Patients: %d" % len(app['PatientId'].unique()))

print("Number of Appointments: %d" % app['PatientId'].count())

print("Attendance mean %f" % (app['PatientId'].count()/ len(app['PatientId'].unique())))

print ("Max of attendance group by patientsnes es: %d" % app.groupby('PatientId')[['AppointmentID']].count().sort_values('AppointmentID',ascending= False).max())
fig, ax = plt.subplots()

a = app.groupby('PatientId')[['AppointmentID']].count()

ax.hist(a['AppointmentID'].tolist(),bins=100)

ax.set(xlim=[0,80], ylabel='Appointment Quantity',

       title='Attendance Distribution by patient')

plt.ylim((0,400))

plt.show()
app['No-show'].value_counts()
app.rename(columns={'No-show': 'Show'},inplace=True)
app['Show'] = app.Show.transform(lambda x: 1 if (x == 'No') else 0)
app.count()
app.describe()
app = app[app.Age>0]
app_day = pd.to_datetime(app.AppointmentDay)

sch_day =  pd.to_datetime(app.ScheduledDay)

wait = app_day -sch_day

app['waiting'] = pd.DataFrame(wait)

app = app[app.waiting >= '0']

app['weekday'] = app_day.dt.weekday
app['waiting'] =(app.waiting/np.timedelta64(1, 'D')).astype(int)
app['cant_asistencias'] = app.groupby('PatientId')[['Show']].transform(sum)

app['cant_turnos'] = app.groupby('PatientId')[['Show']].transform('count')



app['prom_asistencia'] = app.cant_asistencias / app.cant_turnos
app.groupby('PatientId')[['Show']].apply(sum).count()
sns.barplot(x ='weekday',y='waiting',hue='Show', data=app)

plt.show()
sns.countplot(x='Gender',hue='Show', data=app)

plt.show()
sns.boxplot(x="waiting", y="SMS_received", hue="Show", data=app,orient='h')

plt.xlim(0,75)

plt.show()


sns.countplot(x='Age',hue='Show', data=app)



plt.show()
sns.distplot(app.waiting[app.Show == 1],color='Red')

sns.distplot(app.waiting[app.Show == 0],color='Green')



plt.xlim(0,100)





plt.show()
g = sns.FacetGrid(app , hue='Show',size=7)

g.map(plt.scatter,'waiting','Age', alpha = .7)

g.add_legend();

plt.show()
sns.heatmap(app.corr())

plt.show()
app.groupby('Show').count()

app['Gender'] = pd.get_dummies(app.Gender,drop_first=True)
app = app.sort_values('AppointmentID', ascending=True)
kf = KFold(n_splits=5, shuffle=True)



X = app[['SMS_received','waiting','Age','Gender','cant_turnos']]

y = app.Show

a

X = StandardScaler().fit_transform(X) 



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



X_train = app.loc[:50000,['SMS_received','waiting','Age','Gender','cant_turnos','prom_asistencia']]



X_test= app.loc[50001:,['SMS_received','waiting','Age','Gender','cant_turnos','prom_asistencia']]



y_train= app.loc[:50000,'Show']



y_test= app.loc[50001:,'Show']
print(X_train.shape , y_train.shape , X_test.shape , y_test.shape)
lr = LogisticRegression(C=1e10,class_weight='balanced')



lr.fit(X_train,y_train)
lr_predicts = lr.predict(X_test)

lr_log_predicts = lr.predict_proba(X_test)



preds_left = lr_log_predicts[:,1]

# 15 bins

plt.hist(preds_left, bins=15)



# x-axis de 0 a 1

plt.xlim(0,1)

plt.title('Histogram of probabilities')

plt.xlabel('Probability of assist to an appointment')

plt.ylabel('Frecuency')

plt.show()

y_pred = (preds_left > 0.6)

accuracy_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)

cm
print('Amount of correct positives classified: ',recall_score(y_test,y_pred))

print('Precision: ',precision_score(y_test,y_pred))

print('F1 Score: ',f1_score(y_test,y_pred))







print (classification_report(y_test,y_pred))
sns.heatmap(cm,annot=True)

plt.show()
fpr,tpr,ths = roc_curve(y_test, lr_log_predicts[:,1])

df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))

plt.axis([0, 1.01, 0, 1.01])

plt.xlabel('1 - Specificty')

plt.ylabel('TPR / Sensitivity')

plt.title('ROC Curve')

plt.plot(df['fpr'],df['tpr'])

plt.plot(np.arange(0,1, step =0.01), np.arange(0,1, step =0.01))

plt.show() 
from sklearn import svm



clf = svm.SVC(probability=True)



clf.fit(X_train,y_train)
svm_predicts = clf.predict(X_test)

svm_log_predicts = clf.predict_proba(X_test)



preds_left = svm_log_predicts[:,1]

# 15 bins

plt.hist(preds_left, bins=15)



# x-axis de 0 a 1

plt.xlim(0,1)

plt.title('Histogram of probaility')

plt.xlabel('Probability to assist an appointment')

plt.ylabel('Frecuency')

plt.show()
y_pred = (preds_left > 0.85)

accuracy_score(y_test,y_pred)
accuracy_score(y_test,y_pred)
print('Amoun of correct positives classified: ',recall_score(y_test,y_pred))

print('Precision: ',precision_score(y_test,y_pred))

print('F1 Score: ',f1_score(y_test,y_pred))







print (classification_report(y_test,y_pred))
cm_svm = confusion_matrix(y_test,y_pred)

cm_svm
fpr,tpr,ths = roc_curve(y_test, lr_log_predicts[:,1])

df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))

plt.axis([0, 1.01, 0, 1.01])

plt.xlabel('1 - Specificty')

plt.ylabel('TPR / Sensitivity')

plt.title('ROC Curve')

plt.plot(df['fpr'],df['tpr'])

plt.plot(np.arange(0,1, step =0.01), np.arange(0,1, step =0.01))

plt.show() 
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
k_range = list(range(1, 31))

knn = KNeighborsClassifier()

param_grid = dict(n_neighbors=k_range)

grid = RandomizedSearchCV(knn, param_grid, cv=10, scoring='accuracy')

grid.fit(X_train, y_train)

grid.best_params_['n_neighbors']
best_model = KNeighborsClassifier(n_neighbors= grid.best_params_['n_neighbors'])

best_model.fit(X_train,y_train)

KNN_predicts = best_model.predict(X_test)
accuracy_score(y_test,KNN_predicts)
cm_KNN = confusion_matrix(y_test,KNN_predicts)

cm_KNN
print(classification_report(y_test,KNN_predicts))
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import log_loss
nb = GaussianNB()

nb.fit(X_train,y_train)
nb_predicts = nb.predict(X_test)

nb_log_predicts = nb.predict_proba(X_test)



preds_left = nb_log_predicts[:,1]

# 15 bins

plt.hist(preds_left, bins=15)



# x-axis de 0 a 1

plt.xlim(0,1)

plt.title('Histogram of probabilities')

plt.xlabel('Probability of assist an appointment')

plt.ylabel('Frecuency')

plt.show()
accuracy_score(y_test,y_pred)
cm_nb = confusion_matrix(y_test,y_pred)

print(cm_nb)

sns.heatmap(cm_nb,annot=True)
print(classification_report(y_test,y_pred))
fpr,tpr,ths = roc_curve(y_test, lr_log_predicts[:,1])

df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))

plt.axis([0, 1.01, 0, 1.01])

plt.xlabel('1 - Specificty')

plt.ylabel('TPR / Sensitivity')

plt.title('ROC Curve')

plt.plot(df['fpr'],df['tpr'])

plt.plot(np.arange(0,1, step =0.01), np.arange(0,1, step =0.01))

plt.show() 
from sklearn.ensemble import RandomForestClassifier
n_params = { 'n_estimators':[3,5,10,50],

              'criterion':['gini','entropy'],

              'max_depth': [None,3,5],

               'min_samples_split':[2,5],

               'class_weight':['balanced',None]}



gsrf = GridSearchCV(RandomForestClassifier(),n_params,cv= KFold(n_splits=3,shuffle=True))



gsrf.fit(X_train,y_train)



rf_predicts = gsrf.predict(X_test)



print (classification_report(y_test,rf_predicts))

print(confusion_matrix(y_test,rf_predicts))

print (accuracy_score(y_test,rf_predicts))