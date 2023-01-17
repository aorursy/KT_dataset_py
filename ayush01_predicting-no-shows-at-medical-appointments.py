import pylab

import calendar

import numpy as np

import pandas as pd

import seaborn as sn

from scipy import stats

#import missingno as msno

from datetime import datetime

import matplotlib.pyplot as plt

import warnings

pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline
daily_Data=pd.read_csv("../input/KaggleV2-May-2016.csv")
daily_Data.head()
(daily_Data["Gender"] == "F").value_counts()
print('Age:',sorted(daily_Data.Age.unique()))

print('Gender:',daily_Data.Gender.unique())

print("Neighbourhood",daily_Data.Neighbourhood.unique())

print('Scholarship:',daily_Data.Scholarship.unique())

print('Hipertension:',daily_Data.Hipertension.unique())

print('Diabetes:',daily_Data.Diabetes.unique())

print('Alcoholism:',daily_Data.Alcoholism.unique())

print('Handcap:',daily_Data.Handcap.unique())

print('SMS_received:',daily_Data.SMS_received.unique())

#print('No-show:',daily_Data.No-show.unique())
f, ax = plt.subplots(figsize=(15, 10))

sn.countplot(y="Handcap", data=daily_Data, color="c");
from sklearn import preprocessing

#Label Encoding of Gender, Neighbourhood, No-show handcap

le = preprocessing.LabelEncoder()



le.fit(daily_Data["Gender"])

daily_Data["Gender"]=le.transform(daily_Data["Gender"])



#le.fit(daily_Data["Neighbourhood"])

#daily_Data["Neighbourhood"]=le.transform(daily_Data["Neighbourhood"])



le.fit(daily_Data["No-show"])

daily_Data["No-show"]=le.transform(daily_Data["No-show"])



le.fit(daily_Data["Neighbourhood"])

daily_Data["Neighbourhood"]=le.transform(daily_Data["Neighbourhood"])





daily_Data.head()
print('Age:',sorted(daily_Data.Age.unique()))

print('Gender:',daily_Data.Gender.unique())

print("Neighbourhood",daily_Data.Neighbourhood.unique())

print('Scholarship:',daily_Data.Scholarship.unique())

print('Hipertension:',daily_Data.Hipertension.unique())

print('Diabetes:',daily_Data.Diabetes.unique())

print('Alcoholism:',daily_Data.Alcoholism.unique())

print('Handcap:',daily_Data.Handcap.unique())

print('SMS_received:',daily_Data.SMS_received.unique())
daily_Data.head()
daily_Data["AppointmentDay"].head()
#appointment date

daily_Data["a"] = daily_Data.AppointmentDay.apply(lambda x : x.split("T")[0])

daily_Data["a"].head(2)
#schedule date

daily_Data["s"] = daily_Data.ScheduledDay.apply(lambda x : x.split("T")[0])
daily_Data.dtypes
daily_Data["weekday"] = daily_Data.a.apply(lambda dateString : calendar.day_name[datetime.strptime(dateString,"%Y-%m-%d").weekday()])
le.fit(daily_Data["weekday"])

daily_Data["weekday"]=le.transform(daily_Data["weekday"])
daily_Data.head()
daily_Data.Age.unique()
daily_Data['a'] = pd.to_datetime(daily_Data['a'])

daily_Data['s'] = pd.to_datetime(daily_Data['s'])



daily_Data['Days'] = (daily_Data.a - daily_Data.s)/ np.timedelta64(1, 'D')  # if already datetime64 you don't need to use to_datetime





print(daily_Data.head())
print(sorted(daily_Data.Days.unique()))
daily_Data.drop(daily_Data[daily_Data.Days < 0].index, inplace=True)
daily_Data.drop(daily_Data[daily_Data.Age < 0].index, inplace=True)
daily_Data=daily_Data.drop(["a","s","AppointmentDay","ScheduledDay","PatientId","AppointmentID"],axis=1)
daily_Data.reset_index()

daily_Data.index.name="Index"

daily_Data.head()
print('Age:',sorted(daily_Data.Age.unique()))

print('Gender:',daily_Data.Gender.unique())

print("Neighbourhood",sorted(daily_Data.Neighbourhood.unique()))

print('Scholarship:',daily_Data.Scholarship.unique())

print('Hipertension:',daily_Data.Hipertension.unique())

print('Diabetes:',daily_Data.Diabetes.unique())

print('Alcoholism:',daily_Data.Alcoholism.unique())

print('Handcap:',daily_Data.Handcap.unique())

print('SMS_received:',daily_Data.SMS_received.unique())

print('weekday:',daily_Data.weekday.unique())
daily_Data.shape
"""

from sklearn.preprocessing import OneHotEncoder

enc=OneHotEncoder(sparse=False)

columns=['Neighbourhood']

for col in columns:

       enc.fit(daily_Data)

       temp = enc.transform(daily_Data[col])

       temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in daily_Data[col].value_counts().index])

       temp=temp.set_index(daily_Data.index.values)

       daily_Data1=pd.concat([daily_Data,temp],axis=1)

"""
daily_Data.Handcap.unique()

print(daily_Data.head())

print('Age:',sorted(daily_Data.Age.unique()))

print('Gender:',daily_Data.Gender.unique())

print("Neighbourhood",daily_Data.Neighbourhood.unique())

print('Scholarship:',daily_Data.Scholarship.unique())

print('Hipertension:',daily_Data.Hipertension.unique())

print('Diabetes:',daily_Data.Diabetes.unique())

print('Alcoholism:',daily_Data.Alcoholism.unique())

print('Handcap:',daily_Data.Handcap.unique())

print('SMS_received:',daily_Data.SMS_received.unique())
a=(daily_Data["Gender"] == 1).value_counts()
a
daily_Data["No-show"][daily_Data["Gender"] == 0].value_counts()
daily_Data["No-show"][daily_Data["Gender"] == 1].value_counts()
figure = plt.figure(figsize=(15,8))

plt.hist([daily_Data[daily_Data['No-show']==1]['Gender'], daily_Data[daily_Data['No-show']==0]['Gender']], stacked=True, color = ['g','r'],label = ['show','no-show'])

plt.xlabel('Gender')

plt.ylabel('Number of patients')

plt.legend()
figure = plt.figure(figsize=(15,8))

sn.distplot(daily_Data[daily_Data['No-show'] == 0]['Gender'],color='b')

sn.distplot(daily_Data[daily_Data['No-show'] == 1]['Gender'],color='r')

daily_Data["Days"].describe()
print(sorted(daily_Data.Days.unique()))
daily_Data.head()
labels=daily_Data.pop("No-show")
labels.shape
from imblearn.over_sampling import SMOTE



df_new,lab_new=SMOTE(random_state=3).fit_sample(daily_Data,labels)

df_new=pd.DataFrame(df_new)

lab_new=pd.DataFrame(lab_new)
lab_new[0].value_counts()
import numpy as np

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_new, lab_new, test_size=0.30, random_state=42)
from sklearn.ensemble import RandomForestClassifier

#clf = RandomForestClassifier(max_depth=2, random_state=42)

clf = RandomForestClassifier(n_estimators=200,max_depth=100,min_samples_split=50,min_samples_leaf=100)

clf=clf.fit(X_train, y_train)

feat=(clf.feature_importances_)

pred=clf.predict(X_test)

print(pred)



from sklearn.metrics import accuracy_score, f1_score

accuracy = accuracy_score(y_test,pred)

print(accuracy)

print(feat)
plt.hist(feat,bins=30,)
from sklearn.metrics import accuracy_score, f1_score

accuracy = accuracy_score(y_test,pred)

accuracy
f1=f1_score(y_test,pred)

print(f1)
f0=f1_score(y_test,pred, average='macro')  

f1=f1_score(y_test,pred, average='micro')  

f2=f1_score(y_test,pred, average='weighted')  

f3=f1_score(y_test,pred, average=None)

f4=f1_score(y_test,pred, average='binary')

print(f0,f1,f2,f3,f4)
from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, clf.predict(X_test)))
print(classification_report(y_test, clf.predict(X_test)))