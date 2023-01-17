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
import numpy as np

import pandas as pds

import matplotlib.pyplot as plt

from matplotlib import pylab

import seaborn as sns







sns.set_style("whitegrid")
noshow = pd.read_csv(r'/kaggle/input/noshowappointments/KaggleV2-May-2016.csv')

noshow.head(5)
noshow.PatientId.nunique()
print("Total number appointments are :{}\nTotal number of patient registered are :{}".format(noshow.AppointmentID.count(),noshow.PatientId.nunique() ))

print("Total number of patients repeated in percentage: {}".format(((noshow.PatientId.nunique() - noshow.AppointmentID.count()) / noshow.AppointmentID.count() ) * 100))
sns.countplot('No-show', data =noshow )
noshow.info()
noshow['Gender'][:]
from sklearn.preprocessing import LabelEncoder



labelencoder_noshow = LabelEncoder()

noshow['Gender'] = labelencoder_noshow.fit_transform(noshow['Gender'][:])

noshow['No-show'] = labelencoder_noshow.fit_transform(noshow['No-show'][:])
noshow.head()
type(noshow['Gender'])
plt.hist(noshow['Age'], bins = 10);
noshow['Age'].min()

noshow['Age'].max()

#(noshow['Age']>100).sum()
len(noshow[(noshow.Age > 100)])

len(noshow[(noshow.Age < 0)])

print(f"Youngest patient's age is {noshow['Age'].min()}, and oldest's age is {noshow['Age'].max()} \nAnd total {len(noshow[(noshow.Age < 0)])} is less than 0 years of age and {len(noshow[(noshow.Age > 100)])} more are then 100 years of age. \nSo we will eliminate these outliers")
noshow = noshow[(noshow.Age > 0 ) & (noshow.Age <= 90)]
plt.hist(noshow['Age'], bins = 9);
bin_ranges = [-1, 2, 8, 16, 18, 25, 40, 50, 60, 75]

bin_names = ["Baby", "Children", "Teenager", 'Young', 'Young-Adult', 'Adult', 'Adult-II', 'Senior', 'Old']



noshow['age_bin'] = pd.cut((noshow['Age']),bins = bin_ranges, labels= bin_names)
plt.figure(figsize = (10,5))

sns.countplot(noshow['age_bin'], hue = noshow['Gender']);
plt.figure(figsize = (10,5))

sns.countplot(noshow['age_bin'], hue = noshow['No-show']);
##how to display percentage change
pd.to_datetime(noshow.ScheduledDay)
pd.to_datetime(noshow.ScheduledDay).dt.date

pd.to_datetime(noshow.ScheduledDay).dt.day
pd.to_datetime(noshow.ScheduledDay).dt.weekday_name.head()
dt_scheduledDay=pd.to_datetime(noshow.ScheduledDay).dt.date

dt_appointmentDay = pd.to_datetime(noshow.AppointmentDay).dt.date



noshow['Days_delta'] = (dt_appointmentDay - dt_scheduledDay).dt.days

noshow.head()

plt.hist(noshow['Days_delta'], bins = 20);
noshow['Days_delta'].min()
noshow['Days_delta'].max()
len(noshow[(noshow.Days_delta < 0)])

len(noshow[(noshow.Days_delta >= 70)])

print(f'noshow delta days less then 0 are: {len(noshow[(noshow.Days_delta < 0)])}, \nnoshow delta days more then 70 are: {len(noshow[(noshow.Days_delta >= 70)])}')
noshow = noshow[(noshow.Days_delta >= 0) & (noshow.Days_delta <= 70)]
len(noshow['Days_delta'])
noshow['AppointmentDay'] = pd.to_datetime(noshow['AppointmentDay'])

noshow['No_show_weekday'] = noshow['AppointmentDay'].dt.dayofweek



sns.violinplot(y='No-show', x='No_show_weekday', data=noshow)
plt.figure(figsize = (10,5))

sns.boxplot(x='SMS_received', y = 'Age', data = noshow, hue = 'No-show')
plt.figure(figsize = (10,5))

sns.boxplot(x='No-show', y = 'Age', data = noshow, hue = 'Scholarship')
noshow.head(0)
# Exploring features and content to spot identifiers and to separate categorical from numerical columns for modelling later

print ("PatientIds: ", len(noshow.PatientId.unique()))

print ("\nAppointmentIds: ", len(noshow.AppointmentID.unique()))

print ("\nNo. of appointment days: ", len(noshow.AppointmentDay.unique()))

print ("\nGender: ", noshow.Gender.unique())

print ("\nAge: ", sorted(noshow.Age.unique()), "\n Unique values: ", len(noshow.Age.unique()))

print ("\nSMS_received: ", noshow.SMS_received.unique())

print ("\nScholarship: ", noshow.Scholarship.unique())

print ("\nHypertension: ", noshow.Hipertension.unique())

print ("\nDiabetes: ", noshow.Diabetes.unique())

print ("\nAlcoholism: ", noshow.Alcoholism.unique())

print ("\nHandicap: ", noshow.Handcap.unique())

print ("\nNo-show: ", noshow["No-show"].unique())

print ("\nDays delta: ", sorted(noshow.Days_delta.unique()))

print ("\nBrazilian neighbourhoods: ", len(noshow.Neighbourhood.unique()))
noshow=pd.get_dummies(noshow, columns = ['Handcap', 'No_show_weekday'])
noshow.head()
X = noshow.drop(['PatientId','AppointmentID','ScheduledDay','AppointmentDay', 'Neighbourhood','age_bin','No-show'], axis = 1)

y = noshow['No-show']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)

sgd_clf.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)
y_pred = logmodel.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test,y_pred))

print(confusion_matrix(y_test, y_pred))