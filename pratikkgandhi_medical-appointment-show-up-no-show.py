import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style("whitegrid")
# Reading the file:

data = pd.read_csv("../input/No-show-Issue-Comma-300k.csv")
# Overall view of data:

data.info()
# Looking at the first few rows:

data.head()
data = data.rename(columns={'ApointmentData':'AppointmentDate','Alcoolism':'Alcholism','HiperTension':'HyperTension','Handcap':'Handicap'})

data.columns
cat_var = data.dtypes[data.dtypes == 'object'].index

data[cat_var].describe()
data.describe()
#data['Status'] = data['Status'].astype('category')

data['AppointmentRegistration'] = pd.to_datetime(data['AppointmentRegistration'])

data['AppointmentDate'] = pd.to_datetime(data['AppointmentDate'])
# Converting day of the week to numeric form:

def daytoNumber(day):

    if day == 'Monday': 

        return 1

    if day == 'Tuesday': 

        return 2

    if day == 'Wednesday': 

        return 3

    if day == 'Thursday': 

        return 4

    if day == 'Friday': 

        return 5

    if day == 'Saturday': 

        return 6

    if day == 'Sunday': 

        return 7

# Applying the function:

data['DayOfTheWeek'] = data.DayOfTheWeek.apply(daytoNumber)



# Convering Gender to numeric form:

data['Gender'] = data.Gender.apply(lambda x:1 if x=='F' else 0)



# One-Hot Encoding of the Target variable:

data = data.join(pd.get_dummies(data['Status']))



# Converting Status to numeric form:

data['Status'] = data.Status.apply(lambda x: 1 if x=='Show-Up' else 0)
# Creating if it is a weekend or not:

data['WeekendOrNot'] = data.DayOfTheWeek.apply(lambda x: 1 if x >= 6 else 0)
#Creating week of the month:

data['WeekofMonth'] = ((pd.to_datetime(data['AppointmentDate']).dt.day)-1)//7+1
# Calculating the awaiting time between the appointment date and appointment registeration:

data['CreatedAwaitingTime'] = (data['AppointmentDate'] - data['AppointmentRegistration']).apply(lambda x: x.total_seconds() / (3600 * 24))
data.Status.value_counts().plot(kind='bar', alpha = 0.5, facecolor = '#7FFFD4', figsize=(12,6))

plt.title("Number of People: Show-Up v/s No-Show", fontsize = '18')

plt.ylabel("Total Number")

plt.grid(b=True)
data.Age.plot(kind='kde', figsize=(14,6))

plt.title("Distribution of Age with Facility")

plt.xlabel("Age")

plt.grid(b=True)

plt.xlim([0,100]) # Plotting here only with age 0 to 100 because as we saw before from the data there are some outliers and values which do not make sense.
data = data[(data['Age'] >= 0) & (data['Age'] < 100)]
data.Gender.value_counts().plot(kind='bar', alpha = 0.5, facecolor = '#DC143C', figsize=(12,6))

plt.title("Number of Males and Females", fontsize = '18')

plt.ylabel("Total Number")

plt.grid(b=True)
data.Sms_Reminder.value_counts().plot(kind='bar', alpha = 0.5, facecolor = '#32CD32', figsize=(12,6))

plt.title("Number of SMS Reminders", fontsize = '18')

plt.ylabel("Total Number of SMS")

plt.grid(b=True)



smsdata = pd.DataFrame(data.Sms_Reminder.value_counts())

smsdata['Noofreminder'] = smsdata.index

smsdata = smsdata.reset_index()

del smsdata['index']

total = smsdata['Sms_Reminder'].sum()

smsdata['Percentage'] = smsdata.Sms_Reminder.apply(lambda x: x/total*100)

for idx, no in enumerate(pd.unique(smsdata.Noofreminder)):

    print('The percentage for ' + str(no) + ' reminders is ' + str(round(smsdata.loc[idx,'Percentage'])) + '%')
plt.figure(figsize=(10,8))

plt.scatter(range(data.shape[0]), np.sort(data.CreatedAwaitingTime.values))

plt.xlabel('index', fontsize=12)

plt.ylabel('AwaitingTime', fontsize=12)

plt.show()
daycounts = data['DayOfTheWeek'].value_counts()

plt.figure(figsize=(14,7))

sns.barplot(daycounts.index, daycounts.values, alpha=0.8, palette="Blues")

plt.title("Patients by Day of the Week", fontsize = 25, color = "g", alpha = 0.6)

plt.xticks(rotation=45, fontsize=15)

plt.xlabel('Day of the Week', fontsize=20, alpha=0.5)

plt.ylabel('Number of Patients', fontsize=20, alpha=0.5)

plt.show()
DataAge = data.groupby('Age').sum()

DataAge[['Show-Up','No-Show']].plot(fontsize=14, alpha = 0.7, figsize=(14,6))
g= sns.stripplot(data = data, x='Status', y = 'CreatedAwaitingTime', hue='Status', jitter = True)

g.figure.set_size_inches(14,7)

g.axes.set_title("Awating Time for People who Show-Up v/s No-Show", fontsize = 20, color = "r", alpha = 0.6)

sns.plt.ylim(0, 500)

sns.plt.show()
from statsmodels.graphics.mosaicplot import mosaic

mosdata = data[["Status","Gender","Sms_Reminder"]]

mosdata['Status'] = mosdata.Status.apply(lambda x:1 if x=='Show-Up' else 0)

mos = mosaic(mosdata,["Status","Gender","Sms_Reminder"])
TargetnDis = data[['Diabetes', 'Alcholism', 'HyperTension','Handicap', 'Smokes', 'Scholarship', 'Tuberculosis','Show-Up']]

plt.figure(figsize=(14,12))

foo = sns.heatmap(TargetnDis.corr(), vmax=0.6, square=True, annot=True)
from sklearn.model_selection import train_test_split

from sklearn import metrics

#############################

# Setting up the data:

#############################



predictors = ['Age', 'Gender', 'DayOfTheWeek', 'Diabetes', 'Alcholism', 'HyperTension',

       'Handicap', 'Smokes', 'Scholarship', 'Tuberculosis', 'Sms_Reminder',

       'CreatedAwaitingTime','WeekendOrNot']

target = "Status"



X = data.loc[:,predictors]

y = np.ravel(data.loc[:,[target]])



# Split the dataset in train and test:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# Importing the model:

from sklearn.linear_model import LogisticRegression



# Initiating the model:

lr = LogisticRegression()



# Fitting the model:

lr = lr.fit(X_train,y_train)



# Check the accuracy on the training set:

acc_train = lr.score(X_train,y_train)



# Predict on test set:

predicted = lr.predict(X_test)



# Getting the accuracy:

acc_test = metrics.accuracy_score(y_test, predicted)



print ('The accuracy on train data is %s and on test data is %s' % (acc_train,acc_test))
#Examining the coefficients:

list(zip(X.columns, np.transpose(lr.coef_)))
# Importing the model:

from sklearn.ensemble import RandomForestClassifier



# Initiating the model:

rf = RandomForestClassifier()



# Fitting the model:

rf = rf.fit(X_train,y_train)



# Check the accuracy on the training set:

acc_train = rf.score(X_train,y_train)



# Predict on test set:

predicted = rf.predict(X_test)



# Getting the accuracy:

acc_test = metrics.accuracy_score(y_test, predicted)



print ('The accuracy on train data is %s and on test data is %s' % (acc_train,acc_test))
### Iterating over range of trees just to check:

for i in range(10,20):

    rf = RandomForestClassifier(n_estimators=i)

    rf = rf.fit(X_train,y_train)

    acc_train = rf.score(X_train,y_train)

    predicted = rf.predict(X_test)

    acc_test = metrics.accuracy_score(y_test, predicted)

    print ('For {} number of trees the score of train data is {} and on test data is {}'.format(i,acc_train, acc_test) )
# Importing the model:

from sklearn.naive_bayes import GaussianNB



# Initiating the model:

nb = GaussianNB()



# Fitting the model:

nb = nb.fit(X_train,y_train)



# Check the accuracy on the training set:

acc_train = nb.score(X_train,y_train)



# Predict on test set:

predicted = nb.predict(X_test)



# Getting the accuracy:

acc_test = metrics.accuracy_score(y_test, predicted)



print ('The accuracy on train data is %s and on test data is %s' % (acc_train,acc_test))
# Importing the model:

from sklearn.neighbors import KNeighborsClassifier



# Initiating the model:

knn = KNeighborsClassifier()



# Fitting the model:

knn = knn.fit(X_train,y_train)



# Check the accuracy on the training set:

acc_train = knn.score(X_train,y_train)



# Predict on test set:

predicted = nb.predict(X_test)



# Getting the accuracy:

acc_test = metrics.accuracy_score(y_test, predicted)



print ('The accuracy on train data is %s and on test data is %s' % (acc_train,acc_test))
# Trying to see and loop over more neighbors:



for i in range(10,20):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn = knn.fit(X_train,y_train)

    acc_train = knn.score(X_train,y_train)

    predicted = knn.predict(X_test)

    acc_test = metrics.accuracy_score(y_test, predicted)

    print ('For {} number of trees the score of train data is {} and on test data is {}'.format(i,acc_train, acc_test) )