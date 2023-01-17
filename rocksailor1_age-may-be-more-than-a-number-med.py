

import pandas as pd

from pandas import DataFrame,Series

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import chi2

data = pd.read_csv('../input/No-show-Issue-Comma-300k.csv')

data = data[(data['Age'] > 0) | (data['Age'] < 100)]
#Define age group and return a coded sequence as well, arbitrary guesses on breaks

def ageGroup(x):

        if x <= 16:

                return '0 to 16',0

        elif x <= 30:

                return '17 to 30',1

        elif x <= 50:

                return '31 to 50',2

        elif x <= 70:

                return '51 to 70',3

        else:

                return 'Over 70',4



#Change to abs for days wait and then group

def timetoAppt(x):

        x = np.abs(x)

        if x <= 7:

                return '0 to 7',0

        elif x <= 30:

                return '8 to 30',1

        elif x <= 90:

                return '31 to 90',2

        elif x <= 180:

                return '91 to 180',3

        elif x <= 365:

                return '181 to 365',4

        else:

                return 'Over 365',5

#Add column for status as numerical 'noshow' 0 = showed up  1 = missed

def status(x):

        if x == 'No-Show':

                return 1

        if x == 'Show-Up':

                return 0

#Codeify the day of the week with sunday starting the week

def dow(x):

        if x == 'Sunday':

                return 0

        elif x == 'Monday':

                return 1

        elif x == 'Tuesday':

                return 2

        elif x == 'Wednesday':

                return 3

        elif x == 'Thursday':

                return 4

        elif x == 'Friday':

                return 5

        elif x == 'Saturday':

                return 6
data['ageGrp'] = data['Age'].apply(ageGroup)

data['timetoAppt'] = data['AwaitingTime'].apply(timetoAppt)

data['codeStatus'] = data['Status'].apply(status)

data['dow'] = data['DayOfTheWeek'].apply(dow)
#Seperate the coded columns

data['cdageGrp'] = pd.DataFrame(data['ageGrp'].tolist())[1]

data['cdtimetoAppt'] = pd.DataFrame(data['timetoAppt'].tolist())[1]



#Sum the number of comorbid conditions they may have

data['coMorbid'] = data[['Diabetes','Alcoolism','HiperTension','Handcap','Smokes','Tuberculosis']].sum(axis = 1)
#Build a forrest and see what is the most important feature

#Tree Parameters

n_estimators = 10 

criterion = 'gini'

n_jobs = -1

target = data['codeStatus'].values

predictors = ['cdageGrp','cdtimetoAppt','coMorbid','dow','Sms_Reminder','Scholarship']



randomForest = RandomForestClassifier(n_estimators = n_estimators,criterion = criterion, n_jobs = n_jobs)

forest1 = randomForest.fit(data[predictors].values,target)



scores = cross_val_score(randomForest,data[predictors].values,target,cv=10,n_jobs = n_jobs)


print('*******MODEL OUTPUT***********')

print('\nAccuracy: %0.2f' % (scores.mean()))



print('\nFeature Importance:')

x = 0

for i in predictors:

        print(i + ': %0.2f ' % (forest1.feature_importances_[x]))

        x += 1



print('\nAge Group Summaries')

for ageGrp in data['ageGrp'].unique():

        print(ageGrp[0])

        print(data['Status'][data['ageGrp'] == ageGrp].value_counts(normalize = True))



print('\nTime to Appointment (Wait Time) Summaries')

for timetoAppt in data['timetoAppt'].unique():

        print(timetoAppt[0])

        print(data['Status'][data['timetoAppt'] == timetoAppt].value_counts(normalize = True))



print('\nDay of Week')

for dow in data['DayOfTheWeek'].unique():

        print(dow)

        print(data['Status'][data['DayOfTheWeek'] == dow].value_counts(normalize = True))
