import numpy as np

import pandas as pds

import matplotlib.pyplot as plt

from matplotlib import pylab

import seaborn as sns

sns.set_style("whitegrid")



noShow = pds.read_csv('../input/No-show-Issue-Comma-300k.csv')

print(noShow.head())
noShow.rename(columns = {'ApointmentData':'AppointmentData',

                         'Alcoolism': 'Alchoholism',

                         'HiperTension': 'Hypertension',

                         'Handcap': 'Handicap'}, inplace = True)



print(noShow.columns)
noShow.AppointmentRegistration = noShow.AppointmentRegistration.apply(np.datetime64)

noShow.AppointmentData = noShow.AppointmentData.apply(np.datetime64)

noShow.AwaitingTime = noShow.AwaitingTime.apply(abs)



print(noShow.AppointmentRegistration.head())

print(noShow.AppointmentData.head())

print(noShow.AwaitingTime.head())
daysToAppointment = noShow.AppointmentData - noShow.AppointmentRegistration

daysToAppointment = daysToAppointment.apply(lambda x: x.total_seconds() / (3600 * 24))

plt.scatter(noShow.AwaitingTime, daysToAppointment)

plt.axis([0, 400, 0, 400])

plt.xlabel('AwaitingTime')

plt.ylabel('daysToAppointment')

plt.show()
def calculateHour(timestamp):

    timestamp = str(timestamp)

    hour = int(timestamp[11:13])

    minute = int(timestamp[14:16])

    second = int(timestamp[17:])

    return round(hour + minute/60 + second/3600)



noShow['HourOfTheDay'] = noShow.AppointmentRegistration.apply(calculateHour)
print('Age:',sorted(noShow.Age.unique()))

print('Gender:',noShow.Gender.unique())

print('DayOfTheWeek:',noShow.DayOfTheWeek.unique())

print('Status:',noShow.Status.unique())

print('Diabetes:',noShow.Diabetes.unique())

print('Alchoholism:',noShow.Alchoholism.unique())

print('Hypertension:',noShow.Hypertension.unique())

print('Handicap:',noShow.Handicap.unique())

print('Smokes:',noShow.Smokes.unique())

print('Scholarship:',noShow.Scholarship.unique())

print('Tuberculosis:',noShow.Tuberculosis.unique())

print('Sms_Reminder:',noShow.Sms_Reminder.unique())

print('AwaitingTime:',sorted(noShow.AwaitingTime.unique()))

print('HourOfTheDay:', sorted(noShow.HourOfTheDay.unique()))
noShow = noShow[(noShow.Age >= 0) & (noShow.Age <= 95)]
sns.stripplot(data = noShow, y = 'AwaitingTime', jitter = True)

sns.plt.ylim(0, 500)

sns.plt.show()
noShow = noShow[noShow.AwaitingTime < 350]
def probStatus(dataset, group_by):

    df = pds.crosstab(index = dataset[group_by], columns = dataset.Status).reset_index()

    df['probShowUp'] = df['Show-Up'] / (df['Show-Up'] + df['No-Show'])

    return df[[group_by, 'probShowUp']]
sns.lmplot(data = probStatus(noShow, 'Age'), x = 'Age', y = 'probShowUp', fit_reg = True)

sns.plt.xlim(0, 100)

sns.plt.title('Probability of showing up with respect to Age')

sns.plt.show()



sns.lmplot(data = probStatus(noShow, 'HourOfTheDay'), x = 'HourOfTheDay', 

           y = 'probShowUp', fit_reg = True)

sns.plt.title('Probability of showing up with respect to HourOfTheDay')

sns.plt.show()



sns.lmplot(data = probStatus(noShow, 'AwaitingTime'), x = 'AwaitingTime', 

           y = 'probShowUp', fit_reg = True)

sns.plt.title('Probability of showing up with respect to AwaitingTime')

sns.plt.ylim(0, 1)

sns.plt.show()
def probStatusCategorical(group_by):

    rows = []

    for item in group_by:

        for level in noShow[item].unique():

            row = {'Condition': item}

            total = len(noShow[noShow[item] == level])

            n = len(noShow[(noShow[item] == level) & (noShow.Status == 'Show-Up')])

            row.update({'Level': level, 'Probability': n / total})

            rows.append(row)

    return pds.DataFrame(rows)



sns.barplot(data = probStatusCategorical(['Diabetes', 'Alchoholism', 'Hypertension',

                                         'Tuberculosis', 'Smokes', 'Scholarship']),

            x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2')

sns.plt.title('Probability of showing up')

sns.plt.ylabel('Probability')

sns.plt.show()
sns.barplot(data = probStatusCategorical(['Gender']),

            x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2')

sns.plt.title('Probability of showing up')

sns.plt.ylabel('Probability')

sns.plt.show()
sns.barplot(data = probStatusCategorical(['Handicap']),

            x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2')

sns.plt.title('Probability of showing up')

sns.plt.ylabel('Probability')

sns.plt.show()
sns.barplot(data = probStatusCategorical(['DayOfTheWeek']),

            x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2',

           hue_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',

                       'Saturday', 'Sunday'])

sns.plt.title('Probability of showing up')

sns.plt.ylabel('Probability')

sns.plt.show()
sns.barplot(data = probStatusCategorical(['Sms_Reminder']),

            x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2')

sns.plt.title('Probability of showing up')

sns.plt.ylabel('Probability')

sns.plt.show()
def posteriorNoShow(condition):

    levels = list(noShow[condition].unique())

    if condition not in ['DayOfTheWeek', 'Gender']: 

        levels.remove(0)

    rows = []

    for level in levels:

        p = len(noShow[noShow[condition] == level]) / len(noShow)

        p1 = len(noShow[(noShow[condition] == level) & (noShow.Status == 'No-Show')]) / len(noShow[noShow.Status == 'No-Show'])

        p2 = len(noShow[(noShow[condition] == level) & (noShow.Status == 'Show-Up')]) / len(noShow[noShow.Status == 'Show-Up'])

        if len(levels) > 1:

            rows.append({'Levels': level, 

                         'Probability': (p * p1) / (p * p1 + p * p2)})

        else:

            rows.append({'Condition': condition,

                         'Probability': (p * p1) / (p * p1 + p * p2)})

    return rows

    

tuples = []

tuples.extend(posteriorNoShow('Diabetes'))

tuples.extend(posteriorNoShow('Hypertension'))

tuples.extend(posteriorNoShow('Alchoholism'))

tuples.extend(posteriorNoShow('Tuberculosis'))

tuples.extend(posteriorNoShow('Smokes'))

tuples.extend(posteriorNoShow('Scholarship'))



sns.barplot(data = pds.DataFrame(tuples)[['Condition', 'Probability']], 

            x = 'Condition', y = 'Probability', palette = 'Set2')

sns.plt.title('Posterior probability of diseases and scholarship given a no-show')

sns.plt.ylabel('Probability')

sns.plt.show()
sns.barplot(data = pds.DataFrame(posteriorNoShow('Handicap')), 

            x = 'Levels', y = 'Probability', palette = 'Set2')

sns.plt.xlabel('Handicap Levels')

sns.plt.ylabel('Probability')

sns.plt.title('Posterior probability of Handicap given a no-show')

sns.plt.show()
sns.barplot(data = pds.DataFrame(posteriorNoShow('Gender')), 

            x = 'Levels', y = 'Probability', palette = 'Set2')

sns.plt.xlabel('Gender')

sns.plt.ylabel('Probability')

sns.plt.title('Posterior probability of Gender given a no-show')

sns.plt.show()
sns.barplot(data = pds.DataFrame(posteriorNoShow('DayOfTheWeek')), 

            x = 'Levels', y = 'Probability', palette = 'Set2')

sns.plt.xlabel('DayOfTheWeek')

sns.plt.ylabel('Probability')

sns.plt.title('Posterior probability of DayOfTheWeek given a no-show')

sns.plt.show()
def dayToNumber(day):

    if day == 'Monday': 

        return 0

    if day == 'Tuesday': 

        return 1

    if day == 'Wednesday': 

        return 2

    if day == 'Thursday': 

        return 3

    if day == 'Friday': 

        return 4

    if day == 'Saturday': 

        return 5

    if day == 'Sunday': 

        return 6



noShow.Gender = noShow.Gender.apply(lambda x: 1 if x == 'M' else 0)

noShow.DayOfTheWeek = noShow.DayOfTheWeek.apply(dayToNumber)

noShow.Status = noShow.Status.apply(lambda x: 1 if x == 'Show-Up' else 0)
features_train = noShow[['Age', 'Diabetes','Hypertension', 'Tuberculosis', 'Smokes',

                         'Alchoholism', 'Scholarship']].iloc[:296500]



labels_train = noShow.Status[:296500]



features_test = noShow[['Age', 'Diabetes','Hypertension', 'Tuberculosis', 'Smokes',

                         'Alchoholism', 'Scholarship']].iloc[296500:]



labels_test = noShow.Status[296500:]
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import MultinomialNB



clf =  MultinomialNB().fit(features_train, labels_train)

print('Accuracy:', round(accuracy_score(labels_test, 

                                        clf.predict(features_test)), 2) * 100, '%')