# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import pylab

sns.set_style("whitegrid")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
pd.__version__
np.__version__
df =  pd.read_csv('../input/KaggleV2-May-2016.csv')
df.dtypes
df.describe()
df = df[df.Age >= 0]
df.isnull().sum()
df['No-show'].value_counts(dropna=False)
df = df.rename(columns={'No-show': 'No_show'})

clone = df.copy()

df['iGender'] = clone.Gender.apply(lambda x: 1 if x == 'M' else -1) 

df['iNo_show'] = clone.No_show.apply(lambda x: 1 if x == 'Yes' else -1)
df.corr()
df['Sum_disease'] =  df['Diabetes'] + df['Alcoholism'] + df['Scholarship']  + df['Hipertension'] +  df['Handcap']
clone = df.copy()

df['ScheduledDay'] = clone.ScheduledDay.apply(np.datetime64)

df['AppointmentDay'] = clone.AppointmentDay.apply(np.datetime64)

df['AwaitingTime'] = (df.AppointmentDay.dt.date - df.ScheduledDay.dt.date).dt.days

df.head(5)
# Why there's such date

print(df['AwaitingTime'].value_counts(dropna=False))

df = df[df.AwaitingTime >= 0]
# to be simpler, prefix 's' means 'Scheduled', 'a' means 'Appointment'

df['sHour'],df['sDay'],df['sMonth'],df['sYear'] = df.ScheduledDay.dt.hour,df.ScheduledDay.dt.day,df.ScheduledDay.dt.month,df.ScheduledDay.dt.year

df['aHour'],df['aDay'],df['aMonth'],df['aYear'] = df.AppointmentDay.dt.hour,df.AppointmentDay.dt.day,df.AppointmentDay.dt.month,df.AppointmentDay.dt.year

df['sWeekDay'],df['sDayOfTheWeek'] = df.ScheduledDay.dt.weekday,df.ScheduledDay.dt.weekday_name

df['aWeekDay'],df['aDayOfTheWeek'] = df.AppointmentDay.dt.weekday,df.AppointmentDay.dt.weekday_name
print('Age:',sorted(df.Age.unique()))

print('Gender:',df.Gender.unique()) 

print('Scholarship:',df.Scholarship.unique()) 

print('Hipertension:',df.Hipertension.unique())

print('Diabetes:',df.Diabetes.unique())

print('Alcoholism:',df.SMS_received.unique())

print('Handcap:',df.Handcap.unique()) 

print('SMS_received:',df.SMS_received.unique())

print('No_show:',sorted(df.No_show.unique()))

print('AwaitingTime:',sorted(df.AwaitingTime.unique())) 



print('sYear:',sorted(df.sYear.unique()))

print('sMonth:',df.sMonth.unique()) 

print('sDay:',df.sDay.unique())

print('sHour:',df.sHour.unique())

print('aYear:',sorted(df.aYear.unique()))

print('aMonth:',df.aMonth.unique()) 

print('aDay:',df.aDay.unique())

print('aHour:',df.aHour.unique())

print('Sum_disease:',df.Sum_disease.unique())

print('sDayOfTheWeek:',df.sDayOfTheWeek.unique())

print('aDayOfTheWeek:',df.aDayOfTheWeek.unique()) 
# credits:  the functions are referred from Somrik Banerjee's Predicting Show-Up/No-Show

# Currently it assumes 'No' of 'No-show' as showing up. 

# If the assumption does not go real, then all the probability of 'showing up' below, should turn into 'not showing up'.

def probStatusCategorical(group_by):

    rows = []

    for item in group_by:

        for level in df[item].unique():

            row = {'Condition': item}

            total = len(df[df[item] == level])

            n = len(df[(df[item] == level) & (df.No_show == 'No')])

            row.update({'Level': level, 'Probability': n / total})

            rows.append(row)

    return pd.DataFrame(rows)



def probStatus(dataset, group_by):

    dx = pd.crosstab(index = dataset[group_by], columns = dataset.No_show).reset_index()

    dx['probShowUp'] = dx['No'] / df['No_show'].count()

    return dx[[group_by, 'probShowUp']]



def posteriorNoShow(condition):

    levels = list(df[condition].unique())

    if condition not in ['aDayOfTheWeek', 'Gender']: 

        levels.remove(0)

    rows = []

    for level in levels:

        p = len(df[df[condition] == level]) / len(df)

        p1 = len(df[(df[condition] == level) & (df.No_show == 'No')]) / len(df[ 'No_show'])

        p2 = len(df[(df[condition] == level) & (df.No_show == 'No')]) / len(df[ 'No_show'])

        if len(levels) > 1:

            rows.append({'Levels': level, 

                         'Probability': (p * p1) / (p * p1 + p * p2)})

        else:

            rows.append({'Condition': condition,

                         'Probability': (p * p1) / (p * p1 + p * p2)})

    return rows
df['AwaitingTime'].max()
sns.stripplot(data = df, y = 'AwaitingTime', jitter = True,  color='g')

 

plt.ylabel('AwaitingTime(Days)', fontsize=10)

plt.ylim(0, 200)

plt.xlabel('Cases', fontsize=10)

plt.show()
most = df.groupby("AwaitingTime").count().sort_values(by="PatientId", ascending=False).head(40)

most["AwaitingTime_"] = most.index

plt.title("Number of Cases originating in a given AwaitingTime", fontsize=20)

sns.barplot(x="AwaitingTime_", y="PatientId", data=most)

plt.ylabel('Number of Cases', fontsize=10)

plt.xlabel('AwaitingTime(days)', fontsize=10)

plt.show()
df['sHour'].hist(edgecolor='black',figsize=(12,6), color='g')

plt.axvline(df['sHour'].mean(),color='b',linestyle='dashed')

plt.title('Patients count with respect to HourOfTheDay')

plt.ylabel('Patients count', fontsize=12)

plt.xlabel('Hour of the scheduled time', fontsize=12)

plt.show()
sns.lmplot(data = probStatus(df, 'sHour'), x = 'sHour', 

           y = 'probShowUp', fit_reg = True)

plt.title('Probability of showing up with respect to HourOfTheDay')

plt.ylabel('Probability of showing up', fontsize=12)

plt.xlabel('Hour of the scheduled time', fontsize=12)

plt.show()
plt.subplots(figsize=(15,15))

max_df=df.groupby('sDay')['sDay'].count()

max_df=max_df[max_df.values>200]

max_df.sort_values(ascending=True,inplace=True)

mean_df=df[df['sDay'].isin(max_df.index)]

abc=mean_df.groupby(['sMonth','sDay'])['Age'].mean().reset_index()

abc=abc.pivot('sMonth','sDay','Age')

sns.heatmap(abc,annot=True,cmap='RdYlGn',linewidths=0.4)

plt.title('Mean age of the scheduled day')

plt.show()
sns.lmplot(data = probStatus(df, 'sDay'), x = 'sDay', 

           y = 'probShowUp', fit_reg = True)

plt.title('Probability of showing up with respect to Day of The Scheduledtime')

plt.ylabel('Probability of showing up', fontsize=12)

plt.xlabel('Day of the scheduled time', fontsize=12)

sns.lmplot(data = probStatus(df, 'aDay'), x = 'aDay', 

           y = 'probShowUp', fit_reg = True)

plt.title('Probability of showing up with respect to Day of The Appointment time')

plt.ylabel('Probability of showing up', fontsize=12)

plt.xlabel('Day of the Appointment time', fontsize=12)

plt.show()
fig,ax=plt.subplots(1,2,figsize=(18,10))

sns.countplot(df['sDay'],ax=ax[0],palette='Set1').set_title('ScheduledDay')

plt.ylabel('')

sns.countplot(df['aDay'],ax=ax[1],palette='Set1').set_title('AppointmentDay')

plt.xticks(rotation=90)

plt.show()
sns.set(style="ticks")

sns.set(style="darkgrid", color_codes=True)



g = sns.jointplot("aDay", "Age", data=df, kind="reg",

                  xlim=(0, 31), ylim=(0, 115), color="r", size=7)

plt.show()
# 4->Friday, No case in Sunday

sns.barplot(data = probStatusCategorical(['sWeekDay', 'aWeekDay']),

            x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set3')



plt.rcParams['figure.figsize'] = (12.0, 4.0) # set default size of plots

plt.xlim(0, 2)

plt.title('Probability of showing up')

plt.ylabel('Probability')

plt.show()
sns.barplot(data = probStatusCategorical(['sDayOfTheWeek', 'aDayOfTheWeek']),

            x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set3')



plt.rcParams['figure.figsize'] = (12.0, 4.0) # set default size of plots

plt.xlim(0, 2)

plt.title('Probability of showing up')

plt.ylabel('Probability')

plt.show()
sns.barplot(data = probStatusCategorical(['sDayOfTheWeek']),

            x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2',

           hue_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',

                       'Saturday', 'Sunday'])

plt.title('Probability of showing up')

plt.ylabel('Probability')

plt.xlabel('Scheduled Day')

plt.show()



sns.barplot(data = probStatusCategorical(['aDayOfTheWeek']),

            x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2',

           hue_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',

                       'Saturday', 'Sunday'])

plt.title('Probability of showing up')

plt.ylabel('Probability')

plt.xlabel('Appointment Day')

plt.show()
sns.barplot(data = pd.DataFrame(posteriorNoShow('aDayOfTheWeek')), 

            x = 'Levels', y = 'Probability', palette = 'Set2')

plt.xlabel('DayOfTheWeek')

plt.ylabel('Probability')

plt.title('Posterior probability of DayOfTheWeek given a no-show')

plt.show()
fig,ax=plt.subplots(1,2,figsize=(18,10))

sns.countplot(df['sMonth'],ax=ax[0],palette='Set1').set_title('Scheduled  Month')

plt.ylabel('')

sns.countplot(df['aMonth'],ax=ax[1],palette='Set1').set_title('Appointment Month')

plt.xticks(rotation=90)

plt.show()
sns.lmplot(data = probStatus(df, 'sMonth'), x = 'sMonth', 

           y = 'probShowUp', fit_reg = True)

plt.title('Probability of showing up with respect to Day of The Scheduled time')

plt.ylabel('Probability of showing up', fontsize=12)

plt.xlabel('Month of the Scheduled time', fontsize=12)

plt.xlim(0, 12)

plt.ylim(0, 1)

sns.lmplot(data = probStatus(df, 'aMonth'), x = 'aMonth', 

           y = 'probShowUp', fit_reg = True)

plt.title('Probability of showing up with respect to Day of The Appointment time')

plt.ylabel('Probability of showing up', fontsize=12)

plt.xlabel('Month of the Appointment time', fontsize=12)

plt.xlim(0, 12)

plt.ylim(0, 1)

plt.show()
df['sMonth'].head(1)
plt.subplots(figsize=(15,15))

max_month=df.groupby('sMonth')['sMonth'].count()

max_month.sort_values(ascending=True,inplace=True)

mean_df=df[df['sMonth'].isin(df.index)]

abc=mean_df.groupby(['sDay','sMonth'])['iNo_show'].mean().reset_index()

abc=abc.pivot('sDay','sMonth','iNo_show')

sns.heatmap(abc,annot=True,cmap='RdYlGn',linewidths=0.4)

plt.title('Average no show By Scheduled Month & Day')

plt.show()
plt.subplots(figsize=(15,15))

max_month=df.groupby('aMonth')['aMonth'].count()

max_month.sort_values(ascending=True,inplace=True)

mean_df=df[df['aMonth'].isin(df.index)]

abc=mean_df.groupby(['aDay','aMonth'])['iNo_show'].mean().reset_index()

abc=abc.pivot('aDay','aMonth','iNo_show')

sns.heatmap(abc,annot=True,cmap='RdYlGn',linewidths=0.4)

plt.title('Average no show By Appointment Month & Day')

plt.show()
df.groupby('Age')['No_show'].count().plot(figsize=(12,6),color='g')

plt.show()
sns.lmplot(data = probStatus(df, 'Age'), x = 'Age', y = 'probShowUp')

plt.xlim(0, 80)

plt.title('Probability of showing up with respect to Age')

plt.ylabel('Probability of showing up', fontsize=10)

plt.xlabel('Age', fontsize=10)

plt.show()
clone=df.copy()

clone=clone[['Age','SMS_received','No_show']] 

plt.rcParams['figure.figsize'] = (12.0, 80.0)

plat=clone.groupby(['Age','SMS_received'])['No_show'].count().reset_index()

plat=plat.pivot('Age','SMS_received','No_show')

plat.plot.barh(width=0.9)

fig=plt.gcf()

fig.set_size_inches(12,20) 

plt.show()
plt.subplots(figsize=(8,4))

sns.barplot(data = probStatusCategorical(['Gender']),

            x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2')

fig.set_size_inches(12,0.1)

plt.title('Probability of showing up')

plt.ylabel('Probability')

plt.show()
plt.subplots(figsize=(8,4))

sns.barplot(data = probStatusCategorical(['SMS_received']),

            x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set2')

plt.title('Probability of showing up')

plt.ylabel('Probability')

plt.xlabel('Scheduled Day')

fig.set_size_inches(12,0.1)

plt.show()
plt.subplots(figsize=(8,4))

max_df=df.groupby('SMS_received')['SMS_received'].count()

max_df.sort_values(ascending=True,inplace=True)

mean_df=df[df['SMS_received'].isin(df.index)]

abc=mean_df.groupby(['No_show','SMS_received'])['Age'].mean().reset_index()

abc=abc.pivot('No_show','SMS_received','Age')

sns.heatmap(abc,annot=True,cmap='RdYlGn',linewidths=0.4)

plt.title('Average Age By SMS_received & No-show')

plt.show()
plt.subplots(figsize=(8,30))

max_df=df.groupby('SMS_received')['SMS_received'].count()

max_df.sort_values(ascending=True,inplace=True)

mean_df=df[df['SMS_received'].isin(df.index)]

abc=mean_df.groupby(['Age','SMS_received'])['iNo_show'].mean().reset_index()

abc=abc.pivot('Age','SMS_received','iNo_show')

sns.heatmap(abc,annot=True,cmap='RdYlGn',linewidths=0.4)

plt.title('Average no show By SMS_received & Age')

plt.show()
abc.columns = ['a','b']

g = abc['b'] - abc['a'] 

gl1 = len(g)

g = g.dropna()

gl2 = len(g)

plt.subplots(figsize=(80,10))

plt.bar(g.index, g.iloc[:])

plt.title('No show probability increased after SMS received')

plt.xlabel('Age')

plt.ylabel('No show probability increased')

plt.show()

print('Number of No show probability increased is Nan:', gl1 - gl2 )

print('Number of No show probability decreased:',g[g<0])
plt.subplots(figsize=(8,4))

max_df=df.groupby('SMS_received')['SMS_received'].count()

max_df.sort_values(ascending=True,inplace=True)

mean_df=df[df['SMS_received'].isin(df.index)]

abc=mean_df.groupby(['No_show','SMS_received'])['PatientId'].count().reset_index()

abc=abc.pivot('No_show','SMS_received','PatientId')

sns.heatmap(abc,annot=True,cmap='RdYlGn',linewidths=0.4)

plt.title('Patients count By SMS_received & No-show')

plt.show()
plt.subplots(figsize=(8,4))

max_df=df.groupby('SMS_received')['SMS_received'].count()

max_df.sort_values(ascending=True,inplace=True)

mean_df=df[df['SMS_received'].isin(df.index)]

abc=mean_df.groupby(['No_show','SMS_received'])['iGender'].mean().reset_index()

abc=abc.pivot('No_show','SMS_received','iGender')

sns.heatmap(abc,annot=True,cmap='RdYlGn',linewidths=0.4)

plt.title('Average Gender By SMS_received & No-show')

plt.show()



plt.subplots(figsize=(8,20))

abc=mean_df.groupby(['Age','Gender'])['iNo_show'].mean().reset_index()

abc=abc.pivot('Age','Gender','iNo_show')

sns.heatmap(abc,annot=True,cmap='RdYlGn',linewidths=0.4)

plt.title('Average No show By Age & Gender')

plt.show()



plt.subplots(figsize=(8,4))

 

abc=mean_df.groupby(['SMS_received','Gender'])['iNo_show'].mean().reset_index()

abc=abc.pivot('SMS_received','Gender','iNo_show')

sns.heatmap(abc,annot=True,cmap='RdYlGn',linewidths=0.4)

plt.title('Average No show By SMS_received & Gender')

plt.show()
plt.subplots(figsize=(12,4))

sns.barplot(data = probStatusCategorical(['Diabetes', 'Alcoholism', 'Hipertension',

                                         'Handcap', 'Scholarship','Gender']),

            x = 'Condition', y = 'Probability', hue = 'Level', palette = 'Set3')



 

plt.xlim(0, 6)

plt.title('Probability of showing up')

plt.ylabel('Probability')

plt.show()
tuples = []

tuples.extend(posteriorNoShow('Diabetes'))

tuples.extend(posteriorNoShow('Hipertension'))

tuples.extend(posteriorNoShow('Alcoholism'))

tuples.extend(posteriorNoShow('Handcap')) 

tuples.extend(posteriorNoShow('Scholarship'))

plt.subplots(figsize=(8,4))

sns.barplot(data = pd.DataFrame(tuples)[['Condition', 'Probability']], 

            x = 'Condition', y = 'Probability', palette = 'Set2')

plt.title('Posterior probability of diseases and scholarship given a no-show')

plt.ylabel('Probability')

plt.show()
plt.subplots(figsize=(8,4))

sns.barplot(data = pd.DataFrame(posteriorNoShow('Handcap')), 

            x = 'Levels', y = 'Probability', palette = 'Set2')

plt.xlabel('Handicap Levels')

plt.ylabel('Probability')

plt.title('Posterior probability of Handicap given a no-show')

plt.show()
plt.subplots(figsize=(8,4))

sns.lmplot(data = probStatus(df, 'Sum_disease'), x = 'Sum_disease', y = 'probShowUp')

plt.xlim(-1, 6)

plt.ylim(-0.2, 1)

plt.title('Probability of showing up with respect to numbers of diseases')

plt.ylabel('Probability of showing up', fontsize=10)

plt.xlabel('Numbers of Diseases', fontsize=10)

plt.show()
plt.subplots(figsize=(8,4))

df['Neighbourhood'].value_counts()[:10].plot(kind='pie',autopct='%1.1f%%',shadow=True,explode=[0.1,0,0,0,0,0,0,0,0,0])

plt.title('Distribution Of Top Neighbourhood')

plt.show()
# show on the map with distances, the host location is need.
features_train = df[['Age', 'Scholarship','Hipertension', 'Diabetes', 'Alcoholism','Sum_disease', 

                         'Handcap']].iloc[:100000]



labels_train = df.No_show[:100000]



features_test = df[['Age', 'Scholarship','Hipertension', 'Diabetes', 'Alcoholism','Sum_disease', 

                         'Handcap']].iloc[100000:]



labels_test = df.No_show[100000:]
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import MultinomialNB



clf =  MultinomialNB().fit(features_train, labels_train)

print('Accuracy:', round(accuracy_score(labels_test, 

                                        clf.predict(features_test)), 2) * 100, '%')
features_train = df[['Age', 'SMS_received', 'AwaitingTime','sHour','aHour','Sum_disease']].iloc[:100000]

labels_train = df.No_show[:100000]

features_test = df[['Age', 'SMS_received','AwaitingTime','sHour','aHour','Sum_disease']].iloc[100000:]

labels_test = df.No_show[100000:]

clf =  MultinomialNB().fit(features_train, labels_train)

print('Accuracy:', round(accuracy_score(labels_test, 

                                        clf.predict(features_test)), 2) * 100, '%')