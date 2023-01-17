import pandas as pd

import numpy as np

from datetime import datetime

from pylab import *



# Check data.

df = pd.read_csv(

    '../input/No-show-Issue-Comma-300k.csv', parse_dates=['ApointmentData', 'AppointmentRegistration'])

df = df.rename(columns={'Alcoolism': 'Alcoholism', 'ApointmentData': 'AppointmentDate', 'Handcap': 'Handicap'})

df.head()

# There are no missing values. Convenient.

df.groupby('Status').count()

df['ShowedUp'] = df['Status']=='Show-Up'
for c in df:

    if c == 'AppointmentRegistration':

        print(c, 'hour', '\n\t', sorted(df[c].apply(lambda d: d.hour).unique()))

    elif c == 'AppointmentDate':

        print(c, 'month', '\n\t', sorted(df[c].apply(lambda d: d.month).unique()))

    else:

        print(c, '\n\t', sorted(df[c].unique()))
# This looks like a bug

df['AwaitingTime'] *= -1
# Histogram of both show-up and didn't-show-up cases.

hist([

    df['Age'].where(df['ShowedUp']).dropna(), 

    df['Age'].where(~df['ShowedUp']).dropna(),

    ], stacked=True, bins=range(-2, 110, 1))

legend(['Showed up', 'Didn\'t'])

gca().set_xlabel('Age')



# Probability of showing up vs age.

by_age = df['ShowedUp'].groupby(df['Age'])

mean = by_age.mean()

se = by_age.std()/by_age.count()**.5 # Approx. sample uncertainty

figure(figsize=(20, 10))

mean.plot(yerr=se, kind='bar')

gca().set_ylabel('Sample probability of showing up')



# Monthly time series of proportion showing up.

monthly_groups = df['ShowedUp'].groupby(df['AppointmentDate'].apply(lambda d: d.year+d.month/12))

monthly_means = monthly_groups.mean()



# Plot and regress using seaborn's regplot

import seaborn as sns

fig, ax = plt.subplots(figsize=(10, 6))

monthly_means.index

sns.regplot(x=monthly_means.index.values, y=monthly_means.values, ax=ax)

gca().set_ylabel('Sample show-up probability')
registration_hour = df['AppointmentRegistration'].apply(lambda d: d.hour)

showup_by_reghour = df['ShowedUp'].groupby(registration_hour)

mean = showup_by_reghour.mean()

se = showup_by_reghour.std()/showup_by_reghour.count()**.5 # Approx. sample uncertainty



figure()

mean.plot(kind='bar', yerr=se)

df.groupby('Gender')['ShowedUp'].mean().plot(kind='bar')

gca().set_ylim((0, 1))

gca().set_ylabel('P(Shows up)')
groups = df.groupby('DayOfTheWeek')['ShowedUp'].mean()

groups = groups[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']].plot(kind='bar')

gca().set_ylim((0, 1))

gca().set_ylabel('P(Shows up)')
conditions = [

    'Diabetes', 'Alcoholism', 'HiperTension', 'Handicap', 'Smokes', 'Tuberculosis', 

    'Sms_Reminder',

    ]

names, values = zip(*(

    ('%s_%s'%(c,v), df[df[c]==v]['ShowedUp'].mean()) for c in conditions for v in df[c].unique()))

pd.Series(values, names).plot(kind='barh')

gca().set_xlabel('Condition states and indicators')

bins = list(range(0, 50, 2)) + list(range(50, 370, 20))

groups = df['ShowedUp'].groupby(np.digitize(df['AwaitingTime'], bins))

mean = groups.mean()

se = groups.std()/groups.count()**.5 # Approx. sample uncertainty

figure(figsize=(10, 6))

mean.plot(yerr=se, kind='bar')

gca().set_xticklabels(bins)

gca().set_ylabel('Sample probability of showing up')