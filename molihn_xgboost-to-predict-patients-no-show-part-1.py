import numpy as np 

import pandas as pd 

from scipy import stats as ss

import statsmodels.api as sm

import sklearn.metrics as ssm

from datetime import datetime

import matplotlib.pyplot as plt

import seaborn as sns

from statsmodels.stats.proportion import proportions_ztest



import os

print(os.listdir("../input"))
df = pd.read_csv("../input/KaggleV2-May-2016.csv")

print('variables: ' + str(df.columns))
df.info()
df[~ df.PatientId.apply(lambda x: x.is_integer())]
df['PatientId'] = df['PatientId'].astype('int64')
df.set_index('AppointmentID', inplace = True)
df.shape
df['PatientId'].dtype
print('Total appointments: ' + format(df.shape[0], ",d"))

print('Distinct patients: ' + format(df['PatientId'].unique().shape[0], ",d"))
print('Patients with more than one appointment: ' + format((df['PatientId'].value_counts() > 1).sum(), ",d"))
df['PreviousApp'] = df.sort_values(by = ['PatientId','ScheduledDay']).groupby(['PatientId']).cumcount()
a = df.groupby(pd.cut(df.PreviousApp, bins = [-1, 0,1,2,3,4,5, 85], include_lowest = True))[['PreviousApp']].count()

b = pd.DataFrame(a)

b.set_index(pd.Series(['0', '1', '2', '3', '4', '5', '> 5']))
df['NoShow'] = (df['No-show'] == 'Yes')*1
df['PreviousNoShow'] = (df[df['PreviousApp'] > 0].sort_values(['PatientId', 'ScheduledDay']).groupby(['PatientId'])['NoShow'].cumsum() / df[df['PreviousApp'] > 0]['PreviousApp'])
df['PreviousNoShow'].describe()
df['Gender'].value_counts()
colors = ['lightcoral', 'lightskyblue']



plt.pie([71840, 38687], explode = (0.1, 0), labels = ['Female', 'Male'], colors = colors, autopct='%1.1f%%') 



plt.title('Patient Gender', fontsize=15)

plt.gcf().set_size_inches(8, 8)

plt.show()
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])

df['ScheduledDay2'] = df.apply(lambda x: x.ScheduledDay.strftime("%x"), axis = 1)

scheduled_days = df.groupby(['ScheduledDay2'])[['ScheduledDay']].count()
scheduled_days.reset_index(inplace = True)

scheduled_days.columns = ['Date', 'Count']
scheduled_days['Date'] = pd.to_datetime(scheduled_days['Date'])
print('first scheduled: ' + str(scheduled_days.Date.min()))

print('most recent scheduled: ' + str(scheduled_days.Date.max()))
sns.scatterplot(x = 'Date', y = 'Count', data = scheduled_days)

plt.title('Number of Appointments per Scheduled Day')

plt.xlabel('Scheduled Day')

plt.xlim('2015-12', '2016-07')

plt.gcf().set_size_inches(10, 6)

plt.show()
df['WeekdayScheduled'] = df.apply(lambda x: x.ScheduledDay.isoweekday(), axis = 1)

df['WeekdayScheduled'].value_counts()
df = df[df['WeekdayScheduled'] < 6]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightpink']



plt.pie([df['WeekdayScheduled'].value_counts()[1], df['WeekdayScheduled'].value_counts()[5], 

         df['WeekdayScheduled'].value_counts()[4], df['WeekdayScheduled'].value_counts()[3], df['WeekdayScheduled'].value_counts()[2]], 

        labels = ['Monday','Friday','Thursday','Wednesday' ,'Tuesday'], 

        colors = colors, autopct='%1.1f%%') 



plt.title('Day of Week - Scheduled', fontsize=15)

plt.gcf().set_size_inches(8, 8)

plt.show()
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

appoint_days = df.groupby(['AppointmentDay'])[['No-show']].count()
appoint_days.reset_index(inplace = True)

appoint_days.columns = ['Date', 'Count']

appoint_days['Date'] = pd.to_datetime(appoint_days['Date'])
print('first appointment: ' + str(appoint_days.Date.min()))

print('most recent appointment: ' + str(appoint_days.Date.max()))
sns.scatterplot(x = 'Date', y = 'Count', data = appoint_days)

plt.title('Number of Appointments per Day')

plt.xlabel('Appointment Day')

plt.xlim('2016-04-28', '2016-06-09')

plt.gcf().set_size_inches(10, 6)

plt.show()
df['WeekdayAppointment'] = df.apply(lambda x: x.AppointmentDay.isoweekday(), axis = 1)

df['WeekdayAppointment'].value_counts()
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightpink']

df2 = df[df['WeekdayAppointment'] < 6]



plt.pie([df2['WeekdayAppointment'].value_counts()[1], df2['WeekdayAppointment'].value_counts()[5], 

         df2['WeekdayAppointment'].value_counts()[4], df2['WeekdayAppointment'].value_counts()[3], df2['WeekdayAppointment'].value_counts()[2]], 

        labels = ['Monday','Friday','Thursday','Wednesday' ,'Tuesday'], 

        colors = colors, autopct='%1.1f%%') 



plt.title('Day of Week - Appointment', fontsize=15)

plt.gcf().set_size_inches(8, 8)

plt.show()
df2['Age'].describe()
print('Number of obs with negative age: ' + format(df2[df2['Age'] < 0].shape[0]))
df2 = df2[df2['Age'] >=0]

ages = df2.groupby(['Age'])[['PatientId']].count()

ages.reset_index(inplace = True)

ages.columns = ['Age', 'Count']
ax = sns.boxplot(x=df2['Age'], orient = 'v')



#plt.xlabel(' ')

plt.ylabel(' ')

plt.title('Boxplot - Age', fontsize = 15)

plt.gcf().set_size_inches(8, 8)

plt.show()
df2[df2['Age'] > 110]
print('Number of different Neighbourhoods: ' + format(df2['Neighbourhood'].value_counts().size))
df2['Scholarship'].value_counts() 
colors = ['lightskyblue','lightcoral']



plt.pie([df2['Scholarship'].value_counts()[1] ,df2['Scholarship'].value_counts()[0] ], explode = (0.1, 0), labels = ['Yes', 'No'], colors = colors, autopct='%1.1f%%') 



plt.title('Scholarship (receives government aid)', fontsize=15)

plt.gcf().set_size_inches(8, 8)

plt.show()
df2['Hipertension'].value_counts()
plt.pie([df2['Hipertension'].value_counts()[1] ,df2['Hipertension'].value_counts()[0] ], 

        explode = (0.1, 0), labels = ['Yes', 'No'], colors = colors, autopct='%1.1f%%') 



plt.title('Hypertension Diagnosed', fontsize=15)

plt.gcf().set_size_inches(8, 8)

plt.show()
df2['Diabetes'].value_counts()
plt.pie([df2['Diabetes'].value_counts()[1] ,df2['Diabetes'].value_counts()[0] ], 

        explode = (0.1, 0), labels = ['Yes', 'No'], colors = colors, autopct='%1.1f%%') 



plt.title('Diabetes Diagnosed', fontsize=15)

plt.gcf().set_size_inches(8, 8)

plt.show()
df2['Alcoholism'].value_counts()
plt.pie([df2['Alcoholism'].value_counts()[1] ,df2['Alcoholism'].value_counts()[0] ], 

        explode = (0.1, 0), labels = ['Yes', 'No'], colors = colors, autopct='%1.1f%%') 



plt.title('Alcoholism', fontsize=15)

plt.gcf().set_size_inches(8, 8)

plt.show()
df2['Handcap'].value_counts()
df2['HasHandicap'] = (df['Handcap'] > 0)*1
plt.pie([df2['HasHandicap'].value_counts()[1] ,df2['HasHandicap'].value_counts()[0] ], 

        explode = (0.1, 0), labels = ['Yes', 'No'], colors = colors, autopct='%1.1f%%') 



plt.title('Does the patient have any handicap?', fontsize=15)

plt.gcf().set_size_inches(8, 8)

plt.show()
df2['SMS_received'].value_counts()
plt.pie([df2['SMS_received'].value_counts()[1] ,df2['SMS_received'].value_counts()[0] ], 

        explode = (0.1, 0), labels = ['Yes', 'No'], colors = colors, autopct='%1.1f%%') 



plt.title('Was a message sent to the patient to remind of the appointment?', fontsize=15)

plt.gcf().set_size_inches(8, 8)

plt.show()
df2['PreviousDisease'] = df2.apply(lambda x: ((x.Hipertension == 1 )| x.Diabetes == 1 | x.Alcoholism == 1)*1, axis = 1)
df2['PreviousDisease'].value_counts()
plt.pie([df2['PreviousDisease'].value_counts()[1] ,df2['PreviousDisease'].value_counts()[0] ], 

        explode = (0.1, 0), labels = ['Yes', 'No'], colors = colors, autopct='%1.1f%%') 



plt.title('Does the patient have any previous disease (hipertension, diabetes or alcoholism)?', fontsize=15)

plt.gcf().set_size_inches(8, 8)

plt.show()
def get_day(x):

    return x.date()



df2['DaysBeforeApp'] = ((df2.AppointmentDay.apply(get_day) - df2.ScheduledDay.apply(get_day)).astype('timedelta64[D]')).astype(int)
df2['DaysBeforeApp'].value_counts()
df2[df2['DaysBeforeApp'] < 0]
df3 = df2[df2['DaysBeforeApp'] >= 0]
days_before = df3.groupby(['DaysBeforeApp'])[['No-show']].count()

days_before.reset_index(inplace = True)

days_before.columns = ['Days Ahead', 'Count']
sns.scatterplot(x = 'Days Ahead', y = 'Count', data = days_before)

plt.title('Number of Appointments by Lead Days ')

plt.xlabel('Lead Days')

#plt.xlim('2016-04-28', '2016-06-09')

plt.gcf().set_size_inches(10, 6)

plt.show()
def DaysBeforeCat(days):

    if days == 0:

        return '0 days'

    elif days in range(1,3):

        return '1-2 days'

    elif days in range(3,8):

        return '3-7 days'

    elif days in range(8, 32):

        return '8-31 days'

    else:

        return '> 31 days'

    

df3['DaysBeforeCat'] = df3.DaysBeforeApp.apply(DaysBeforeCat)
df3['DaysBeforeCat'].value_counts()
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightpink']



plt.pie([df3['DaysBeforeCat'].value_counts()[0], df3['DaysBeforeCat'].value_counts()[3], 

         df3['DaysBeforeCat'].value_counts()[2], df3['DaysBeforeCat'].value_counts()[1], df3['DaysBeforeCat'].value_counts()[4]], 

        labels = ['0 days','1-2 days' ,'3-7 days','8-31 days','> 31 days'], 

        explode = (0.1, 0, 0, 0, 0),

        colors = colors, autopct='%1.1f%%') 



plt.title('Lead Days', fontsize=15)

plt.gcf().set_size_inches(8, 8)

plt.show()
df3['No-show'].value_counts()[1]
ns = df3['No-show'].value_counts()[1]

show = df3['No-show'].value_counts()[0]

rate = (show + 0.0) / ns

print('For every no-show, there are {:1.2f} shows'.format(rate))
def unique_condition(df, var, cols):

    if df.groupby(cols).ngroups == df[var].unique().size:

        return 'Sizes match: unique value per ' + var

    else: 

        return 'Mismatch: more than one value per ' + var



unique_condition(df3, 'PatientId', ['PatientId','Hipertension', 'Diabetes', 

                                    'Alcoholism', 'Gender', 'Handcap', 'Scholarship'])
print('Reservations scheduled after appointment time: ' + str(df3[df3['DaysBeforeApp'] < 0].size))
inconsist = []

for num in df3['PatientId'].unique():

    ages = df3[df3['PatientId'] == num]['Age'].unique()

    if ages.size == 1:

        break

    if ages.size > 2:

        inconsist.append(num)

        print('Patient ' + str(num)+ 'has age inconsistency')

    else:

        if abs(ages[0]-ages[1]) > 1:

            inconsist.append(num)

            print('Patient ' + str(num)+ 'has age inconsistency')

            

if len(inconsist) == 0:

    print('There is no inconsistency in ages')
sns.set()



def cat_var(df3, var):

    

    print(df3.groupby([var])['NoShow'].mean())

    

    ns_rate = [df3.groupby([var])['NoShow'].mean()[i] for i in df3[var].unique()]

    s_rate = [1-df3.groupby([var])['NoShow'].mean()[i] for i in df3[var].unique()]

    barWidth = 0.5



    plt.bar(df3[var].unique(), ns_rate, color='lightcoral', edgecolor='white', width=barWidth, label = 'No-Show')

    plt.bar(df3[var].unique(), s_rate, bottom=ns_rate, color='mediumseagreen', edgecolor='white', width=barWidth, label = 'Show')

    plt.axhline(y=df3['NoShow'].mean(), color='black', linewidth= 0.8, linestyle='--', label = 'Overall mean')

    plt.xticks(df3[var].unique())

    plt.xlabel(var)

    plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)

    plt.title('No-Show Rate by '+ var, fontsize=15)

    plt.gcf().set_size_inches(6, 6)

    plt.show() 

    

    counts = np.array(df3.groupby([var])['NoShow'].sum())

    nobs = np.array(df3.groupby([var])['NoShow'].count())



    table = df3.groupby(['NoShow', var]).size().unstack(var)

    pvalue = ss.chi2_contingency(table.fillna(0))[1]

    

    print('Means test p-value: {:1.3f}'.format(pvalue))

    if pvalue < 0.05:

        print('Reject null hypothesis: no-show rate is different for at least one group')

    else:

        print('Cannot reject no-show rates are same for all groups')
cat_var(df3, 'Gender')
ax = sns.boxplot(x="NoShow", y="Age",data= df3, palette = 'RdBu')

ax.set_xticklabels(['Show', 'No Show'])

plt.xlabel(' ')

plt.ylabel('Age (in years)')

plt.title('Age Boxplot by No-Show', fontsize = 15)

plt.gcf().set_size_inches(12, 8)

plt.show()
print('Correlation with No-Show: %.3f' % ss.pointbiserialr(df3['NoShow'], df3['Age'])[0])

cat_var(df3, 'Scholarship')
cat_var(df3, 'Hipertension')
cat_var(df3, 'Diabetes')
cat_var(df3, 'Alcoholism')
cat_var(df3, 'SMS_received')
aux
aux = df3.groupby(['DaysBeforeApp'])[['SMS_received']].agg(['count','sum'])

aux.columns = ['count', 'SMS_received']

aux[:20]['SMS_received'].plot()

plt.gcf().set_size_inches(10, 6)

plt.xlabel('Anticipation Days')

plt.ylabel('Count (n)')

plt.title('Frequency of Days Before Appointment (anticipation)')

plt.xticks(range(0, 20))

plt.show()
fourdaysormore = df3[df3['DaysBeforeApp'] > 3]

cat_var(fourdaysormore, 'SMS_received')
prevapp = df3.groupby(['PreviousApp'])[['NoShow']].agg(['count', 'mean'])

prevapp.columns = ['count', 'NoShow_rate']
prevapp.reset_index(inplace = True)
import warnings

warnings.filterwarnings("ignore")

prevapp = prevapp[(prevapp['count'] > 30) & (prevapp['PreviousApp'] > 0)]
fig = plt.figure()



count = fig.add_subplot(111)

rate = count.twinx()



count.set_ylabel('N')

rate.set_ylabel('No-Show rate')



line1 = count.bar(prevapp['PreviousApp'], prevapp['count'])

line2 = rate.plot(prevapp['PreviousApp'], prevapp['NoShow_rate'], color = 'red', label = 'No-show Rate')

count.legend([line1, line2], ['Count', 'No-show Rate'])

plt.gcf().set_size_inches(12, 8)

count.set_xlabel('Number of Previous Appointments')

plt.title('Number of Previous Appointments: total and no-show rates (n > 30)')

plt.show()
print('Correlation with No-Show (all appointments): %.3f' % ss.pointbiserialr(df3['NoShow'], df3['PreviousApp'])[0])

print('Correlation with No-Show (1 or more previous app): %.3f' % ss.pointbiserialr(df3[df3['PreviousApp'] > 0]['NoShow'], df3[df3['PreviousApp'] > 0]['PreviousApp'])[0])
prop_ns = df3.groupby(pd.cut(df3['PreviousNoShow'], np.arange(0, 1.05, 0.05), include_lowest = True))[['NoShow']].mean()

prop_ns = prop_ns.reset_index()

prop_ns['middle'] = np.arange(0.025, 1.025, 0.05)

prop_ns.iloc[0,2] = 0

prop_ns.iloc[19,2] = 1
no_na = df3.dropna(subset = ['PreviousNoShow'])
prop_ns = prop_ns.drop([18], axis = 0)

plt.plot(prop_ns['middle'], prop_ns['NoShow'], color = '#16a4e3')



plt.xlim(0,1)

plt.ylim(0,1)

plt.xlabel('Previous Appointments No-Show Rate', labelpad=10)

plt.ylabel('No-Show (%)')

plt.grid(True)

plt.gcf().set_size_inches(12, 8)

plt.title('No Show Rate by Proportion of Previous Appointments No-Show', fontsize = 15)



plt.show()



print('Correlation with No-Show: %.3f' % ss.pointbiserialr(no_na['NoShow'], no_na['PreviousNoShow'])[0])

cat_var(df3, 'WeekdayScheduled')
cat_var(df3, 'WeekdayAppointment')
cat_var(df3, 'HasHandicap')
cat_var(df3, 'PreviousDisease')
cat_var(df3, 'DaysBeforeCat')