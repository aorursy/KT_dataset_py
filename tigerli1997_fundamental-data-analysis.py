import pandas as pd
import numpy as np

# read csv file
filename ='../input/noshowappointments/KaggleV2-May-2016.csv'
df = pd.read_csv(filename) # default sep=','
df.info()
# found no any missing data
# personally, I want to rename these columns in standard English
df.rename(index=str, columns={'Hipertension':'Hypertension','Handcap':'Handicap'},inplace=True)
sum(df.duplicated())
#found no duplicated records
df['Gender'].value_counts()
# found no error records in Gender column
df['No-show'].value_counts()
# found no error records in No-show column
df.describe()
df.Handicap.value_counts()
df[df['Age']<0]
#delete rows with abnormal value: Age = -1, 1 row was deleted
df.drop(df[df['Age']<0].index,inplace=True)
#delete rows with abnormal value:Handicap > 1, 199 rows were deleted
df.drop(df[df.Handicap > 1].index, inplace=True)
type(df['ScheduledDay'][0])
type(df['AppointmentDay'][0])
df['waitdays'] = pd.to_datetime(df['AppointmentDay'])- pd.to_datetime(df['ScheduledDay'])
df['waitdays'] = df['waitdays'].apply(lambda x: x.days)
df['waitdays'].describe()
#here, we see min: -7days there could be something wrong
# drop them
df.drop(df[df.waitdays < -1].index, inplace=True)
df.drop(['PatientId','AppointmentID','ScheduledDay','AppointmentDay'], axis=1,inplace=True)
df['No-show'] = df['No-show'].replace({'Yes':1,'No':0})
df.rename(index=str,columns={'No-show':'No_show'},inplace =True)
#preview of clean_dataset
df.head()
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

filename = '../input/noshow/clean_data.csv'
dff = pd.read_csv(filename) # here I created another dataframe named 'dff', be careful and don't mix it with the former 'df' : )
gender_noshow = dff.groupby('Gender').sum()['No_show']
gender_noshow.plot.pie(figsize=(5,5),title = 'Gender distribution on NoShow')
median = dff['Age'].median()
low_age = dff.query('Age < {}'.format(median))
high_age = dff.query('Age >= {}'.format(median))

noshow_sum_low = low_age['No_show'].sum()
noshow_sum_high = high_age['No_show'].sum()
locations = [1, 2]
heights = [noshow_sum_low, noshow_sum_high]
labels = ['Low', 'High']
plt.bar(locations, heights, tick_label=labels,color='lightcoral')
plt.title('Relationship Age --- No show')
plt.xlabel('Age Distribution')
plt.ylabel('Total Count On NoShow')
age_noshow = dff.groupby('Age').sum()['No_show']
age_noshow = age_noshow/ df['Age'].value_counts()
age_noshow.sort_values(ascending=False).head(10)
#Number of no-show patients in each hospital
Nei_noshow = dff.groupby('Neighbourhood').sum()['No_show']
top10_noshow = Nei_noshow.sort_values(ascending=False).head(10)
top10_noshow.plot(kind='bar',figsize=(10,5),\
                  title = 'TOP 10 number of patients no-show in Neighbourhood')
#Number of patients who made appointments in each hospital
Nei_group = dff.Neighbourhood.value_counts()
#divide by index and sort result to get rate
Nei_rate = (Nei_noshow / Nei_group).sort_values(ascending=False)
#here, we should drop row with rate=1 and rate =0 for better data model
Nei_rate.drop(labels=[Nei_rate.idxmax(),Nei_rate.idxmin()],inplace=True)
top10_rate_noshow=Nei_rate.head(10)
top10_rate_noshow.plot(kind='bar',figsize=(10,5),\
                      title = 'TOP 10 rates of patients no-show in Neighbourhood')
df_new = dff.groupby('No_show')['Scholarship', 'Hypertension',\
                       'Diabetes', 'Alcoholism', 'Handicap', 'SMS_received'].sum()
noshow_6r = df_new.query("No_show == 1")
noshow_total = dff['No_show'].value_counts()[1]
prop_6r = noshow_6r / noshow_total
sns.set_style('darkgrid')
prop_6r.plot(kind='bar',figsize=(8,8),\
            title='Relationship on 6 factors with No-Show')
wait_noshow = dff.groupby('waitdays').sum()['No_show']
wait_noshow.plot(figsize=(20,10),title='trends on No-show times as wait days increased')