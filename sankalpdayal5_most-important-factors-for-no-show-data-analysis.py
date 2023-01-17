# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Import all necessary libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb



%matplotlib inline
# Load data

df = pd.read_csv('../input/KaggleV2-May-2016.csv', parse_dates=True)
df.head()
df.info()
df.shape
df.columns
# Modify Data types



df['PatientId'] = df['PatientId'].astype('str')

df['AppointmentID'] = df['AppointmentID'].astype('str')

df['PatientId'] = df['PatientId'].str.split('.', expand=True)[0]
# Format Date



df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'], format="%Y-%m-%dT%H:%M:%SZ")

df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'], format="%Y-%m-%dT%H:%M:%SZ")
# Get dummies for No-show and Gender columns



df[['Present','Absent']] = pd.get_dummies(df['No-show'])

# df[['Female','Male']] = pd.get_dummies(df['Gender'])
# Drop extra columns



df.drop(['Absent','No-show'], inplace=True, axis=1)

# df.drop(['Gender','Female'], inplace=True, axis=1)
# No of Duplicated values



for _ in df.columns:

    print(_,sum(df[_].duplicated()))
# No of Unique values



for _ in df.columns:

    print(_,len(df[_].unique()))

    print((df[_].unique()),'\n')
df.head()
df.info()
df.describe()
# Removing the record with negative Age



df[df['Age'] < 0].index

df.drop(df[df['Age'] < 0].index, inplace=True)
# Confirming that the data is removed



df[df['Age'] < 0]
# Adding time difference between Appointment and Scheduled



df['WaitingTime'] = df['ScheduledDay'] - df['AppointmentDay']

dates = df['WaitingTime'].abs()



def dayCount(dates):

    return dates.days



dates = dates.map(dayCount)

df.head()
# Adding month of Appointment and Day of Appointment



df['Month'] = df['AppointmentDay'].dt.month_name()

df['Day'] = df['AppointmentDay'].dt.day_name()

df['Hour'] = df['ScheduledDay'].dt.hour
# Reorder Columns



column_order = df.columns.tolist()

column_order = ['PatientId','AppointmentID','Gender','ScheduledDay','AppointmentDay','Age','Neighbourhood','Scholarship',

                'Hipertension','Diabetes','Alcoholism','Handcap','SMS_received','WaitingTime','Month','Day','Hour',

                'Present']

df = df[column_order]

df.shape
df.head()
temp = df['Present'].value_counts()

x_marker = ['Present', 'Absent']

plt.pie(temp, labels = x_marker, autopct='%1.1f%%', shadow=True, startangle=90)

plt.title('Show-up Ratio');



# df['Present'].value_counts().to_list()
temp = df['Month'].value_counts()

# .to_list()

x_marker = df['Month'].value_counts().index.tolist()

plt.pie(temp, labels = x_marker, autopct='%1.1f%%', shadow=True, startangle=90,counterclock=False)

plt.title('Monthly Show-up');
sb.countplot(data=df, x='Month', hue='Present');

plt.title('No of Show-up based on month')

plt.legend(['Absent','Present'], title='Show-up');

plt.xlabel('Day of Week')

plt.ylabel('Total no of Show-up');
sb.countplot(data=df, x='Month', hue='Gender');

plt.title('No of Show-up based on month')

plt.legend(['Male','Female'], title='Gender');

plt.xlabel('Day of Week')

plt.ylabel('Total no of Show-up');
x_marker = ['Monday', 'Tuesday', 'Wednesday', 'Thursday','Friday', 'Saturday']

sb.countplot(data=df, x='Day', hue='Present', order=x_marker);

plt.title('No of Show-up based on day of the week')

plt.legend(['Absent','Present'], title='Show-up');

plt.xlabel('Day of Week')

plt.ylabel('Total no of Show-up');
x_marker = ['Monday', 'Tuesday', 'Wednesday', 'Thursday','Friday', 'Saturday']

sb.barplot(data=df, x='Day', y='Present', hue='Gender', order=x_marker);

plt.title('Percentage of Show-up based on day of the week');

plt.legend(loc='lower right', title='Gender');

plt.xlabel('Day of Week')

plt.ylabel('Percentage of Show-up');
df.groupby('Hour').count()

# x_marker = ['Monday', 'Tuesday', 'Wednesday', 'Thursday','Friday', 'Saturday']

sb.countplot(data=df, x='Hour', hue='Gender');

plt.legend(['Male','Female'], title='Gender');

plt.xlabel('Hour of the day')

plt.ylabel('Total no of Show-up')

plt.title('Show-up based on hour of the Day (Gender based)');
sb.countplot(data=df, x='Hour', hue='Present');

plt.legend(['Absent','Present'], title='Show-up');

plt.xlabel('Hour of the day')

plt.ylabel('Total no of Show-up')

plt.title('Show-up based on hour of the Day (Show-up based)');
sb.countplot(data=df, x='Scholarship', hue='Gender');
sb.countplot(data=df, x='Hipertension', hue='Gender');
sb.countplot(data=df, x='Diabetes', hue='Gender');
sb.countplot(data=df, x='Alcoholism', hue='Gender');
sb.countplot(data=df, x='Handcap', hue='Gender');
sb.countplot(data=df, x='SMS_received', hue='Gender');
g = sb.PairGrid(data = df, vars = ['Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'Present'])

g.map_diag(plt.hist)

g.map_offdiag(plt.scatter);
sb.heatmap(df.corr(), annot = True, fmt = '.2f', cmap = 'vlag_r', center = 0);
neighbourhood_counts = df['Neighbourhood'].value_counts()

neighbourhood_order = neighbourhood_counts.index

plt.xlim(0,df['Neighbourhood'].value_counts().max())

# base_color = sb.color_palette()[3]

sb.countplot(data = df, y = 'Neighbourhood', hue='Present', order = neighbourhood_order[:15])

plt.xlabel('People counts')

plt.title('Top 15 Neighbourhoods who registered');
sb.countplot(data = df, y = 'Neighbourhood', hue='Present', order = neighbourhood_order[-15:])

# plt.xlim(0,df['Neighbourhood'].value_counts().max())

plt.xlabel('People counts')

plt.title('Bottom 15 Neighbourhoods who registered');

# df['Neighbourhood'].value_counts().max()
bin_size = np.arange(df['Age'].min(), df['Age'].max()+1, 1)

plt.hist(df['Age'], bins=bin_size);