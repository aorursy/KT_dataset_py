# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
my_filepath = "../input/jobs-applied-in-linkedin-with-easyapplybot/jobs.csv"



my_data = pd.read_csv(my_filepath,encoding='latin1')



my_data.head()
non_duplicated = my_data.drop_duplicates(['jobID'])



print("Jobs found: ", len(non_duplicated['result']))

print("Successfully sent applications: ", non_duplicated['result'].value_counts()[1])

print("Failed applications: ", non_duplicated['result'].value_counts()[0])
import datetime

non_duplicated['date'] = non_duplicated.apply(lambda x: datetime.datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S.%f').date(), axis = 1)

non_duplicated['time'] = non_duplicated.apply(lambda x: datetime.datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S.%f').time(), axis = 1)
non_duplicated = non_duplicated.drop('timestamp', axis=1)
non_duplicated['day'] = non_duplicated.apply(lambda x: x['date'].day, axis = 1)

non_duplicated['weekday'] = non_duplicated.apply(lambda x: x['date'].weekday(), axis = 1)

non_duplicated['month'] = non_duplicated.apply(lambda x: x['date'].month, axis = 1)

non_duplicated['week'] = non_duplicated.apply(lambda x: x['date'].isocalendar()[1], axis = 1)
def convert_result(row):

    if row.result == True:

        row.result = 1

    else:

        row.result = 0

    return row



def convert_att(row):

    if row.attempted == True:

        row.attempted = 1

    else:

        row.attempted = 0

    return row



non_duplicated = non_duplicated.apply(convert_result, axis = 1)

non_duplicated = non_duplicated.apply(convert_att, axis = 1)
non_duplicated.head()
group = non_duplicated[['date','attempted', 'result']].groupby(['date']).sum()



group.head()
plt.figure(figsize=(37,10))

plt.title("Attempts per day")



sns.barplot(x=group.index, y=group['attempted'])
plt.figure(figsize=(37,10))

plt.title("Submitted applications per day")



sns.barplot(x=group.index, y=group['result'])
group2 = non_duplicated[['weekday','attempted', 'result']].groupby(['weekday']).sum()



group2.head()
plt.figure(figsize=(17,6))

plt.title("Attempts per weekday")



sns.barplot(x=group2.index, y=group2['attempted'])
plt.figure(figsize=(17,6))

plt.title("Submitted applications per weekday")



sns.barplot(x=group2.index, y=group2['result'])
failed = group2['attempted'] - group2['result']



plt.figure(figsize=(17,6))



plt.title("Failed attemps per weekday")



sns.barplot(x=failed.index, y=failed)
#2019339369

non_duplicated['date_call'] = np.nan



print(non_duplicated.query('jobID == 2019339369')['date_call']) # id is equal to 92

non_duplicated.loc[92,'date_call'] = datetime.date(2020, 9, 14)

print(non_duplicated.query('jobID == 2164722520')['date_call']) # id is equal to 950

non_duplicated.loc[950,'date_call'] = datetime.date(2020, 10, 8)
call = non_duplicated['date_call'].notnull().sum()

d_call = non_duplicated['result'].value_counts()[1] - call



p_call = call*100/d_call

p_dcall = 100 - p_call



# Creating dataset 

labels = "Did call", "Didn't call"

  

data = [p_call, p_dcall]

  

# Creating plot 

fig = plt.figure(figsize =(12, 8)) 

plt.pie(data, labels = labels, autopct='%1.1f%%', startangle=90) 

  

# show plot 

plt.show() 



pd.DataFrame({'Total applications':[non_duplicated['result'].value_counts()[1]], 'Number of calls':[call]}).head()

per_week = non_duplicated[['attempted', 'result', 'week']].groupby(['week']).sum()

print(per_week)
per_week.loc[(per_week.index > 37) & (per_week.index < 42)]