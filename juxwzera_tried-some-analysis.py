# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/udemy-courses/udemy_courses.csv')

data.head()
data.shape
data.info()
data['content_duration'].unique()
data = data[data['content_duration'].str.contains('hour')]
data.head()
def ajuste_hour(hour):

    hora = hour.replace('hour','')

    hora1 = hora.replace('s','')

    output = float(hora1)

    return output
data['content_duration'] = data['content_duration'].apply(ajuste_hour)
data.info()
data['price'].unique()
def free_course(course):

    price = course.replace('Free','0')

    price1 = float(price)

    return price1
data['price'] = data['price'].apply(free_course)
data.info()
data['published_timestamp']= pd.to_datetime(data['published_timestamp'], format = '%Y-%m-%dT%H:%M:%SZ')
data.head()
data.info()
data['date'] = pd.to_datetime(data['published_timestamp'].dt.date, format = '%Y-%m-%d')

data.head()
categorical = [var for var in data.columns if data[var].dtype=='O']



print('There are {} categorical variables\n'.format(len(categorical)))



print('The categorical variables are :\n\n', categorical)
data['is_paid'].unique(), data['level'].unique(), data['subject'].unique()
data['is_paid'] = data['is_paid'].replace({'TRUE': 'True', 'FALSE':'False'})

data['is_paid'].unique()
# so we have 3 categorical data, we can explore these data here

# view the frequency 

for var in categorical:

    print(data[var].value_counts()/np.float(len(data)))
ax = data['is_paid'].value_counts().plot(kind ='bar',

                                        figsize = (10,8))



ax.set_title('Almost every course in udemy is paid', fontsize = 22)

ax.set_ylabel('Amount of courses', fontsize = 20)

ax.set_xlabel('Paid', fontsize = 20)
ax = data['subject'].value_counts().plot(kind ='bar',

                                        figsize = (10,8),

                                        color = 'g',

                                        width = 0.6,

                                        alpha = 0.6)



ax.set_title('Huge focus on web development and Business', fontsize = 22)

ax.set_ylabel('Amount of courses', fontsize = 20)

ax.set_xlabel('Courses', fontsize = 20)
ax = data['level'].value_counts().plot(kind ='bar',

                                        figsize = (10,8),

                                      width = 0.8,

                                      alpha= 0.7)



ax.set_title('Designed for everybody!', fontsize = 22)

ax.set_ylabel('Amount of courses', fontsize = 20)

ax.set_xlabel('Levels', fontsize = 20)
grouped = data.groupby(['level','subject'])
grouped_pct = grouped['course_id']
grouped_pct.agg('describe')
# now we are going to explore numerical data.

numerical = [var for var in data.columns if data[var].dtype!='O']



print('There are {} numerical variables\n'.format(len(numerical)))



print('The numerical variables are :', numerical)
data.corr().round(4)
ax = sns.boxplot(data['price'], orient = 'v', width = 0.3)

ax.figure.set_size_inches(12,6)

ax.set_ylabel('Price', fontsize = 20)

ax.set_title('Distribuition from courses', fontsize  = 24)
ax = sns.distplot(data['price'])

ax.figure.set_size_inches(12,6)

ax.set_xlabel('Prices')

ax.set_title('Distribution', fontsize = 22)
ax = sns.boxplot(x = 'level',y = 'price', data = data, orient = 'v', width = 0.3)

ax.figure.set_size_inches(12,6)

ax.set_ylabel('Price', fontsize = 20)

ax.set_title('Distribution of price with level', fontsize  = 24)

ax.set_xlabel('Level', fontsize= 20)
ax = sns.boxplot(x = 'subject',y = 'price', data = data, orient = 'v', width = 0.3)

ax.figure.set_size_inches(12,6)

sns.set_palette('CMRmap')

ax.set_ylabel('Price', fontsize = 20)

ax.set_title('Distribution of price with subjects', fontsize  = 24)

ax.set_xlabel('Subjects', fontsize= 20)
ax = sns.pairplot(data, y_vars = 'price', x_vars = ['num_subscribers', 'num_reviews', 'num_lectures','content_duration'], height = 5, kind = 'reg')

ax.fig.suptitle('Scatter plot with variables', fontsize=20, y=1.05)

ax
data['log_price'] = np.log(data['price']+1)

data['log_num_subscribers'] = np.log(data['num_subscribers'])

data['log_num_reviews'] = np.log(data['num_reviews'] + 1)

data['log_num_lectures'] = np.log(data['num_lectures'])

data['log_content_duration'] = np.log(data['content_duration'])
data.head()
## Filtering only paid courses

data_new = data[data['is_paid'] == 'True']
data_new.head()
ax = sns.distplot(data_new['log_price'], bins =15)

ax.figure.set_size_inches(12,6)

ax.set_xlabel('Price')

ax.set_title('Distribution of log_price', fontsize = 22)
group_subject = data_new.groupby('subject')['price']
type(group_subject)
group_subject.groups
group_subject.mean()
Q1 = group_subject.quantile(.25)

Q3 = group_subject.quantile(.75)

IIQ = Q3 - Q1 #intervalo interquartilico

lower_limit = Q1 - 1.5*IIQ

upper_limit = Q3 + 1.5*IIQ
data_new1 = pd.DataFrame()



for subject in group_subject.groups.keys():

    is_subject = data_new['subject'] == subject

    is_limit = (data_new['price'] >= lower_limit[subject]) & (data_new['price'] <= upper_limit[subject])

    selection = is_subject & is_limit

    data_select = data_new[selection]

    data_new1 = pd.concat([data_new1, data_select])
ax = sns.boxplot(x = 'subject',y = 'price', data = data_new1, orient = 'v', width = 0.3)

ax.figure.set_size_inches(12,6)

sns.set_palette('CMRmap')

ax.set_ylabel('Preço', fontsize = 20)

ax.set_title('Preço dos cursos', fontsize  = 24)

ax.set_xlabel('Subjects', fontsize= 20)
ax = sns.distplot(data_new1['log_price'], bins =15)

ax.figure.set_size_inches(12,6)

ax.set_xlabel('Price')

ax.set_title('Distribution of price', fontsize = 22)