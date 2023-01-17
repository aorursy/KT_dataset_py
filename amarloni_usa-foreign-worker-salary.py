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
import pandas as  pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_excel('/kaggle/input/usa-foreign-workers-salary/USAforeignworkerssalarydata-1556559586172.xlsx')

data.head()
data.info()
data.columns
data = data.drop(['WORK_POSTAL_CODE', 'COUNTRY_OF_CITIZENSHIP', 'EXPERIENCE_REQUIRED_NUM_MONTHS',

                 'EDUCATION_LEVEL_REQUIRED', 'COLLEGE_MAJOR_REQUIRED', 'EXPERIENCE_REQUIRED_Y_N'], axis = 1)
object1 = data.select_dtypes(include=['object']).columns

object1
int_1 = data.select_dtypes(include=['int64']).columns

int_1
float_1 = data.select_dtypes(include=['float64']).columns

float_1
for column in object1:

    data[column].fillna(data[column].mode()[0], inplace = True)
for column in float_1:

    data[column].fillna(data[column].mean(), inplace = True)
for column in int_1:

    data[column].fillna(data[column].mean(), inplace = True)
data.info()
from sklearn.preprocessing import LabelEncoder, StandardScaler

le= LabelEncoder()

sc = StandardScaler()
#for column in object1:

#    data[column] = le.fit_transform(data[column])

    
#data.head()
plt.figure(figsize=(20,7))

sns.violinplot(x= 'VISA_CLASS', y= 'PAID_WAGE_PER_YEAR' , data = data, hue = 'FULL_TIME_POSITION_Y_N')
plt.figure(figsize=(25,7))

sns.scatterplot(x= 'WORK_STATE', y= 'PAID_WAGE_PER_YEAR' , data = data, hue = 'FULL_TIME_POSITION_Y_N')

plt.xticks(rotation=90)

plt.show()


plt.figure(figsize=(20,7))

sns.violinplot(x= 'JOB_TITLE_SUBGROUP', y= 'PAID_WAGE_PER_YEAR' , data = data, hue = 'FULL_TIME_POSITION_Y_N')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(20,7))

sns.violinplot(x= 'VISA_CLASS', y= 'PAID_WAGE_PER_YEAR' , data = data)
columns = ['CASE_STATUS', 'PREVAILING_WAGE_SUBMITTED_UNIT',

       'PAID_WAGE_SUBMITTED_UNIT', 'JOB_TITLE', 'WORK_STATE',

       'FULL_TIME_POSITION_Y_N', 'VISA_CLASS', 'JOB_TITLE_SUBGROUP']

for column in columns:

    plt.figure(figsize=(20,7))

    plt.xticks(rotation=90)

    sns.countplot(x=data[column], data = data)

    plt.show()