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
udemy= pd.read_csv('/kaggle/input/udemy-courses/udemy_courses.csv')
udemy.head()
#checking the columns of the data set

for i in udemy:

    print(i)
#summary

udemy.describe()
#Checking missing values

udemy.isna().sum()
#Paid courses data frame

Paid=udemy[udemy['is_paid']==True]

Paid.head()
Paid.shape
Paid_total_num=Paid.shape[0]

Paid_total_num
Total_course_num=udemy.shape[0]

Total_course_num
#Percentages of Paid Courses

PercPaidCourses=(Paid_total_num/Total_course_num)*100

PercPaidCourses
Paid['num_subscribers'].max()
#The maximum subcribed course

Paid[Paid['num_subscribers']==121584]
udemy.nlargest(10, 'num_subscribers')
udemy.groupby(['subject']).mean()
udemy.groupby(['level']).mean()
#Most reviews

udemy.nlargest(5, 'num_reviews')