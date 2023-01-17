# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Firstly, we should read our data from dataset!
data = pd.read_csv('../input/StudentsPerformance.csv')
# Let's see info of data
data.info()
data.head()
data.tail(10)
data.describe()
# columns of data are:
data.columns
data.drop(['race/ethnicity','parental level of education','lunch'],axis=1,inplace=True)
# Let's see our new smallest data.
data.info()
data.columns
# Firstly, I want to see Gender situtation.
# Countplot:
sns.countplot(data['gender'])
# Pieplot
explode = (0.1, 0)
fig1, ax1 = plt.subplots(figsize=(5,5))
ax1.pie(data['gender'].value_counts(),explode=explode, labels=['female','male'], autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()
# Let's look at Test Preparation Course in the same way
sns.countplot(data['test preparation course'])
explode = (0.1, 0)
fig1, ax1 = plt.subplots(figsize=(5,5))
ax1.pie(data['test preparation course'].value_counts(),explode=explode, labels=['none','completed'], autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()
# Connection between gender and test prep. course
fig1, ax1 = plt.subplots(figsize=(7,5))
sns.countplot(data['gender'],hue=data['test preparation course'])
# Rename to some columns. I don't like the name of columns like 'math score'
data = data.rename(columns={'math score':'math_score'})
data = data.rename(columns={'reading score':'reading_score'})
data = data.rename(columns={'writing score':'writing_score'})
# Check it
data.columns
# Look at distribution of math score, reading score and writing score
sns.distplot(data['math_score'],bins=25)
sns.distplot(data['reading_score'],bins=25)
sns.distplot(data['writing_score'],bins=25)
# I want to calculate average of these scores and then I labeled every student as below average and above average.
average_math = data['math_score'].sum()/1000
average_reading = data['reading_score'].sum()/1000
average_writing = data['writing_score'].sum()/1000
# Print these values.
print("Average of math scores: {}".format(average_math))
print("Average of reading scores: {}".format(average_reading))
print("Average of writing scores: {}".format(average_writing))
# Check the scores of students and write '0' or '1' if it is below or above average.
# 1 for above
# 0 for below
# We should assign numerical values because pie plot needs integers.
data.math_score = [0 if each < average_math else 1 for each in data.math_score]
data.reading_score = [0 if each < average_reading else 1 for each in data.reading_score]
data.writing_score = [0 if each < average_writing else 1 for each in data.writing_score]
# Let's see our new data!
data.head()
# As you can see; our values are changed!
labels=['Female','Male']
explode = (0.1, 0)
fig1, ax1 = plt.subplots(figsize=(5,5))
ax1.pie(data.groupby('gender')['math_score'].sum(),explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  
plt.tight_layout()
plt.title("Gender-Math Score")
plt.legend()
plt.show()
labels=['Female','Male']
explode = (0.1, 0)
fig1, ax1 = plt.subplots(figsize=(5,5))
ax1.pie(data.groupby('gender')['reading_score'].sum(),explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  
plt.tight_layout()
plt.title("Gender-Reading Score")
plt.legend()
plt.show()
labels=['Female','Male']
explode = (0.1, 0)
fig1, ax1 = plt.subplots(figsize=(5,5))
ax1.pie(data.groupby('gender')['writing_score'].sum(),explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  
plt.tight_layout()
plt.title("Gender-Writing Score")
plt.legend()
plt.show()
above_math = data['math_score'] == 1 # First filter
above_reading = data['reading_score'] == 1 # Second filter
above_writing = data['writing_score'] == 1 # Third filter
ExcellentStudent = data[above_math & above_reading & above_writing] # And, Excellent Students!
ExcellentStudent
sns.countplot(ExcellentStudent['gender'])
# Of course female people most :)
sns.countplot(ExcellentStudent['test preparation course'])
# Most of Excellent Student didn't go any test preparation course.