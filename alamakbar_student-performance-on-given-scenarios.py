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
data = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
data.head()
#first creating a total score column of each subject.
data['total score'] = data['math score'] + data['reading score'] + data ['writing score']
data.head()
data.info()
#finding the average score among all subjects of both male and female also the average of total 3 subjects

print('Average Math Score:', np.mean(data['math score']))
print('Average Reading Score:', np.mean(data['reading score']))
print('Average Writing Score:', np.mean(data['writing score']))
print('Total Average of all subjects:', np.mean((data['math score']+data['reading score']+data['writing score'])/ 3))

#checking for null values in data
data.isnull().sum()
male = data[data['gender'] == 'male']
female = data[data['gender'] == 'female']
print("Average Score of male")
print("maths score:", male['math score'].mean())
print("reading score:", male['reading score'].mean())
print("writing score:", male['writing score'].mean())
print("Total Score:", (male['total score']/3).mean())
print("Average Score of Female")
print("maths score:", female['math score'].mean())
print("reading score:", female['reading score'].mean())
print("writing score:", female['writing score'].mean())
print("Total Score:", (female['total score']/3).mean())
data['test preparation course'].unique() #we have two value in test preparation 'none' and 'completed'
total_average_score = np.mean(data['total score'] / 3)
print(total_average_score)
data.groupby('test preparation course')['math score'].mean()
#df.groupby('Gender')['Math'].mean()
data.groupby('test preparation course')['reading score'].mean()
data.groupby('test preparation course')['writing score'].mean()
data['parental level of education'].unique()
data.groupby('parental level of education')['math score'].agg(['mean', 'max', 'min'])
data.groupby('parental level of education')['writing score'].agg(['mean', 'max', 'min'])
data.groupby('parental level of education')['reading score'].agg(['mean', 'max', 'min'])
