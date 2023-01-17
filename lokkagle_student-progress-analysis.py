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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
data.head()
data.isna().sum()
data.info()
data.describe().T
data.select_dtypes(include= np.object).nunique()
plt.style.use('seaborn')
data.select_dtypes(include= np.int64).hist(figsize = (15,5))
plt.show()
plt.style.use('fivethirtyeight')

for i in data[list(data.select_dtypes(include= np.object).columns)[0]].unique():
    data[data['gender'] == i].select_dtypes(include = np.int64).hist(figsize = (15,5))
    print('data distribution relates to gender : {}'.format(i))
plt.show()
for i in data[list(data.select_dtypes(include= np.object).columns)[1]].unique():
    data[data['race/ethnicity'] == i].select_dtypes(include = np.int64).hist(figsize = (15,5))
    print('data distribution relates to groups : {}'.format(i))
plt.show()
for i in data[list(data.select_dtypes(include= np.object).columns)[2]].unique():
    data[data['parental level of education'] == i].select_dtypes(include = np.int64).hist(figsize = (15,5))
    print('data distribution relates to parental level of education : {}'.format(i))
plt.show()
for i in data[list(data.select_dtypes(include= np.object).columns)[-1]].unique():
    data[data['test preparation course'] == i].select_dtypes(include = np.int64).hist(figsize = (15,5))
    print('data distribution relates to test preparation course : {}'.format(i))
plt.show()
data.columns
for i in ['reading score','writing score']:
    sns.relplot(x = 'math score', y = i, data = data, hue = 'gender')
plt.show()
for i in ['reading score','writing score']:
    sns.relplot(x = 'math score', y = i, data = data, hue = 'race/ethnicity')
plt.show()
for i in ['reading score','writing score']:
    sns.relplot(x = 'math score', y = i, data = data, hue = 'parental level of education')
plt.show()
for i in ['reading score','writing score']:
    sns.relplot(x = 'math score', y = i, data = data, hue = 'test preparation course')
plt.show()
for i in data.select_dtypes(include= np.object).columns:
    sns.countplot(x = i, data = data )
    plt.show()
# inferential stats
# probability of students getting more than 50 % marks
def get_students_math_marks_morethan50(data):
    total_students = data.shape[0]
    students_getting_marks_morethan50 = (data[data['math score'] > 50].shape[0] / total_students)* 100
    return print('students getting more than 50% marks in math : {}'.format(students_getting_marks_morethan50))

def get_students_reading_score_morethan50(data):
    total_students = data.shape[0]
    students_getting_marks_morethan50_read = (data[data['reading score'] > 50].shape[0] / total_students)* 100
    return print('students getting more than 50% marks in reading : {}'.format(students_getting_marks_morethan50_read))

def get_students_writing_score_morethan50(data):
    total_students = data.shape[0]
    students_getting_marks_morethan50_write = (data[data['writing score'] > 50].shape[0] / total_students)* 100
    return print('students getting more than 50% marks in writing : {}'.format(students_getting_marks_morethan50_write))

#####--------------------------


def get_students_math_marks_lessthan50(data):
    total_students = data.shape[0]
    students_getting_marks_lessthan50 = (data[data['math score'] < 50].shape[0] / total_students)* 100
    return print('students getting less than 50% marks in math : {}'.format(students_getting_marks_lessthan50))

def get_students_reading_score_lessthan50(data):
    total_students = data.shape[0]
    students_getting_marks_lessthan50_read = (data[data['reading score'] < 50].shape[0] / total_students)* 100
    return print('students getting less than 50% marks in reading : {}'.format(students_getting_marks_lessthan50_read))

def get_students_writing_score_lessthan50(data):
    total_students = data.shape[0]
    students_getting_marks_lessthan50_write = (data[data['writing score'] < 50].shape[0] / total_students)* 100
    return print('students getting less than 50% marks in writing : {}'.format(students_getting_marks_lessthan50_write))
######----------------------------

def get_students_math_marks_morethan90(data):
    total_students = data.shape[0]
    students_getting_marks_morethan90 = (data[data['math score'] > 90].shape[0] / total_students)* 100
    return print('students getting more than 90% marks in math : {}'.format(students_getting_marks_morethan90))

def get_students_reading_score_morethan90(data):
    total_students = data.shape[0]
    students_getting_marks_morethan90_read = (data[data['reading score'] > 90].shape[0] / total_students)* 100
    return print('students getting more than 90% marks in reading : {}'.format(students_getting_marks_morethan90_read))

def get_students_writing_score_morethan90(data):
    total_students = data.shape[0]
    students_getting_marks_morethan90_write = (data[data['writing score'] > 90].shape[0] / total_students)* 100
    return print('students getting more than 90% marks in writing : {}'.format(students_getting_marks_morethan90_write))
get_students_math_marks_morethan50(data)
get_students_reading_score_morethan50(data)
get_students_writing_score_morethan50(data)
get_students_math_marks_lessthan50(data)
get_students_reading_score_lessthan50(data)
get_students_writing_score_lessthan50(data)
get_students_math_marks_morethan90(data)
get_students_reading_score_morethan90(data)
get_students_writing_score_morethan90(data)
total_students = data.shape[0]
number_of_students_passing_in_all_subjects = data[(data['math score'] > 40) &
                                                  (data['writing score'] > 40) & 
                                                  (data['reading score'] > 40)].shape[0]
probability_of_students_passing_in_all_the_subjects = (number_of_students_passing_in_all_subjects/total_students)*100
print("The Probability of Students Passing in all the Subjects is {0:.2f} %".format(probability_of_students_passing_in_all_the_subjects))

total_students = data.shape[0]
number_of_students_scoring_more_than_90 = data[(data['math score'] > 90) &
                                                  (data['writing score'] > 90) & 
                                                  (data['reading score'] > 90)].shape[0]

probability_of_students_scoring_more_than_90_in_all_subjects = (number_of_students_scoring_more_than_90/total_students)*100
print("The Probability of Students Passing in all the Subjects is {0:.2f} %".
      format(probability_of_students_scoring_more_than_90_in_all_subjects))
# confidence interval for math

import scipy.stats as stats
import math

# lets seed the random values
np.random.seed(10)

# lets take a sample size
sample_size = 1000
sample = np.random.choice(a= data['math score'],
                          size = sample_size)
sample_mean = sample.mean()

# Get the z-critical value*
z_critical = stats.norm.ppf(q = 0.95)  

 # Check the z-critical value  
print("z-critical value: ",z_critical)                                

# Get the population standard deviation
pop_stdev = data['math score'].std()  

# checking the margin of error
margin_of_error = z_critical * (pop_stdev/math.sqrt(sample_size)) 

# defining our confidence interval
confidence_interval = (sample_mean - margin_of_error,
                       sample_mean + margin_of_error)  

# lets print the results
print("Confidence interval:",end=" ")
print(confidence_interval)
print("True mean: {}".format(data['math score'].mean()))
# confidence interval for reading

# lets seed the random values
np.random.seed(10)

# lets take a sample size
sample_size = 1000
sample = np.random.choice(a= data['reading score'],
                          size = sample_size)
sample_mean = sample.mean()

# Get the z-critical value*
z_critical = stats.norm.ppf(q = 0.95)  

 # Check the z-critical value  
print("z-critical value: ",z_critical)                                

# Get the population standard deviation
pop_stdev = data['reading score'].std()  

# checking the margin of error
margin_of_error = z_critical * (pop_stdev/math.sqrt(sample_size)) 

# defining our confidence interval
confidence_interval = (sample_mean - margin_of_error,
                       sample_mean + margin_of_error)  

# lets print the results
print("Confidence interval:",end=" ")
print(confidence_interval)
print("True mean: {}".format(data['reading score'].mean()))
## confidence interval for writing score

np.random.seed(10)

# lets take a sample size
sample_size = 1000
sample = np.random.choice(a= data['writing score'],
                          size = sample_size)
sample_mean = sample.mean()

# Get the z-critical value*
z_critical = stats.norm.ppf(q = 0.95)  

 # Check the z-critical value  
print("z-critical value: ",z_critical)                                

# Get the population standard deviation
pop_stdev = data['writing score'].std()  

# checking the margin of error
margin_of_error = z_critical * (pop_stdev/math.sqrt(sample_size)) 

# defining our confidence interval
confidence_interval = (sample_mean - margin_of_error,
                       sample_mean + margin_of_error)  

# lets print the results
print("Confidence interval:",end=" ")
print(confidence_interval)
print("True mean: {}".format(data['writing score'].mean()))
for i in ['math score','reading score', 'writing score']:
    print('results for :{}'.format(i))
    print(data.groupby('gender')[i].agg(['min', 'mean','max']), '\n')
for i in ['math score','reading score', 'writing score']:
    print('results for :{}'.format(i))
    print(data.groupby(['race/ethnicity','gender'])[i].agg(['min', 'mean','max']), '\n')
for i in ['math score','reading score', 'writing score']:
    print('results for :{}'.format(i))
    print(data.groupby(['parental level of education','gender'])[i].agg(['min', 'mean','max']), '\n')
for i in ['math score','reading score', 'writing score']:
    print('results for :{}'.format(i))
    print(data.groupby(['lunch','gender'])[i].agg(['min', 'mean','max']), '\n')
for i in ['math score','reading score', 'writing score']:
    print('results for :{}'.format(i))
    print(data.groupby(['test preparation course','gender'])[i].agg(['min', 'mean','max']), '\n')
data.columns
for i in ['race/ethnicity', 'parental level of education', 'lunch','test preparation course']:
    pd.crosstab( data[i], data['gender']).plot(kind = 'bar',figsize = (15,5))
plt.show()
for i in ['race/ethnicity', 'parental level of education', 'lunch','test preparation course']:
    pd.crosstab( data[i], data['gender']).plot(kind = 'bar',figsize = (15,5), stacked = True)
plt.show()
