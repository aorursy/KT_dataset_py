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
import matplotlib.pyplot as plt



train = pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")



print(train.shape)

print("*"*50)

print(train.info())

print("*"*50)
pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

pd.set_option("display.max_colwidth", 10000)



train.head()
# regroup some high shcool with high school

train['parental level of education'] = train['parental level of education'].str.replace('some high school', 'high school')
# confirm statistics data for math score



train_math = train.drop(['reading score', 'writing score'], axis=1)



result_gender_math = train_math.groupby('gender').agg({'math score' : ['max', 'min', 'mean', 'median', 'var', 'std']}).reset_index()

result_gender_math.columns = ['gender', 'math_score_max', 'math_score_min', 'math_score_mean', 'math_score_median', 'math_score_var', 'math_score_std']

print(result_gender_math)

# male is higher averagely



result_race_ethnicity_math = train_math.groupby('race/ethnicity').agg({'math score' : ['max', 'min', 'mean', 'median', 'var', 'std']}).reset_index()

result_race_ethnicity_math.columns = ['race/ethnicity', 'math_score_max', 'math_score_min', 'math_score_mean', 'math_score_median', 'math_score_var', 'math_score_std']

print(result_race_ethnicity_math)

# group A is lowest and group E is highest the race/ethnicity has an oblivious difference



result_parental_level_of_education_math = train_math.groupby('parental level of education').agg({'math score' : ['max', 'min', 'mean', 'median', 'var', 'std']}).reset_index()

result_parental_level_of_education_math.columns = ['parental level of education', 'math_score_max', 'math_score_min', 'math_score_mean', 'math_score_median', 'math_score_var', 'math_score_std']

print(result_parental_level_of_education_math)

# master's degree is highest next is bachelor's degree and high school is lowest but the difference is not so much except high school



result_lunch_math = train_math.groupby('lunch').agg({'math score' : ['max', 'min', 'mean', 'median', 'var', 'std']}).reset_index()

result_lunch_math.columns = ['lunch', 'math_score_max', 'math_score_min', 'math_score_mean', 'math_score_median', 'math_score_var', 'math_score_std']

print(result_lunch_math)

# standard is higher and free/reduced varies widely



result_test_preparation_course_math = train_math.groupby('test preparation course').agg({'math score' : ['max', 'min', 'mean', 'median', 'var', 'std']}).reset_index()

result_test_preparation_course_math.columns = ['test preparation course', 'math_score_max', 'math_score_min', 'math_score_mean', 'math_score_median', 'math_score_var', 'math_score_std']

print(result_test_preparation_course_math)

# completed is higher averagely

# separate features and scores



train_x = train.drop(['math score', 'reading score', 'writing score'], axis=1)

train_y_1 = train['math score']

train_y_2 = train['reading score']

train_y_3 = train['writing score']
# label encode for features

from sklearn.preprocessing import LabelEncoder



for c in ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']:

    le = LabelEncoder()

    le.fit(train_x[c])

    train_x[c] = le.transform(train_x[c])
# calculate correlation coefficient for math score



import scipy.stats as st



corrs = []

for c in train_x.columns:

    corr = np.corrcoef(train_x[c], train_y_1)[0, 1]

    corrs.append(corr)

corrs = np.array(corrs)



idx = np.argsort(np.abs(corrs))[::-1]

top_cols_math, top_importances_math = train_x.columns.values[idx][:5], np.abs(corrs[idx])[:5]

print(top_cols_math, top_importances_math)
# visualize math score result



plt.figure(figsize=(20,10))

x_pos = range(0, 5)

plt.bar(x_pos, top_importances_math, tick_label = top_cols_math)

plt.show()
# confirm statistics data for reading score



train_reading = train.drop(['math score', 'writing score'], axis=1)



result_gender_reading = train_reading.groupby('gender').agg({'reading score' : ['max', 'min', 'mean', 'median', 'var', 'std']}).reset_index()

result_gender_reading.columns = ['gender', 'reading_score_max', 'reading_score_min', 'reading_score_mean', 'reading_score_median', 'reading_score_var', 'reading_score_std']

print(result_gender_reading)

# female is higher and varies widely



result_race_ethnicity_reading = train_reading.groupby('race/ethnicity').agg({'reading score' : ['max', 'min', 'mean', 'median', 'var', 'std']}).reset_index()

result_race_ethnicity_reading.columns = ['race/ethnicity', 'reading_score_max', 'reading_score_min', 'reading_score_mean', 'reading_score_median', 'reading_score_var', 'reading_score_std']

print(result_race_ethnicity_reading)

# group A is lowest and group E is highest the race/ethnicity has a moderate difference



result_parental_level_of_education_reading = train_reading.groupby('parental level of education').agg({'reading score' : ['max', 'min', 'mean', 'median', 'var', 'std']}).reset_index()

result_parental_level_of_education_reading.columns = ['parental level of education', 'reading_score_max', 'reading_score_min', 'reading_score_mean', 'reading_score_median', 'reading_score_var', 'reading_score_std']

print(result_parental_level_of_education_reading)

# master's degree is highest next is bachelor's degree and high school is lowest but the parental level of education has an oblivious difference



result_lunch_reading = train_reading.groupby('lunch').agg({'reading score' : ['max', 'min', 'mean', 'median', 'var', 'std']}).reset_index()

result_lunch_reading.columns = ['lunch', 'reading_score_max', 'reading_score_min', 'reading_score_mean', 'reading_score_median', 'reading_score_var', 'reading_score_std']

print(result_lunch_reading)

# standard is higher averagely



result_test_preparation_course_reading = train_reading.groupby('test preparation course').agg({'reading score' : ['max', 'min', 'mean', 'median', 'var', 'std']}).reset_index()

result_test_preparation_course_reading.columns = ['test preparation course', 'reading_score_max', 'reading_score_min', 'reading_score_mean', 'reading_score_median', 'reading_score_var', 'reading_score_std']

print(result_test_preparation_course_reading)

# completed is higher averagely
# calculate correlation coefficient for reading score



import scipy.stats as st



corrs = []

for c in train_x.columns:

    corr = np.corrcoef(train_x[c], train_y_2)[0, 1]

    corrs.append(corr)

corrs = np.array(corrs)



idx = np.argsort(np.abs(corrs))[::-1]

top_cols_reading, top_importances_reading = train_x.columns.values[idx][:5], np.abs(corrs[idx])[:5]

print(top_cols_reading, top_importances_reading)
# visualize reading score result



plt.figure(figsize=(20,10))

x_pos = range(0, 5)

plt.bar(x_pos, top_importances_reading, tick_label = top_cols_reading)

plt.show()
# confirm statistics data for writing score



train_writing = train.drop(['math score', 'reading score'], axis=1)



result_gender_writing = train_writing.groupby('gender').agg({'writing score' : ['max', 'min', 'mean', 'median', 'var', 'std']}).reset_index()

result_gender_writing.columns = ['gender', 'writing_score_max', 'writing_score_min', 'writing_score_mean', 'writing_score_median', 'writing_score_var', 'writing_score_std']

print(result_gender_writing)

# female is higher and varies widely



result_race_ethnicity_writing = train_writing.groupby('race/ethnicity').agg({'writing score' : ['max', 'min', 'mean', 'median', 'var', 'std']}).reset_index()

result_race_ethnicity_writing.columns = ['race/ethnicity', 'writing_score_max', 'writing_score_min', 'writing_score_mean', 'writing_score_median', 'writing_score_var', 'writing_score_std']

print(result_race_ethnicity_writing)

# group A is lowest and group E is highest the race/ethnicity has a moderate difference



result_parental_level_of_education_writing = train_writing.groupby('parental level of education').agg({'writing score' : ['max', 'min', 'mean', 'median', 'var', 'std']}).reset_index()

result_parental_level_of_education_writing.columns = ['parental level of education', 'writing_score_max', 'writing_score_min', 'writing_score_mean', 'writing_score_median', 'writing_score_var', 'writing_score_std']

print(result_parental_level_of_education_writing)

# master's degree is highest next is bachelor's degree and high school is lowest but the parental level of education has an oblivious difference



result_lunch_writing = train_writing.groupby('lunch').agg({'writing score' : ['max', 'min', 'mean', 'median', 'var', 'std']}).reset_index()

result_lunch_writing.columns = ['lunch', 'writing_score_max', 'writing_score_min', 'writing_score_mean', 'writing_score_median', 'writing_score_var', 'writing_score_std']

print(result_lunch_writing)

# standard is higher averagely



result_test_preparation_course_writing = train_writing.groupby('test preparation course').agg({'writing score' : ['max', 'min', 'mean', 'median', 'var', 'std']}).reset_index()

result_test_preparation_course_writing.columns = ['test preparation course', 'writing_score_max', 'writing_score_min', 'writing_score_mean', 'writing_score_median', 'writing_score_var', 'writing_score_std']

print(result_test_preparation_course_writing)

# completed is higher averagely

# calculate correlation coefficient for writing score



import scipy.stats as st



corrs = []

for c in train_x.columns:

    corr = np.corrcoef(train_x[c], train_y_3)[0, 1]

    corrs.append(corr)

corrs = np.array(corrs)



idx = np.argsort(np.abs(corrs))[::-1]

top_cols_writing, top_importances_writing = train_x.columns.values[idx][:5], np.abs(corrs[idx])[:5]

print(top_cols_writing, top_importances_writing)
# visualize writing score result



plt.figure(figsize=(20,10))

x_pos = range(0, 5)

plt.bar(x_pos, top_importances_writing, tick_label = top_cols_writing)

plt.show()