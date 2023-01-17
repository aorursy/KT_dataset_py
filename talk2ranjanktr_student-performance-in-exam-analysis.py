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
std_perfor = pd.read_csv("../input/StudentsPerformance.csv")
std_perfor.head()
#Changing the name of column
std_perfor.rename(columns={'math score':'math_score','reading score':'reading_score','writing score':'writing_score',
                          'test preparation course':'test_preparation_course','parental level of education':'parental_level_of_education'
                          ,'race/ethnicity':'group'},inplace=True)
std_perfor.head()
##### Histgram score of math
std_perfor.math_score.hist(bins=50)
##### Histgram score of reading
std_perfor.reading_score.hist(bins=50)
##### Histgram score of writing
std_perfor.writing_score.hist(bins=50)
#Minimum mark required to pass the exam
passmark = 40
#Math Pass or not pass status of student depend on gender
std_perfor['math_pass_status'] = np.where(std_perfor.math_score > passmark, 'P',"F")
print(std_perfor['math_pass_status'].value_counts())
sns.countplot(x=std_perfor.gender,hue=std_perfor.math_pass_status,palette='Set2')
#Math pass or not pass status of student depend of the level of education
sns.countplot(x=std_perfor.parental_level_of_education,hue=std_perfor.math_pass_status,palette='dark')
plt.xticks(rotation=90)
#Reading pass or not pass status of student depend on gender
std_perfor['reading_pass_status'] = np.where(std_perfor.reading_score > passmark , "P",'F')
print(std_perfor.reading_pass_status.value_counts())
sns.countplot(x=std_perfor.gender,hue=std_perfor.reading_pass_status,palette='bright')
#Reading pass or not pass of student depend of level of education
sns.countplot(x=std_perfor.parental_level_of_education,hue=std_perfor.reading_pass_status,palette='dark')
plt.xticks(rotation=80)
#Total percentage of student
std_perfor['total_mark'] = std_perfor.math_score + std_perfor.reading_score + std_perfor.writing_score
std_perfor['percentage'] = std_perfor['total_mark']/3
#Overall Percentage of male and female percentage
std_perfor.groupby('gender')['percentage'].mean().plot(kind='bar')
plt.legend()
#Percentage of student depend of level of education
std_perfor.groupby('parental_level_of_education')['percentage'].mean().plot(kind='bar')
#Overall percentage of student depend of gender and level of education
per_of_gender_level = std_perfor.groupby(['gender','parental_level_of_education'])['percentage'].mean().reset_index()
sns.barplot(x=per_of_gender_level.parental_level_of_education,y=per_of_gender_level.percentage,hue=per_of_gender_level.gender,palette='Accent')
plt.ylabel("Percentage")
plt.xticks(rotation=80)
plt.legend(loc='best')
#Number of student choose the education type  free or standard
std_perfor.lunch.value_counts().plot(kind='bar')
plt.legend()
data = std_perfor.groupby(['lunch','gender'])['percentage'].mean().reset_index()
sns.barplot(x='lunch',y='percentage',hue='gender',data=data,palette='Paired')
plt.legend(loc='best')
plt.ylabel("Percentage")
#Bar plot show that female has higher  percentage than male in standard and free education
std_perfor.group.value_counts().plot(kind='bar')
#Bar plot gender with group
bar_gen_group = std_perfor.groupby(['gender','group'])['percentage'].mean().reset_index()
sns.barplot(x='gender',y='percentage',hue='group',palette=sns.color_palette("Blues_r", 5),data=bar_gen_group)
plt.legend(loc='best')
plt.ylabel("Percentage")
