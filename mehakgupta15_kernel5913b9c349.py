# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#reading data in variable data

data = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')

#observing data

data

#Observation:

#1. Gender,race/ethnicity,parental education,lunch,test preparation course are categorical values

#2. Whereas math score,reading score,writing score are numerical values
# task-1 Correlation exist between gender and reading score

sns.catplot(x='gender',y='reading score',data = data)
#to observe whether correlation exist between reading score,math score and writing score

sns.heatmap(data.corr(),cbar = False,annot=True,)
#correlation between reading score and writing score based on race/ethnicity

sns.relplot(x='reading score',y='writing score',hue='race/ethnicity',data=data)
#distribution plot for maths score

sns.distplot(data['math score'],bins=5)



#observation

#maximum student have score between 60 to 80 in maths
#distribution plot for reading score

sns.distplot(data['reading score'],bins=5)



#observation

#maximum student have score between 60 to 80 in reading
#distribution plot for maths score

sns.distplot(data['writing score'],bins=5)



#observation

#maximum student have score between 65 to 80 in writing
#To get relation between students prepared for course and math score on the basis of gender



sns.barplot(x='test preparation course',y ='math score',hue='gender',data=data)