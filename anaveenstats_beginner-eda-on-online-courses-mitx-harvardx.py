# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import seaborn as sns
from sklearn import preprocessing
import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt  
matplotlib.style.use('ggplot')
%matplotlib inline
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
course_list=pd.read_csv('../input/appendix.csv')
course_list1=course_list[['% Male','Course Title']].sort_values(by='% Male',ascending=False).head()
course_list1.set_index('Course Title',inplace=True)
plt.xticks(rotation=90)
plt.plot(course_list1)
course_list1
course_list1=course_list[['% Female','Course Title']].sort_values(by='% Female',ascending=False).head()
course_list1.set_index('Course Title',inplace=True)
plt.xticks(rotation=90)
plt.plot(course_list1)
course_list1
course_list.groupby('Course Subject')['Course Title'].count().plot(kind='bar')
course_list.groupby('Course Subject')['Participants (Course Content Accessed)'].sum().plot(kind='bar')
