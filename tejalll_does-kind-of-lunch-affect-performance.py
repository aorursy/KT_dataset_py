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
pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
#loading data to cook!

stud_filepath = "../input/students-performance-in-exams/StudentsPerformance.csv"

stud_data = pd.read_csv(stud_filepath)

stud_data #reviewing data to work with 
stud_data['total score'] = stud_data['math score'] + stud_data['reading score']+ stud_data['writing score']

sns.swarmplot(x=stud_data['lunch'],y=stud_data['total score'])
sns.catplot(x="lunch",y="total score",hue="race/ethnicity",kind="bar",data=stud_data)