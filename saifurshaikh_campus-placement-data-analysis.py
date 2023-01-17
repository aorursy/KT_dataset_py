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
data = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
pd.options.display.max_columns = None

pd.options.display.max_rows = None

import warnings

warnings.filterwarnings('ignore')
data.head()
data.info()
data.salary.fillna(0, inplace=True)
data.head()
data.drop('sl_no', axis=1, inplace=True)
data.describe()
data.columns
data.rename(columns={'gender':'Gender', 'ssc_p':'SSC_Percentage', 'ssc_b':'SSC_Board', 'hsc_p':'HSC_Percentage',

                    'hsc_b':'HSC_Board', 'hsc_s':'HSC_Stream', 'degree_p':'Degree_Percentage', 'degree_t':'Degree_Topic',

                    'workex':'Work_Exp', 'etest_p':'EmpTest_Percentage', 'specialisation':'Specialisation',

                    'mba_p':'MBA_Percentage', 'status':'Status', 'salary':'Salary'}, inplace=True)
data.head()
data.Specialisation.value_counts()
import matplotlib.pyplot as plt

import seaborn as sns
data.groupby(['Specialisation'])['Status'].value_counts()
sns.countplot(x=data.Specialisation, data=data, hue=data.Status)
# From the above, we see that students from Marketing and Finance are most likely to get placed

# as compared to the ones from Marketing and HR. 
data.groupby(['Work_Exp'])['Status'].value_counts()
sns.countplot(x=data.Work_Exp, data=data, hue=data.Status)
# Here we see that even though prior work experience is not very much important to get placed

# However, the students who have prior work experience are less likely to get rejected
data.groupby(['SSC_Board'])['Status'].value_counts()
sns.countplot(x=data.SSC_Board, data=data, hue=data.Status)
data.groupby(['HSC_Board'])['Status'].value_counts()
# We see that the SSC or HSC Board doesn't matters to get placed
plt.figure(figsize=(12,7))

sns.scatterplot(x=data.HSC_Percentage, y=data.SSC_Percentage, data=data, hue=data.Status)

plt.show()
# The higher the percentage in both the classes, the more likely is a student to get placed. 
plt.figure(figsize=(12,7))

sns.scatterplot(x=data.HSC_Percentage, y=data.MBA_Percentage, data=data, hue=data.Status)

plt.show()
data.groupby(['HSC_Stream'])['Status'].value_counts()
# HSC stream doesn't play a very important role in getting placed

# However, if we check in terms of percenatge, Arts students are less likely to get placed. 
data_new = data.copy()

data_new['Status'] = np.where(data.Status=='Placed', 1, 0)
corr = data_new.corr()

plt.figure(figsize=(15,9))

sns.heatmap(corr, annot=True, cmap='RdYlGn', linewidths=0.5)

plt.show()
# We see that the most correlated column with Status is Salary. 

# Which is obvious because if the candidate is placed, only then they have a salary. 

# However, HSC and Degree percentage also influence a bit on getting placed. 
data.head()
data.groupby(['Degree_Topic'])['Status'].value_counts()
sns.countplot(x=data.Degree_Topic, data=data, hue=data.Status)
# Students from Commerce and Managements are the most to get placed. 

# While the students from any other Degree Specialisation are more likely to get rejected. 
sns.catplot(x=data.Status, y=data.EmpTest_Percentage, data=data)
sns.catplot(x=data.Status, y=data.MBA_Percentage, data=data)
data.groupby(['Status'])['MBA_Percentage'].mean()
data.groupby(['Status'])['EmpTest_Percentage'].mean()
# Let's see if the campus placement is Gender biased



data.groupby(['Gender'])['Status'].value_counts()
sns.countplot(x=data.Gender, hue=data.Status)