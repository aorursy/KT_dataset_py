
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
data.head(10)
data.isnull().sum()
data.info()
data.describe()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize = (20,10))
sns.countplot(data['math score'])
plt.xticks(rotation = 90)
data['math score'].value_counts()
data['math_pass'] = np.where(data['math score']>40, 'P', 'F')
data.math_pass.value_counts()
plt.figure(figsize = (10,10))
sns.countplot(data['parental level of education'], hue = data['math_pass'])
plt.xticks(rotation = 90)
data['overall_pass'] = data.apply(lambda x : 'F' if x['math score']<40 or x['reading score']<40 or x['writing score']<40 else 'P', axis = 1)
data['overall_pass'].value_counts()
plt.figure(figsize = (10,10))
sns.countplot(data['parental level of education'], hue = data['overall_pass'])
plt.xticks(rotation = 90)
data['Total Marks'] = data['math score'] + data['reading score'] + data['writing score']
data['Percentage'] = data['Total Marks']/3
plt.figure(figsize = (25,10))
sns.countplot(data['Percentage'])
plt.xticks(rotation = 90)
def getgrade(Percentage, overall_pass):
    if(overall_pass == 'F'):
        return 'F'
    if(Percentage>=80):
        return 'A'
    if(Percentage>=70):
        return 'B'
    if(Percentage>=60):
        return 'C'
    if(Percentage>=50):
        return 'D' 
    if(Percentage>=40):
        return 'E'    
    else:
        return 'F'
    
data['Grade'] = data.apply(lambda x: getgrade(x['Percentage'], x['overall_pass']), axis = 1)
data['Grade'].value_counts()
plt.figure(figsize =(10,10))
sns.countplot(data['parental level of education'], hue= data['Grade'])
plt.xticks(rotation = 90)