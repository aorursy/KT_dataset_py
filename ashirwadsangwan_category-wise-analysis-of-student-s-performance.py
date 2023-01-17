# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/StudentsPerformance.csv')
data.head()
data.info()
import warnings
warnings.filterwarnings('ignore')
sns.pairplot(data,hue = 'gender');

plt.figure(figsize = (15,30))
plt.subplot(311)
data['math score'].value_counts().sort_index().plot(kind = 'bar');
plt.xlabel('Math Score',fontsize = 15);
plt.ylabel('Number of Students',fontsize = 15);
plt.title('Distribution of Math Score',fontsize = 20);
plt.subplot(312)
data['reading score'].value_counts().sort_index().plot(kind = 'bar');
plt.xlabel('Reading Score',fontsize = 15);
plt.ylabel('Number of Students',fontsize = 15);
plt.title('Distribution of Reading Score',fontsize = 20);
plt.subplot(313)
data['writing score'].value_counts().sort_index().plot(kind = 'bar');
plt.xlabel('Writing Score',fontsize = 15);
plt.ylabel('Number of Students',fontsize = 15);
plt.title('Distribution of Writing Score',fontsize = 20);
data['math_result'] = np.where(data['math score']>40, 'Passed', 'Failed')
print(data['math_result'].value_counts())
plt.figure(figsize = (10,10))
data['math_result'].value_counts().plot(kind = 'pie');
plt.xlabel('Math Score',fontsize = 15);
plt.ylabel('');
plt.legend();
data['reading_result'] = np.where(data['reading score']>40, 'Passed', 'Failed')
print(data['reading_result'].value_counts())
plt.figure(figsize = (10,10))
data['reading_result'].value_counts().plot(kind = 'pie');
plt.xlabel('Reading Score',fontsize = 15);
plt.ylabel('');
plt.legend();
data['writing_result'] = np.where(data['writing score']>40, 'Passed', 'Failed')
print(data['writing_result'].value_counts())
plt.figure(figsize = (10,10))
data['writing_result'].value_counts().plot(kind = 'pie');
plt.xlabel('Writing Score',fontsize = 15);
plt.ylabel('');
plt.legend();
plt.figure(figsize = (12,7))
sns.countplot(x = 'math_result',hue = 'gender',data = data);
plt.xlabel('Math Result',fontsize = 15);
plt.figure(figsize = (12,7))
sns.countplot(x = 'reading_result',hue = 'gender',data = data);
plt.xlabel('Reading Result',fontsize = 15);
plt.figure(figsize = (12,7))
sns.countplot(x = 'writing_result',hue = 'gender',data = data);
plt.xlabel('Writing Result',fontsize = 15);
plt.figure(figsize = (12,7))
sns.countplot(x = 'math_result',hue = 'race/ethnicity',data = data);
plt.xlabel('Math Result',fontsize = 15);
plt.figure(figsize = (12,7))
sns.countplot(x = 'reading_result',hue = 'race/ethnicity',data = data);
plt.xlabel('Reading Result',fontsize = 15);
plt.figure(figsize = (12,7))
sns.countplot(x = 'writing_result',hue = 'race/ethnicity',data = data);
plt.xlabel('Writing Result',fontsize = 15);
plt.figure(figsize = (12,7))
sns.countplot(x = 'math_result',hue = 'parental level of education',data = data);
plt.xlabel('Math Result',fontsize = 15);
plt.figure(figsize = (12,7))
sns.countplot(x = 'reading_result',hue = 'parental level of education',data = data);
plt.xlabel('Reading Result',fontsize = 15);
plt.figure(figsize = (12,7))
sns.countplot(x = 'writing_result',hue = 'parental level of education',data = data);
plt.xlabel('Writing Result',fontsize = 15);
plt.figure(figsize = (12,7))
sns.countplot(x = 'math_result',hue = 'test preparation course',data = data);
plt.xlabel('Math Result',fontsize = 15);
plt.figure(figsize = (12,7))
sns.countplot(x = 'reading_result',hue = 'test preparation course',data = data);
plt.xlabel('Reading Result',fontsize = 15);
plt.figure(figsize = (12,7))
sns.countplot(x = 'writing_result',hue = 'test preparation course',data = data);
plt.xlabel('Writing Result',fontsize = 15);
