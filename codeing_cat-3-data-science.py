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
#1. Read the dataset
import pandas as pd
import seaborn as sns
df=pd.read_csv("../input/human-resources-data-set/HRDataset_v13.csv")
df.head()
#2. Describe the dataset
df.describe()
#3. Find mean,median and mode of columns.

df['PayRate'].mean()
df['PayRate'].median()
df['PayRate'].mode()
# 4. Find the distribution of columns with numerical data. Evaluate if they are normal or not

df.dtypes
# from above we can figure out the column with numerical data i.e dayslatelast30,SpecialProjectsCount,EmpSatisfaction ,EngagementSurvey etc
sns.distplot(df['MarriedID']) #cheking distribution 
sns.distplot(df['EmpSatisfaction'])
sns.distplot(df['EngagementSurvey'])
 #Draw different types of graphs to represent information hidden in the dataset.
print(df['Sex'].value_counts())
df['Sex'].value_counts().plot(kind='bar')
# count of people working at the position
count=df.groupby(df["Position"]).count()
count = pd.DataFrame(count.to_records())
count = count['Position']
count
# graph  to show emploies woring in particular position

plt.figure(figsize=(15,10))
sns.countplot(y='Position', data=df,order=count )
#from the above graph we can find,,,post employies are working at Production technician 1

df['PayRate'].plot.hist()
# graph to represent males and females in paricular field
plt.figure(figsize=(16,5))
sns.countplot(x=df['Department'],hue=df['Sex'])
# pic to show maritial status of emploies working
plt.figure(figsize=(10,6))
df['MaritalDesc'].value_counts().plot(kind='pie',autopct='%0.1f%%')
#its shows 42.2% employee(max) are still single,,,,
#6. Find columns which are correlated.

df.corr().columns


df.corr()
#  7. Find columns which are not correlated.


comparison_column = np.where(df["col1"] == df["col2"], True, False)
compare 