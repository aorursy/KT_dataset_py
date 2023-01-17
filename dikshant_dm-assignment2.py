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
!pip install mlxtend  
from mlxtend.frequent_patterns import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df =pd.read_csv("../input/dataaa/Employee_skills_traits.csv")
df.info()
df.shape
df.columns
df_plot = df
df_plot.info()
df_plot.rename(str.strip, axis='columns', inplace=True)
sns.distplot(df_plot['Age'])

df_plot['frontend'] = df_plot['HTML CSS Java Script']
df_plot['backend'] = df_plot['.Net'] + df_plot['SQL Server'] + df_plot['PHP mySQL']
df_plot['backend'].unique()
df_plot['backend'] = df_plot['backend'].apply(lambda x: 1 if x>=1 else 0)

df_plot.drop(['.Net', 'SQL Server', 'HTML CSS Java Script', 'PHP mySQL'], axis=1, inplace=True)
df_plot.drop('ID', axis=1, inplace=True)
df_plot.shape
fig,a =  plt.subplots(3,3, figsize=(15,15))
sns.boxplot(y=df_plot['Age'], x=df_plot[df_plot.columns[0]], ax=a[0][0])
sns.boxplot(y=df_plot['Age'], x=df_plot[df_plot.columns[1]], ax=a[0][1])
sns.boxplot(y=df_plot['Age'], x=df_plot[df_plot.columns[2]], ax=a[0][2])
sns.boxplot(y=df_plot['Age'], x=df_plot[df_plot.columns[3]], ax=a[1][0])
sns.boxplot(y=df_plot['Age'], x=df_plot[df_plot.columns[5]], ax=a[1][1])
sns.boxplot(y=df_plot['Age'], x=df_plot[df_plot.columns[6]], ax=a[1][2])
sns.boxplot(y=df_plot['Age'], x=df_plot[df_plot.columns[7]], ax=a[2][0])
sns.boxplot(y=df_plot['Age'], x=df_plot[df_plot.columns[8]], ax=a[2][1])
sns.boxplot(y=df_plot['Age'], x=df_plot[df_plot.columns[9]], ax=a[2][2])
fig,a =  plt.subplots(3,3, figsize=(15,15))
sns.boxplot(y=df_plot['Employment period'], x=df_plot[df_plot.columns[0]], ax=a[0][0])
sns.boxplot(y=df_plot['Employment period'], x=df_plot[df_plot.columns[1]], ax=a[0][1])
sns.boxplot(y=df_plot['Employment period'], x=df_plot[df_plot.columns[2]], ax=a[0][2])
sns.boxplot(y=df_plot['Employment period'], x=df_plot[df_plot.columns[3]], ax=a[1][0])
sns.boxplot(y=df_plot['Employment period'], x=df_plot[df_plot.columns[5]], ax=a[1][1])
sns.boxplot(y=df_plot['Employment period'], x=df_plot[df_plot.columns[6]], ax=a[1][2])
sns.boxplot(y=df_plot['Employment period'], x=df_plot[df_plot.columns[7]], ax=a[2][0])
sns.boxplot(y=df_plot['Employment period'], x=df_plot[df_plot.columns[8]], ax=a[2][1])
sns.boxplot(y=df_plot['Employment period'], x=df_plot[df_plot.columns[9]], ax=a[2][2])
fig,a =  plt.subplots(3,3, figsize=(15,15))
sns.boxplot(y=df_plot['Time in current department'], x=df_plot[df_plot.columns[0]], ax=a[0][0])
sns.boxplot(y=df_plot['Time in current department'], x=df_plot[df_plot.columns[1]], ax=a[0][1])
sns.boxplot(y=df_plot['Time in current department'], x=df_plot[df_plot.columns[2]], ax=a[0][2])
sns.boxplot(y=df_plot['Time in current department'], x=df_plot[df_plot.columns[3]], ax=a[1][0])
sns.boxplot(y=df_plot['Time in current department'], x=df_plot[df_plot.columns[5]], ax=a[1][1])
sns.boxplot(y=df_plot['Time in current department'], x=df_plot[df_plot.columns[6]], ax=a[1][2])
sns.boxplot(y=df_plot['Time in current department'], x=df_plot[df_plot.columns[7]], ax=a[2][0])
sns.boxplot(y=df_plot['Time in current department'], x=df_plot[df_plot.columns[8]], ax=a[2][1])
sns.boxplot(y=df_plot['Time in current department'], x=df_plot[df_plot.columns[9]], ax=a[2][2])
df_plot['Gender'].value_counts()
# plotting a pie chart

size = [514, 484]
labels = "Male", "Female"
colors = ['yellow', 'orange']
explode = [0, 0.1]

plt.rcParams['figure.figsize'] = (8, 8)
plt.pie(size, labels = labels, colors = colors, explode = explode, shadow = True, autopct = "%.2f%%")
plt.title('A Pie Chart Representing GenderGap', fontsize = 30)
plt.axis('off')
plt.legend()
plt.show()
df =pd.read_csv("../input/dataaa/Employee_skills_traits.csv")
df.rename(str.strip, axis='columns', inplace=True)
df.columns
#check for missing values
df.isna().any()
# get the data types for each of the columns
df.info()
#duplicateRowsDF = df[df.duplicated(['ID'])]
# sorting by first name 
df.sort_values("ID", inplace = True) 
  
# dropping ALL duplicte values 
df.drop_duplicates(subset ="ID", 
                     keep='first', inplace = True) 
  
# Run the describe function to get a overview of distribution of data
df.describe()
bins = [23,34,44,56]
labels = ['Young','Mid-Aged','Old']
df['Age'] = pd.cut(df['Age'],bins= bins,labels=labels )
df.info()

# df['Employment period'].unique()
    
bins = [0,3,10,15,20]
labels = ['Fresher','Experienced','Professional','Senior']
df['Exp'] = pd.cut(df['Employment period'],bins= bins,labels=labels )
df.info()
#dummies = pd.get_dummies(df[['Age']], drop_first=True)
y = pd.get_dummies(df.Age )
df = pd.concat([df, y], axis=1)
print(y.head())


y_exp = pd.get_dummies(df.Exp )
df = pd.concat([df, y_exp], axis=1)
print(y_exp.head())

   
bins = [0,3,8,12]
labels = ['Newly','oldInDep','Veteran']
df['CurrentCompany'] = pd.cut(df['Time in current department'],bins= bins,labels=labels )
df.info()
y_CurComp = pd.get_dummies(df.CurrentCompany )
df = pd.concat([df, y_CurComp], axis=1)

df.info()
# Dropping the columns which are categorical and would not be required further for our analysis
df =  df.drop(['Age','Time in current department','Employment period', 'ID','Exp','CurrentCompany'], axis = 1) 

# Converting all of the remaining columns from 0/1 to Boolean 
df = df.astype(bool)
df.head()

#from mlxtend.frequent_patterns import *

frq_items_fpgrowth = fpgrowth(df, min_support = 0.250, use_colnames = True) 
  
# Collecting the inferred rules in a dataframe 
rules_fpgrowth = association_rules(frq_items_fpgrowth, metric ="lift", min_threshold = 1) 
rules_fpgrowth = rules_fpgrowth.sort_values(['confidence', 'lift'], ascending =[False, False])
rules_fpgrowth
fi = apriori(df, min_support=0.01, use_colnames=True, max_len=4)
#Finding the combination for 3 itemsets
fi['length'] = fi['itemsets'].apply(lambda x: len(x))
fi[ (fi['length'] >= 3) &(fi['support'] >= 0.15) ]
rulesn = association_rules(fi, metric ="lift", min_threshold = 1) 
rulesn.head(20).sort_values(by='confidence', ascending=False)
