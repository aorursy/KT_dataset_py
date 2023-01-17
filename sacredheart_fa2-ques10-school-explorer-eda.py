# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import csv

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
school = pd.read_csv("../input/2016 School Explorer.csv")
school.head()
import matplotlib.pyplot as plt
import matplotlib
import plotly.plotly as py
import seaborn as sns 
sns.set(style="darkgrid")
plt.figure(figsize=(15,10))
descending_order = school["City"].value_counts().sort_values(ascending=False).index
graph = sns.countplot(x="City",data=school,order=descending_order)

graph.set_xticklabels(graph.get_xticklabels(),fontsize=15,rotation=60, ha="right")
plt.tight_layout()
plt.show()
from scipy import stats
sns.set(color_codes=True)

#plt.figure(figsize=(15,5)

school.hist(column ="Percent Asian",by = "Community School?",bins=20)
plt.show()

school = school.replace('N/A', np.nan)
school.iloc[50:64,10:26]
#Replacing NA values with NaN


#Dropping the NaN value rows as there are only few
school.dropna(subset=['Percent of Students Chronically Absent'],inplace=True)
school.iloc[50:64,10:26]
#Spliitig %age
school['Percent of Students Chronically Absent'] = school['Percent of Students Chronically Absent'].apply(lambda x: x.split('%')[0])

#Changing datatype of column from object to float
school['Percent of Students Chronically Absent'] = school['Percent of Students Chronically Absent'].astype(float)

#Converting it into percentage format
school['Percent of Students Chronically Absent'] = school['Percent of Students Chronically Absent']/100

#Creating new column with conditions and changing 'Percent of Students Chronically Absent' to categorical column
school['students_absent']= np.where(school['Percent of Students Chronically Absent']<=0.25,'1st Quratile',np.where(school['Percent of Students Chronically Absent']<=0.50,'2nd Quratile',np.where(school['Percent of Students Chronically Absent']<=0.75,'3rd Quratile','4th quartile')))

# #school
school['students_absent'].head()
new_df = school.filter(['students_absent','Supportive Environment Rating'],axis=1)
new_df.head()
#new_df['Percent of Students Chronically Absent'] = new_df['Percent of Students Chronically Absent'].str.rstrip('%').astype('float') / 100.0
new_df=new_df.dropna()
new_df.head(6)
my_tab2= pd.crosstab(school['students_absent'],school['Supportive Environment Rating'])
my_tab2
import scipy.stats as scs
from scipy import stats
import operator
from scipy.stats import chi2_contingency

#Applying contingency test to check the association
stats.chi2_contingency(my_tab2)
school = school.replace('N/A', np.nan)
school['School Income Estimate']
#Cleaning ['School Income Estimate'] columns data



school['School Income Estimate']=school['School Income Estimate'].str.replace('$','')
school['School Income Estimate']=school['School Income Estimate'].str.replace(',','')
# school['School Income Estimate']=school['School Income Estimate'].str.split('$').str[1]
#print(school['School Income Estimate'])
# school['School Income Estimate'].str.split(',').str.join(''))
school['School Income Estimate']

school['School Income Estimate']=school['School Income Estimate'].astype(float)
school['School Income Estimate'].dtype

#Replacing NAN values with mean
school['School Income Estimate'].fillna(school['School Income Estimate'].mean(), inplace=True)
school['School Income Estimate']

district_income_estimate = school.groupby('District')['School Income Estimate'].mean()
#district_income_estimate = pd.DataFrame(district_income_estimate)

#list(district_income_estimate)
sns.set(style="darkgrid")
#district_income_estimate = district_income_estimate["School Income Estimate"].value_counts().sort_values(ascending=False).index
plt.figure(figsize=(15,5))
# a = sns.barplot(x="District",y="School Income Estimate",data=school)

district_income_estimate.plot(kind="bar")

plt.figure(figsize=(20, 10))

import seaborn as sns


# school["Community School?"].dtype
school["Percent of Students Chronically Absent"]
a = sns.boxplot(x="Community School?", y="Percent of Students Chronically Absent",data=school)
a.set_xticklabels(a.get_xticklabels(),fontsize=25,rotation=20, ha="right")

title = "Percentage of Students Chronically Absent in comparison with community and non-community schools"
