# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load





import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import scipy.stats as stats





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv') #Importing the data

data.head() #First five observations of the data
print(data.info())

print(data.shape)
data.describe(include=['O']) #Categorical Variables
data.describe() #Numerical Variables
my_data = data.copy() #copying data to keep original intact
my_data.isnull().sum() #Number of missing values for each column
my_data.dropna(inplace = True) #drop rows with missing values

my_data.isnull().sum() 
#Gender vs Status

gpb_gender = my_data[["gender", "status"]].groupby(['gender'], as_index=False).count()

print(gpb_gender)



sns.set(style="whitegrid")

ax = sns.barplot(x="gender", y="status", data=gpb_gender)
#  Board of Education- Central/ Others 10th Grade vs Status



ssc_b_gb = my_data[["ssc_b", "status"]].groupby(['ssc_b'], as_index=False).count()

print(ssc_b_gb)



sns.set(style="whitegrid")

ax = sns.barplot(x="ssc_b", y="status", data=ssc_b_gb)
#  Board of Education- Central/ Others 12th Grade vs Status



hsc_b_gb = my_data[["hsc_b", "status"]].groupby(['hsc_b'], as_index=False).count()

print(hsc_b_gb)



sns.set(style="whitegrid")

ax = sns.barplot(x="hsc_b", y="status", data=hsc_b_gb)
# Specialization in Higher Secondary Education vs Status





hsc_s_gb = my_data[["hsc_s", "status"]].groupby(['hsc_s'], as_index=False).count()

print(hsc_s_gb)



sns.set(style="whitegrid")

ax = sns.barplot(x="hsc_s", y="status", data=hsc_s_gb)
#Under Graduation(Degree type)- Field of degree education vs Status



degree_t_gb = my_data[["degree_t", "status"]].groupby(['degree_t'], as_index=False).count()

print(degree_t_gb)



sns.set(style="whitegrid")

ax = sns.barplot(x="degree_t", y="status", data=degree_t_gb)
#Work Experience vs Status





wex_gb = my_data[["workex", "status"]].groupby(['workex'], as_index=False).count()

print(wex_gb)



sns.set(style="whitegrid")

ax = sns.barplot(x="workex", y="status", data=wex_gb)
# specialisation MBA vs status



spec_gb = my_data[["specialisation", "status"]].groupby(['specialisation'], as_index=False).count()

print(spec_gb)



sns.set(style="whitegrid")

ax = sns.barplot(x="specialisation", y="status", data=spec_gb)
#Let us make a another copy of the data



my_data1 = data.copy()

my_data1.head()



#Converting Status with 1 - Placed 0- Not Placed 



from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

my_data1['status'] = enc.fit_transform(my_data1['status'])



#Code for drawing a heatmap using seaborn

plt.figure(figsize=(20,10))

heat_map= my_data1.corr()

sns.heatmap(heat_map,annot=True)

heat_map

group_dp = my_data1[my_data1.status == 1]

group_dp = group_dp[["degree_p", "status"]].groupby(['degree_p'], as_index=False).count().sort_values(by='degree_p',ascending=False)

print(group_dp)



group_dp = my_data1[my_data1.status == 1]

group_dp = group_dp[["degree_p", "status"]].groupby(['degree_p'], as_index=False).count().sort_values(by='status',ascending=False)

print(group_dp)



group_dp = my_data1[my_data1.status == 0]

group_dp = group_dp[["degree_p", "status"]].groupby(['degree_p'], as_index=False).count().sort_values(by='degree_p',ascending=False)

print(group_dp)



group_dp = my_data1[my_data1.status == 0]

group_dp = group_dp[["degree_p", "status"]].groupby(['degree_p'], as_index=False).count().sort_values(by='status',ascending=False)

print(group_dp)



sns.set(style="whitegrid")

ax = sns.scatterplot(x="degree_p", y="status", data=group_dp)
spec_gb = my_data[["specialisation", "status"]].groupby(['specialisation'], as_index=False).count()

print(spec_gb)



sns.set(style="whitegrid")

ax = sns.barplot(x="specialisation", y="status", data=spec_gb)