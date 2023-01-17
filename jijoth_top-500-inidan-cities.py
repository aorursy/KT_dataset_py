# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/top-500-indian-cities/cities_r2.csv")
columns=df.columns

columns

df.dtypes
states=df.state_name.unique()

location=df.location.unique()
state_population=df.groupby(['state_name'])['population_total','literates_total'].sum().reset_index()
state_population=state_population.sort_values(by='population_total',ascending=False).reset_index(drop=True)

state_population['Literates_Prop']=state_population['literates_total']/state_population['population_total']*100

state_population
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(16, 10))

sns.barplot(y='state_name',x='population_total',data=state_population,color='b')

sns.barplot(y='state_name',x='literates_total',data=state_population,color='r')

Literates_Prop=state_population.sort_values(by='Literates_Prop',ascending=False).reset_index(drop=True)

plt.figure(figsize=(16,10))

sns.barplot(x='Literates_Prop',y='state_name',data=Literates_Prop)
female_literacy=df.groupby(['state_name'])['literates_total','literates_male','literates_female'].sum().reset_index()
female_literacy=female_literacy.sort_values(by='literates_total',ascending=False)

female_literacy['male_literacy_prop']=female_literacy['literates_male']/female_literacy['literates_total']*100

female_literacy['female_literacy_prop']=female_literacy['literates_female']/female_literacy['literates_total']*100

female_literacy['literacy_diff_gender']=female_literacy['female_literacy_prop']-female_literacy['male_literacy_prop']



plt.figure(figsize=(16,10))

#sns.barplot(y='state_name',x='literates_total',data=female_literacy,color='b')

sns.barplot(y='state_name',x='literates_male',data=female_literacy,color='g')

sns.barplot(y='state_name',x='literates_female',data=female_literacy,color='r')

plt.figure(figsize=(16,10))

sns.barplot(y='state_name',x='female_literacy_prop',data=female_literacy.sort_values(by='female_literacy_prop',ascending=False))
plt.figure(figsize=(16,10))

sns.barplot(y='state_name',x='literacy_diff_gender',data=female_literacy.sort_values(by='literacy_diff_gender',ascending=False))
female_graduate=df.groupby(['state_name'])['total_graduates','male_graduates','female_graduates'].sum().reset_index()

female_graduate=female_graduate.sort_values(by='total_graduates',ascending=False)

female_graduate['male_graduate_prop']=female_graduate['male_graduates']/female_graduate['total_graduates']*100

female_graduate['female_graduate_prop']=female_graduate['female_graduates']/female_graduate['total_graduates']*100

female_graduate['graduate_diff_gender']=female_graduate['female_graduate_prop']-female_graduate['male_graduate_prop']
plt.figure(figsize=(16,10))

sns.barplot(y='state_name',x='graduate_diff_gender',data=female_graduate.sort_values(by='graduate_diff_gender',ascending=False))
df['join_id']=df['state_code']*100+df['dist_code']
DistCodes=pd.read_csv("/kaggle/input/state-district-codes-ind/State_District_Codes_India.csv")
DistCodes
df=df.merge(DistCodes,on='join_id',how='left')
df=df.drop(['States','join_id','Sates_Code','Dist Code'],axis=1)
df
temp=df[pd.isnull(df['Dist Name'])]
temp
df=df.sort_values(by='effective_literacy_rate_total',ascending=False)

plt.figure(figsize=(30, 100))

sns.barplot(y='Dist Name',x='effective_literacy_rate_total',data=df)
df=df.sort_values(by='effective_literacy_rate_female',ascending=False)

plt.figure(figsize=(30, 100))

sns.barplot(y='Dist Name',x='effective_literacy_rate_female',data=df)
df=df.sort_values(by='total_graduates',ascending=False)

plt.figure(figsize=(20, 100))

sns.barplot(y='Dist Name',x='total_graduates',data=df)
df=df.sort_values(by='female_graduates',ascending=False)

plt.figure(figsize=(20, 100))

sns.barplot(y='Dist Name',x='female_graduates',data=df)
df['female_graduate_ratio_gender']=df['female_graduates']/df['male_graduates']*100

df=df.sort_values(by='female_graduate_ratio_gender',ascending=False)

plt.figure(figsize=(20, 20))

sns.barplot(y='Dist Name',x='female_graduate_ratio_gender',data=df[df['female_graduate_ratio_gender']>=100])