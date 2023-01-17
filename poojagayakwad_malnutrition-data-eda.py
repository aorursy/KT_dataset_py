# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
est = pd.read_csv("/kaggle/input/malnutrition-across-the-globe/malnutrition-estimates.csv")

avg_data =pd.read_csv("/kaggle/input/malnutrition-across-the-globe/country-wise-average.csv")
df=est.copy()

est.isnull().any()
# Replacing NaN values of some columns with zero

df['Severe Wasting']= df['Severe Wasting'].fillna(0.0)

df['Wasting'] = df['Wasting'].fillna(0.0)

df['Overweight'] = df['Overweight'].fillna(0.0)

df['Stunting'] = df['Stunting'].fillna(0.0)

df['Underweight'] = df['Underweight'].fillna(0.0)

# list of undernutrition parameters

param_list =[ 'Overweight','Severe Wasting','Stunting', 'Underweight', 'Wasting']





df2 =df.loc[df.Country == 'INDIA']



ind_data =pd.DataFrame(columns =['parameter','Year','value'])



years =[1989, 1992, 1993, 1997, 1999, 2006, 2014, 2015, 2017]



# storing data year wise

for param in param_list:

    for year in years:

        value=df2.loc[df2.Year == year][param]

        value_dict = {'parameter':param,'Year':year,'value':value}

        ind_data = pd.concat([ind_data, pd.DataFrame(data=[value_dict])])

    

        

plt.figure(figsize =(10,8))

sns.barplot(data =ind_data, x='Year', y='value', hue='parameter')

plt.show()


data =pd.DataFrame(columns =['parameter','Income Classification','average'])



for param in param_list:

    for i in range(4):

        #value=df.loc[df['Income Classification']==i][param].mean()

        value = avg_data.loc[avg_data['Income Classification']==i][param].mean()

        value_dict = {'parameter':param,'Income Classification':i,'average':value}

        data = pd.concat([data, pd.DataFrame(data=[value_dict])])

        #data = pd.concat([data, pd.DataFrame.from_records([value_dict])])

        



plt.figure(figsize =(8,8))

sns.barplot(data =data, x='Income Classification', y='average', hue='parameter')

plt.show()

df = df.loc[df.Year >=2018]

temp = df.loc[(df.LDC == 1) & (df.LIFD ==1)]



# 10 contries in which Stunting percentage is highest

stunting_df = temp.sort_values(by='Stunting', ascending =False).head(10)

underweight_df =temp.sort_values(by='Underweight', ascending =False).head(10)

overweight_df=temp.sort_values(by='Overweight', ascending =False).head(10)

severe_wasting_df=temp.sort_values(by='Severe Wasting', ascending =False).head(10)

wasting_df =temp.sort_values(by='Wasting', ascending =False).head(10)



fig =plt.figure(figsize = (20,11))

ax1 = fig.add_subplot(3,2,1)

ax2 =fig.add_subplot(3,2,2)

ax3 =fig.add_subplot(3,2,3)

ax4 =fig.add_subplot(3,2,4)

ax5 =fig.add_subplot(3,2,5)



sns.barplot(data =wasting_df,ax=ax1, y='Country', x='Wasting', orient='h')

sns.barplot(data =severe_wasting_df,ax=ax2, y='Country', x='Severe Wasting', orient='h')

sns.barplot(data =stunting_df,ax=ax3, y='Country', x='Stunting', orient='h')

sns.barplot(data =underweight_df,ax=ax4, y='Country', x='Underweight', orient='h')

sns.barplot(data =overweight_df,ax=ax5, y='Country', x='Overweight', orient='h')

plt.show()
df = est.copy()

df = df.loc[df.Year >=2017]

data2 =pd.DataFrame(columns =['parameter','LLDC or SID2','average'])



for param in param_list:

    for i in [0.0,1.0,2.0]:

        #value=df.loc[df['Income Classification']==i][param].mean()

        value=df.loc[df['LLDC or SID2']==i][param].mean()

        value_dict = {'parameter':param,'LLDC or SID2':i,'average':value}

        

        data2 = pd.concat([data2, pd.DataFrame(data=[value_dict])])

        #data = pd.concat([data, pd.DataFrame.from_records([value_dict])])

        

#sns.factorplot(data =data, y='average', x='parameter', col='Income Classification', kind='bar')

plt.figure(figsize =(8,8))

sns.barplot(data =data2,  y='average', x='LLDC or SID2', hue='parameter')

plt.show()



# Recent surveys do not have data about SID countries
avg_data.sort_values(by=['Severe Wasting','Stunting','Wasting','Underweight','Overweight']).head(6)
avg_data.sort_values(by=['Severe Wasting','Stunting','Wasting','Underweight','Overweight'], ascending =False).head(10)