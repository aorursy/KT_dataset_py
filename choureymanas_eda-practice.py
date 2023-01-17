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
#Lets import important libraries

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

%matplotlib inline
#Fetching file into a dataframe

df = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")
#Lets understand data

df.head()
df.info()
df.isnull().sum()
df['salary'].fillna(0,inplace = True)
df.isnull().sum()
#Count of males and females in data

df_gender_analysis =  df[['gender','status']].groupby(['gender'], as_index = False).count()

df_gender_analysis1 = df.loc[df['status'] == 'Placed'].groupby(['gender'],as_index = False).count()

df_gender_analysis['Placed'] = df_gender_analysis1['status']

df_gender_analysis['Placed_Percent'] = df_gender_analysis['Placed']/df_gender_analysis['status']*100

df_gender_analysis.rename(columns = {'gender':'Gender' , 'status':'Total_Students'})
#Salary analysis by gender

df_gender_analysis2 = df[['gender','salary']].groupby(['gender'],as_index = False).mean()

df_gender_analysis2
#Lets see salary distribution  

plt.figure(figsize=(12,9))

df_male = df.loc[df['gender'] == 'M']

ax = sns.distplot(df_male['salary'].loc[df['salary']!=0])

ax.ticklabel_format(style = 'plain')

plt.show()
#Lets see salary distribution by gender.

fig, ax =plt.subplots(1,2 , figsize = (14,7))

df_male = df.loc[df['gender'] == 'M']

df_female = df.loc[df['gender'] == 'F']

sns.distplot(df_male['salary'].loc[df['salary']!=0] , ax = ax[0])

sns.distplot(df_female['salary'].loc[df['salary']!=0] , ax = ax[1])

ax[0].ticklabel_format(style = 'plain')

ax[1].ticklabel_format(style = 'plain')

ax[0].set_title("Male")

ax[1].set_title("Female")

plt.show()
sns.countplot(x = 'workex' , data =df , hue = 'status')
#Lets see mean salaries for workex and no workex

df[['workex','salary']].groupby(['workex']).mean()
sns.boxplot(x = 'workex' , y = 'salary' , data=df.loc[df['status'] == 'Placed'])
figure , ax = plt.subplots(1,2 , figsize = (15,9))

df1 = df.loc[df['status'] == 'Placed']

sns.distplot(df1.loc[df['workex'] == 'Yes']['salary'] ,hist = False,ax = ax[0])

ax[0].set_title("Work Experience")

sns.distplot(df1.loc[df['workex'] == 'No']['salary']  , hist = False , ax = ax[1])

ax[1].set_title("No Work Experience")

ax[0].ticklabel_format(style = 'plain')

ax[1].ticklabel_format(style = 'plain')
df_boards_analysis =  df[['ssc_b','status']].groupby(['ssc_b'], as_index = False).count()

df_boards_analysis1 = df.loc[df['status'] == 'Placed'].groupby(['ssc_b'],as_index = False).count()

df_boards_analysis['Placed'] = df_boards_analysis1['status']

df_boards_analysis['Placed_Percent'] = df_boards_analysis['Placed']/df_boards_analysis['status']*100

df_boards_analysis.rename(columns = {'ssc_b':'SSC_Board' , 'status':'Total_Students'})
sns.catplot(x = 'status' , y = 'ssc_p' , data = df )
sns.boxplot(x = 'status' , y = 'ssc_p' , data = df )
sns.boxplot(x = 'ssc_b' , y = 'salary' , data = df.loc[df['status']=='Placed'])
df_hscboards_analysis =  df[['hsc_b','status']].groupby(['hsc_b'], as_index = False).count()

df_hscboards_analysis1 = df.loc[df['status'] == 'Placed'].groupby(['hsc_b'],as_index = False).count()

df_hscboards_analysis['Placed'] = df_hscboards_analysis1['status']

df_hscboards_analysis['Placed_Percent'] = df_hscboards_analysis['Placed']/df_hscboards_analysis['status']*100

df_hscboards_analysis.rename(columns = {'hsc_b':'HSC_Board' , 'status':'Total_Students'})
sns.boxplot(x = 'hsc_b' , y = 'salary' , data = df.loc[df['status']== 'Placed'] )
sns.boxplot(x = 'status' , y = 'hsc_p' , data = df )
df_hscsubject_analysis =  df[['hsc_s','status']].groupby(['hsc_s'], as_index = False).count()

df_hscsubject_analysis1 = df.loc[df['status'] == 'Placed'].groupby(['hsc_s'],as_index = False).count()

df_hscsubject_analysis['Placed'] = df_hscsubject_analysis1['status']

df_hscsubject_analysis['Placed_Percent'] = df_hscsubject_analysis['Placed']/df_hscsubject_analysis['status']*100

df_hscsubject_analysis.rename(columns = {'hsc_s':'HSC_subject' , 'status':'Total_Students'})
sns.boxplot(x = 'hsc_s' , y = 'salary' , data = df.loc[df['status']== 'Placed'] )
sns.scatterplot(x = 'ssc_p', y = 'hsc_p' , data =df , hue = 'status')
sns.scatterplot(x = 'ssc_p', y = 'degree_p' , data =df , hue = 'status')
sns.scatterplot(x = 'degree_p', y = 'hsc_p' , data =df , hue = 'status')
sns.countplot(x='degree_t' , data =df )
df_degree_analysis =  df[['degree_t','status']].groupby(['degree_t'], as_index = False).count()

df_degree_analysis1 = df.loc[df['status'] == 'Placed'].groupby(['degree_t'],as_index = False).count()

df_degree_analysis['Placed'] = df_degree_analysis1['status']

df_degree_analysis['Placed_Percent'] = df_degree_analysis['Placed']/df_degree_analysis['status']*100

df_degree_analysis.rename(columns = {'degree_t':'Degree_Stream' , 'status':'Total_Students'})
sns.boxplot(x = 'status' , y = 'degree_p' , data = df )
sns.scatterplot(x = 'degree_p' , y= 'salary' , data = df.loc[df['salary'] != 0])
sns.catplot(x = 'degree_t' , y = 'salary' , data = df.loc[df['salary'] != 0])
sns.boxplot(x = 'degree_t' , y = 'salary' , data = df.loc[df['salary'] != 0])
sns.countplot(x = 'degree_t' , data =df , hue = 'gender')
sns.catplot(x = 'degree_t' , y = 'salary' , data =df.loc[df['status']== 'Placed'] , hue = 'gender')
sns.boxplot(x = 'degree_t' , y = 'salary', hue='gender' , data = df.loc[df['salary'] != 0])
sns.regplot(x = 'mba_p', y = 'salary' , data = df)
sns.scatterplot(x = 'mba_p' , y = 'degree_p' , data = df , hue = 'status' )
sns.catplot(x = 'status' , y = 'etest_p' , data = df )
df[['status','etest_p']].groupby('status').mean()
sns.regplot(x = 'etest_p' , y = 'salary' , data = df.loc[df['salary'] != 0])
df_specialisation_analysis =  df[['specialisation','status']].groupby(['specialisation'], as_index = False).count()

df_specialisation_analysis1 = df.loc[df['status'] == 'Placed'].groupby(['specialisation'],as_index = False).count()

df_specialisation_analysis['Placed'] = df_specialisation_analysis1['status']

df_specialisation_analysis['Placed_Percent'] = df_specialisation_analysis['Placed']/df_specialisation_analysis['status']*100

df_specialisation_analysis.rename(columns = {'specialisation':'MBA_specialisation' , 'status':'Total_Students'})
sns.barplot(x = 'specialisation' , y = 'salary' , data =df.loc[df['status']== 'Placed'] , hue = 'gender')
sns.catplot(x = 'specialisation' , y = 'salary' , data =df.loc[df['status']== 'Placed'] , hue = 'gender')
df = pd.get_dummies(df,columns = ['status'])

df = pd.get_dummies(df,columns = ['specialisation'])

df = pd.get_dummies(df,columns = ['gender'])

#df['status'] = le.fit_transform(df['status'])



#df_dummy.rename()


df.rename(columns = {'status_Not Placed':'Not Placed' , 'status_Placed':'Placed' ,

                    'specialisation_Mkt&Fin':'Marketting and Finance' , 'specialisation_Mkt&HR':'Marketting and HR',

                    'gender_F':'Female' , 'gender_M':'Male'}, inplace =True)
df.corrwith(df['Placed'])