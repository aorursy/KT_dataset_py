import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # for data-visualizarion

import matplotlib.pyplot as plt # this is also used for data visualization

sns.set(style = "whitegrid")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv") # using this for data processing

df_old = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv") # using this data for data-visualization
df.head() 
df.info()
# Creating the function to print the column name with the unique value it holds

def getUnique(df):

    for col in df.columns:

        print(col + " : " + str(df[col].nunique()))
getUnique(df) # as the data in df_old is same as df, so they will both print the same data
df['salary'].fillna(value = 0.0 , inplace = True) # as the unplaced students dont have a salary and so it is null. Changing the value to zero
df.head()
# Here we have multiple categorical values. These categorical value needs to be changed to dummy variables so that we can use 

# some modelling for it. Although here we are not looking to create a predictive model, given the data size.

# the practice is regarding preprocessing data, hence creating a function to help me get the dummy variables



def dummyVariable(data_column , dataframe , s) :

        just_dummies = pd.get_dummies(data_column ,prefix = s )

        dataframe = pd.concat([dataframe , just_dummies] , axis = 1)

        print(dataframe.shape)

        return dataframe
df = dummyVariable(df['status'], df , 'status')

df = dummyVariable(df['workex'], df , 'workex')

df = dummyVariable(df['specialisation'] , df , 'specialisation')

df = dummyVariable(df['degree_t'] , df , 'degree_t')

df = dummyVariable(df['hsc_s'] , df , 'hsc_s')

df = dummyVariable(df['hsc_b'] , df , 'hsc_b')

df = dummyVariable(df['ssc_b'] , df , 'ssc_b')

df = dummyVariable(df['gender'] , df , 'gender')
for col in df.columns :

    print(col)


df.drop(['gender' , 'gender_F' , 'ssc_b_Others' , 'ssc_b' , 'hsc_b_Others' , 'hsc_b' , 'hsc_s_Arts' , 'hsc_s' , 'degree_t_Others' , 'degree_t' , 'specialisation_Mkt&Fin' , 'specialisation' , 'workex_No' , 'workex' , 'status_Not Placed' , 'status'] ,axis =1 , inplace =True)
df.head()
df.shape # this shape is not similar to the old df, we have 2 extra columns. 
getUnique(df)
df.info() # here we can see that the object columns have been removed(categorical columns)
getUnique(df_old) # df_old still holds the original data with categorical columns
df_old.shape
df.describe()
f, axes = plt.subplots(2, 2, figsize=(15, 10) )

sns.despine(left=True)

ax_1 = sns.boxplot(x = "specialisation" , y = "mba_p" ,hue = "status" , data = df_old , ax = axes[1,1])

ax_2 = sns.boxplot(x = "ssc_b" , y = "ssc_p" , hue = "status"  , data = df_old  , ax = axes[0,0])

ax_3 = sns.boxplot(x = "hsc_s" , y = "hsc_p" , hue = "status"  , data = df_old , ax = axes[0,1])

ax_4 = sns.boxplot(x = "degree_t" , y = "degree_p" , hue = "status"  , data = df_old , ax = axes[1,0])

plt.tight_layout()
f, axes = plt.subplots(2, 2, figsize=(13, 7) )

sns.despine(left=True)

ax_1 = sns.countplot(x = "specialisation"  ,hue = "status" , data = df_old[df_old['workex']=="Yes"] , ax = axes[1,1])

ax_2 = sns.countplot(x = "ssc_b"  ,hue = "status" , data = df_old[df_old['workex']=="Yes"] , ax = axes[0,0])

ax_3 = sns.countplot(x = "hsc_s"  ,hue = "status" , data = df_old[df_old['workex']=="Yes"] , ax = axes[0,1])

ax_4 = sns.countplot(x = "degree_t"  ,hue = "status" , data = df_old[df_old['workex']=="Yes"] , ax = axes[1,0])

plt.tight_layout()
f, axes = plt.subplots(2, figsize=(13, 7) )

sns.despine(left=True)

ax_5 = sns.countplot(x = "gender"  ,hue = "status" , data = df_old , ax = axes[0])

ax_6 = sns.countplot(x = "workex"  ,hue = "status" , data = df_old , ax = axes[1])

plt.tight_layout()
f, axes = plt.subplots(2, 2, figsize=(15, 10) )

sns.despine(left=True)

ax_1 = sns.boxplot(x = "specialisation" , y = "mba_p" ,hue = "gender" , data = df_old , ax = axes[1,1] , palette = "Set1")

ax_2 = sns.boxplot(x = "ssc_b" , y = "ssc_p" , hue = "gender"  , data = df_old  , ax = axes[0,0] , palette = "Set1")

ax_3 = sns.boxplot(x = "hsc_s" , y = "hsc_p" , hue = "gender"  , data = df_old , ax = axes[0,1] , palette = "Set1")

ax_4 = sns.boxplot(x = "degree_t" , y = "degree_p" , hue = "gender"  , data = df_old , ax = axes[1,0] , palette = "Set1")

plt.tight_layout()
f, axes = plt.subplots(2, 2, figsize=(15, 10) )

sns.despine(left=True)

ax_1 = sns.boxplot(x = "specialisation" , y = "mba_p" ,hue = "workex" , data = df_old , ax = axes[1,1] , palette = "Set2")

ax_2 = sns.boxplot(x = "ssc_b" , y = "ssc_p" , hue = "workex"  , data = df_old  , ax = axes[0,0] , palette = "Set2")

ax_3 = sns.boxplot(x = "hsc_s" , y = "hsc_p" , hue = "workex"  , data = df_old , ax = axes[0,1] , palette = "Set2")

ax_4 = sns.boxplot(x = "degree_t" , y = "degree_p" , hue = "workex"  , data = df_old , ax = axes[1,0] , palette = "Set2")

plt.tight_layout()