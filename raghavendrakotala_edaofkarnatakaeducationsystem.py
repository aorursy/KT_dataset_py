# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Town-wise-education - Karnataka.csv')
#we can look at the top 5 words
df.head()
#provides the statics of numerical data like (count,mean,max etc..)
df.describe()

#gives column wise information like type of the colum and counts.. we can see expect type of area, area name, age-group, table name all are numeric fields only
df.info()
#lets plot some graphs with categarical features first, i will be using seaborn
import seaborn as sns
import numpy as np
df_cate = df.select_dtypes(include=np.object)
df_cate.info()
#find out how many unique values present in the each categorical columns
for columns in df_cate.columns:
    #df_cate[columns].value_counts()
    print('unique values for {} colums is {}'.format(columns,df_cate[columns].unique()))
for columns in ['Area Name','Age-Group']:
    print('values counts of each value in the {} columns is {}'.format(columns,df_cate[columns].value_counts()))
from matplotlib import pyplot
a4_dims = (11.7, 8.27)
#df = mylib.load_data()
fig, ax = pyplot.subplots(figsize=a4_dims)
sns.countplot(y=df_cate['Age-Group'],ax=ax)
a4_dims = (11.7, 8.27)
#df = mylib.load_data()
fig, ax = pyplot.subplots(figsize=a4_dims)
sns.countplot(y=df_cate['Area Name'],ax=ax)
#we can see both column values are distributed equally in the above figures
a4_dims = (15, 15)
#df = mylib.load_data()
fig, ax = pyplot.subplots(figsize=a4_dims)
sns.catplot(x='Age-Group',y='Literate - Males',data=df,ax=ax)
#from below we can say most number of literature people fall under group of 20 - 45
df['Total - Persons']
a4_dims = (15, 15)
#df = mylib.load_data()
fig, ax = pyplot.subplots(figsize=a4_dims)
sns.catplot(y='Area Name',x='Literate - Males',data=df,ax=ax)
#we can say bangalore is having most number of literature peoples, lets go indepth of this part
df_temp = df.groupby(['Area Name']).sum()['Literate - Persons'].sort_values(ascending=False).reset_index()
#print(df_temp.info())
a4_dims = (12, 8)
#df = mylib.load_data()
fig, ax = pyplot.subplots(figsize=a4_dims)
sns.catplot(x = 'Area Name',y='Literate - Persons',data=df_temp[:5],ax=ax)
#we can see bangalore is having most number of literates among the top 5 areas
df_temp_true = df_temp.sort_values(by=['Literate - Persons'],ascending=True)
a4_dims = (10, 8)
#df = mylib.load_data()
fig, ax = pyplot.subplots(figsize=a4_dims)
sns.catplot(x = 'Area Name',y='Literate - Persons',data=df_temp_true[:5],ax=ax)
#we can see chikmagalur is having least number of literates among least 5 areas
df_temp = df.groupby(['Area Name']).sum()['Illiterate - Persons'].sort_values(ascending=False).reset_index()
#print(df_temp.info())
a4_dims = (12, 8)
#df = mylib.load_data()
fig, ax = pyplot.subplots(figsize=a4_dims)
sns.catplot(x = 'Area Name',y='Illiterate - Persons',data=df_temp[:5],ax=ax)
#we can see bangalore is having most number of literature among the top 5 areas
df['Literate - Persons'].sum()/df['Total - Persons'].sum()
df_temp = df.groupby(['Area Name']).sum()['Illiterate - Persons'].sort_values(ascending=False)
df_temp = df.groupby(['Area Name']).sum()
df_temp['illiterate_percentage'] = df_temp.apply(lambda x:x['Illiterate - Persons']/x['Total - Persons'],axis=1)

df_temp = df_temp.sort_values(by=['illiterate_percentage'],ascending=False)
df_temp.index
df_temp['illiterate_percentage']
a4_dims = (15, 15)
#df = mylib.load_data()
fig, ax = pyplot.subplots(figsize=a4_dims)
sns.catplot(x='illiterate_percentage',y=df_temp.index,data=df_temp,ax=ax)
#graph showing % of illiterate people on areawise
# Raichur is having most number of illterate people, and udupi is having least number of illterate people
#Thats it as of now, we will do more analysis later



