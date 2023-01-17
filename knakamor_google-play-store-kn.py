# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np #linear algebra

import pandas as pd #data processing

import matplotlib.pyplot as plt

import seaborn as sns

import scipy as sp

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#lets read our data

df=pd.read_csv('../input/googleplaystore.csv')
#head of data

df.head()
df.info()

df.describe()

df.dtypes

'''Description'''

df['Reviews'].describe()
#get the position of the M value

df[df['Reviews'].str.contains('M')]
#drop series containing M

df.drop(df.index[10472],inplace=True)
#convert column into integer values

df['Reviews']=df['Reviews'].astype(int)
#check for conversion again

df.dtypes
def univariateAnalysis(featureName):

    mDB=pd.DataFrame({'Absolute_Frequency':featureName.value_counts()})    

    #get index of data as x axis and convert to an array

    x=mDB.index.values

    #get y and convert to an array

    y=mDB.Absolute_Frequency.values

    #scaling

    plt.figure(figsize=(8,8))

    #colors = sns.color_palette("CMRmap", len(db))

    #plotting

    BarplotDB=sns.barplot(x,y,palette="CMRmap")

    #get ticks and make it 90 degree to be visible clearly

    BarplotDB.set_xticklabels(BarplotDB.get_xticklabels(), rotation=90)

df.Type.value_counts()

df.Type.isna().sum()
'''Transformation'''

df['Type'].fillna('Free',inplace=True)

df.Type.isna().sum()
'''Analysis'''

univariateAnalysis(df.Type)

'''Description'''

df.Size.value_counts().plot(kind='bar')
'''Transformation'''

#remove M  from values of size 

df['Size'] = df['Size'].map(lambda x: x.rstrip('M'))

#extract numeric values from size column and put it in new series

newSize=df[df[['Size']].apply(lambda x: x[0].isdigit(), axis=1)]

'''here some rows were removed but we will fix it latter dont worry'''

#make new dataframe

newSizeData=newSize.Size

#convert numeric values into floats 

newSizeData=newSizeData.astype(float)

'''removing fractions will not affect size at all'''
#mean

newSizeData.mean()
#median

newSizeData.median()
#mode

newSizeData.mode()
df.Size.str.contains("k").value_counts()

#replace all values containg K by .5

df.loc[df['Size'].str.contains('k'), 'Size'] = '.5M'

'''NOW WE WILL MAKE A THIng replace varies with 0 and get'''

df.loc[df['Size'].str.contains('Varies with device'), 'Size'] = '31M'

df['Size'] = df['Size'].map(lambda x: x.rstrip('M'))
df.Size.value_counts()
#count categories values                  

df['Content Rating'].value_counts().plot(kind='bar')
df['Content Rating'].isna().sum()
univariateAnalysis( df['Content Rating'])
'''Description'''

df['Current Ver'].value_counts()

'''Transformation'''

df['Current Ver']=df['Current Ver'].fillna(value=4.1)

'''Analysis'''

univariateAnalysis( df['Android Ver'])

df['Installs'].value_counts()
df['Installs'].isnull().sum()
'''Transformation'''

#remove + from installs

df['Installs'] = df['Installs'].map(lambda x: x.rstrip('+'))

#remove commas from installs

df['Installs']  = df['Installs'] .str.replace(',', '')

df['Installs'] =df['Installs'] .astype(int)

df['Installs'].value_counts()
'''Analysis'''

univariateAnalysis(df['Installs'])

'''Description'''

df['Rating'].value_counts() 

df['Rating'].isna().sum()
'''Analysis'''

univariateAnalysis(df['Rating'])


def barplotAnalysis(x,y):

    

    plt.figure(figsize=(10,8))

    plt.xticks(rotation=90)

    #plotting

    sns.barplot(x,y,palette="CMRmap")

barplotAnalysis(df['Content Rating'],df['Installs'])
barplotAnalysis(df['Category'],df['Installs'])
barplotAnalysis(df['Rating'],df['Installs'])


#convert to float

#bins

x = [0, 10,20,50,100,1000]

df['Size']=df['Size'].astype(float)



df['Size'] = pd.cut(df['Size'],x)

df.dtypes

#coding bins groups

#db.Size.astype("category").cat.codes

barplotAnalysis(df['Size'],df['Installs'])

df.dtypes

df.head(2)
sns.set(style="ticks", color_codes=True)

g = sns.FacetGrid(df, col="Type", row="Content Rating")

g = g.map(plt.hist, "Rating")
#catplot 

sns.catplot(x='Type', y='Rating', kind='violin',palette = 'Blues',  data=df)
df.Genres.value_counts().head(2)
#catplot 

sns.catplot(x='Genres', y='Rating', kind='box',palette = 'Blues',  data=df[df.Genres.isin(['Tools', 'Entertainment', \

                                                                                          'Education'])])
sns.pairplot(df, hue ='Type')
df.info()
from scipy.stats import ttest_ind

ttest_ind(df[df.Type=='Free'].Installs, df[df.Type!='Free'].Installs)
from scipy.stats import ttest_ind

ttest_ind(df[df.Type=='Free'].Rating.dropna(), df[df.Type!='Free'].Rating.dropna())
from scipy.stats import ttest_ind

ttest_ind(df[df.Type=='Free'].Reviews.dropna(), df[df.Type!='Free'].Reviews.dropna())