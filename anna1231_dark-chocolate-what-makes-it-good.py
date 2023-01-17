# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np                # linear algebra

import pandas as pd               # data frames

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

import scipy.stats                # statistics

from sklearn import preprocessing



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
chocobars = pd.read_csv("../input/flavors_of_cacao.csv")



#Primeros registros

chocobars.head()
# No de observaciones y tipo de datos

print(chocobars.info())



# Numero de Observaciones y Columnas

print(chocobars.shape)
chocobars.columns = ['Maker', 'Origin', 'REF', 'Review_Date', 'Cocoa_Percent', 'Maker_Location', 

              'Rating', 'Bean_Type', 'Broad_Origin']
chocobars['Cocoa_Percent']=(chocobars['Cocoa_Percent']).str.replace('%', ' ')

chocobars['Cocoa_Percent']=(chocobars['Cocoa_Percent']).astype(float)
chocobars.head()
# No de observaciones y tipo de datos

print(chocobars.info())



# Numero de Observaciones y Columnas

print(chocobars.shape)
#First the numerical variables

chocobars.iloc[:,~chocobars.columns.isin(['Maker','Origin','Maker_Location','Bean_Type','Broad_Origin'])].describe()
corr=chocobars.iloc[:,~chocobars.columns.isin(['Maker','Origin','Maker_Location','Bean_Type','Broad_Origin'])].corr()

print (corr)
#Now the text variables

chocobars.iloc[:,~chocobars.columns.isin(['REF','Review_Date','Cocoa_Percent','Rating'])].describe()
#Now how about the country of origin?

chocobars['Broad_Origin'].value_counts()
plt.figure(figsize=(20,20))

sns.countplot(y='Broad_Origin', data=chocobars,  order = chocobars['Broad_Origin'].value_counts().index)

plt.ylabel('Broad_Origin', fontsize=15)

plt.xlabel('Number of Bars', fontsize=15)

plt.title('Number of Bars by Origin',fontsize=15)

plt.show()
#How about the country of the maker?

chocobars['Maker_Location'].value_counts()
#How about the country of the maker?

chocobars['Maker_Location'].value_counts()

plt.figure(figsize=(20,20))

sns.countplot(y='Maker_Location', data=chocobars,  order = chocobars['Maker_Location'].value_counts().index)

plt.ylabel('Maker_Location', fontsize=15)

plt.xlabel('Number of Bars', fontsize=15)

plt.title('Number of Bars by Maker Location',fontsize=15)

plt.show()
group_Origin = chocobars.groupby('Broad_Origin').agg({'Rating': ['count', 'min','median', 'max', 'mean', 'std' ]})

group_Origin.columns=['Rcount', 'Rmin','Rmedian', 'Rmax', 'Rmean', 'Rstd']

#Coeficient of variation is a standarized measure of the dispersion

group_Origin['coef_var']=group_Origin.Rstd/group_Origin.Rmean

#group_Origin

group_Origin.sort_values(['Rmean','coef_var'], ascending=[False,True])
chocobars.loc[chocobars['Maker_Location'] == 'Amsterdam','Maker_Location'] = 'Netherlands'

chocobars.loc[chocobars['Maker_Location'] == 'Niacragua','Maker_Location'] = 'Nicaragua'

chocobars.loc[chocobars['Maker_Location'] == 'Eucador','Maker_Location'] = 'Ecuador'
plt.figure(figsize=(20,20))

sns.boxplot(x="Rating", y="Maker_Location", data=chocobars,  order = chocobars['Maker_Location'].value_counts().index)

sns.swarmplot(x="Rating", y="Maker_Location", data=chocobars,  order = chocobars['Maker_Location'].value_counts().index)

plt.ylabel('Rating', fontsize=15)

plt.xlabel('Maker Location', fontsize=15)

plt.title('Rating by Maker Location',fontsize=15)

plt.show()
group_ML = chocobars.groupby('Maker_Location').agg({'Rating': ['count', 'min','median', 'max', 'mean', 'std' ]})

group_ML.columns=['Rcount', 'Rmin','Rmedian', 'Rmax', 'Rmean', 'Rstd']

#Coeficient of variation is a standarized measure of the dispersion

group_ML['coef_var']=group_ML.Rstd/group_ML.Rmean

#group_ML

group_ML.sort_values(['Rmean','coef_var'], ascending=[False,True])
france=chocobars[chocobars['Maker_Location']=='France']

plt.figure(figsize=(20,20))

sns.boxplot(x="Rating", y="Maker", data=chocobars,  order = chocobars['Maker'].value_counts().index)

sns.swarmplot(x="Rating", y="Maker", data=france,  order = france['Maker'].value_counts().index)

plt.ylabel('Rating', fontsize=15)

plt.xlabel('Maker Location', fontsize=15)

plt.title('Rating by Maker Location',fontsize=15)

plt.show()

group_F = france.groupby('Maker').agg({'Rating': ['count', 'min','median', 'max', 'mean', 'std' ]})

group_F.columns=['Rcount', 'Rmin','Rmedian', 'Rmax', 'Rmean', 'Rstd']

#Coeficient of variation is a standarized measure of the dispersion

group_F['coef_var']=group_F.Rstd/group_F.Rmean

group_F.sort_values(['Rmean','coef_var'], ascending=[False,True])