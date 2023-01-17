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
# Reading Cardio Dataset

cardio=pd.read_csv('/kaggle/input/cardiogoodfitness/CardioGoodFitness.csv')
# Displaying first five rows of the dataset

cardio.head(2)
#Summary of the data

cardio.describe().round()

# Checking the shape of the data

cardio.shape
# Information about the data

cardio.info()
# Checking null  values

  

cardio.isnull().any()

# Prroduct usage by Gender and Marital Status by Creating Pivot Table



group=pd.pivot_table(cardio,values='Usage',index=['Product','Gender'],columns=['MaritalStatus'])

group

group=cardio[['Product','Age','Income','Education','Usage','Miles','Fitness']].groupby('Product').mean().round()

group
agegroup=cardio[['Gender','Usage']].groupby('Gender').mean().round()

agegroup
maritalgroup=cardio[['MaritalStatus','Usage','Miles','Fitness']].groupby('MaritalStatus').mean().round(2)

maritalgroup
# checking number male and female using the product category 



genuser=pd.pivot_table(cardio,values='Income',index=['Product'],columns=['Gender'],aggfunc=len)

genuser
genuser=pd.pivot_table(cardio,values='Age',index=['Product'],aggfunc='mean')

genuser
gp=cardio[['Product','Education']].groupby('Education').count()

gp
go=pd.pivot_table(cardio,values='Age',index=['Product'],columns=['Education'],aggfunc=len)

go
ms=pd.pivot_table(cardio,values='Age',index=['Product'],columns=['MaritalStatus'],aggfunc=len)

ms
ic=pd.pivot_table(cardio,values='Income',index=['Product'],aggfunc='mean').round()

ic
pd.crosstab(cardio.Product,cardio.Education,margins=True,margins_name='Total')
pd.crosstab([cardio.Product,cardio.Gender],[cardio.Usage,cardio.MaritalStatus],margins=True,margins_name='Total')
cardio['Age_Group']=pd.cut(cardio.Age,bins=[15,20,25,30,35,40,45,50],labels=['15-20','20-25','25-30','30-35','35-40','40-45','45-50'])

cardio.head()
pd.crosstab(cardio.Age_Group,cardio.Product,margins=True,margins_name='Total')
cardio['Income_level']=pd.cut(cardio.Income,bins=[25000,35000,45000,55000,65000,75000,85000,95000,105000],labels=['25000-35000','35000-45000','45000-55000','55000-65000','65000-75000','75000-85000','85000-95000','95000-105000'])

cardio.head()
pd.crosstab(cardio.Income_level,cardio.Product,margins=True,margins_name='Total')
pd.crosstab(cardio.Usage,cardio.Fitness,margins=True,margins_name='Total')



fu=pd.pivot_table(cardio,values='Fitness',index='Usage',aggfunc='mean').round(2)

fu
ml=cardio[cardio.Gender=='Male']

fl=cardio[cardio.Gender=='Female']



sl=cardio[cardio.MaritalStatus=='Single']

pt=cardio[cardio.MaritalStatus=='Partnered']





print('The average Usage of men is:',ml['Usage'].mean())

print('The average Usage of women is:',fl['Usage'].mean())

print('The average Fitness of men is:',ml['Fitness'].mean())

print('The average Fitness of women is:',fl['Fitness'].mean())

print('The average Usage of single people is:',sl['Usage'].mean())

print('The average Usage of partnered people is:',pt['Usage'].mean())
import matplotlib.pyplot as plt

cardio.hist(figsize=(30,30))