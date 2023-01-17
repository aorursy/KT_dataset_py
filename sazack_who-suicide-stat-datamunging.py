# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/who_suicide_statistics.csv')
print("The number of rows:",dataset.shape[0])
print("The number of columns:", dataset.shape[1])
dataset = dataset.sort_values(['year'], ascending=True)
dataset.head()
from sklearn.preprocessing import LabelEncoder
LabelEncoder_Dataset = LabelEncoder()
dataset.sex = LabelEncoder_Dataset.fit_transform(dataset.sex)
dataset.age = LabelEncoder_Dataset.fit_transform(dataset.age)
dataset.suicides_no.fillna(0,inplace=True)
dataset.head()
age0 = dataset[dataset['age'] ==0]['suicides_no'].values.sum()
age1 = dataset[dataset['age'] ==1]['suicides_no'].values.sum()
age2 = dataset[dataset['age'] ==2]['suicides_no'].values.sum()
age3 = dataset[dataset['age'] ==3]['suicides_no'].values.sum()
age4 = dataset[dataset['age'] ==4]['suicides_no'].values.sum()
age5 = dataset[dataset['age'] ==5]['suicides_no'].values.sum()

X = pd.DataFrame([age3,age0,age1,age2,age4,age5])
X.index=['5-14', '15-24', '25-34','35-54-34','55-74','75+']
X.plot(kind = 'bar',title="Total Suicide based on the Age group")
male = dataset[dataset['sex'] ==0]['suicides_no'].values.sum()
female = dataset[dataset['sex'] ==1]['suicides_no'].values.sum()
AgeDF = pd.DataFrame([male,female])
AgeDF.index =['Male','Female']
AgeDF.plot(kind='bar',title="total suicide based on Sex")
dict ={}
a=[]
i=0;
unique_years = dataset.year.unique()
for uyear in unique_years:
    sum =  dataset[dataset['year'] ==uyear]['suicides_no'].values.sum()
    dict[uyear] = sum
a.append(dict)
YearBasedDF = pd.DataFrame(a)
YearBasedDF.index = ['Suicides']
YearBasedDF = YearBasedDF.transpose()
YearBasedDF.head(30)
YearBasedDF.plot(kind='line', title="Total Suicide each year")

dict ={}
b=[]
i=0;
unique_countries = dataset.country.unique()
for country in unique_countries:
    sum =  dataset[dataset['country'] ==country]['suicides_no'].values.sum()
    dict[country] = sum
b.append(dict)
countryDF = pd.DataFrame.from_dict(b)
countryDF.index=['Suicides']
countryDF = countryDF.transpose()
countryDF.plot(kind='bar' ,figsize = (20,15),title="Suicide Based on Country")