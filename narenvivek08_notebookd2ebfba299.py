import pandas as pd

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns
xls = pd.ExcelFile('../input/analyst/Data Analyst Assignment (1).xlsx')

data = pd.read_excel(xls, 'Assignment-2')

data.head()


month1=data.columns[1:32]

month2=data.columns[32:62]

month3=data.columns[62:93]

month4=data.columns[93:123]

month5=data.columns[123:154]

month6=data.columns[154:185]

month7=data.columns[185:214]

month8=data.columns[214:245]

month9=data.columns[245:275]

month10=data.columns[275:306]

month11=data.columns[306:336]

month12=data.columns[336:367]
months=[month1,month2,month3,month4,month5,month6,month7,month8,month9,month10,month11,month12]

month_name=['month1','month2','month3','month4','month5','month6','month7','month8','month9','month10','month11','month12']

for month,monthname in zip(months,month_name):

    data[monthname]=data[month].sum(axis=1)

data    


month_sales=data.iloc[:,367:]

month_sales.head()
overall_sales=[]

overall_sales=month_sales.sum(axis=0)

overall_sales
overall_sales=overall_sales.to_frame()

overall_sales
overall_sales.plot(kind='bar')
sorted_sales=overall_sales.sort_values(by=0,ascending=False)

sorted_sales
data[month1].plot(kind='box', figsize=(10,10))

data[month2].plot(kind='box', figsize=(10,10))





data[month3].plot(kind='box', figsize=(10,10))

data[month4].plot(kind='box', figsize=(10,10))

data[month5].plot(kind='box', figsize=(10,10))

data[month6].plot(kind='box', figsize=(10,10))

data[month7].plot(kind='box', figsize=(10,10))

data[month8].plot(kind='box', figsize=(10,10))

data[month9].plot(kind='box', figsize=(10,10))

data[month10].plot(kind='box', figsize=(10,10))

data[month11].plot(kind='box', figsize=(10,10))

data[month12].plot(kind='box', figsize=(10,10))