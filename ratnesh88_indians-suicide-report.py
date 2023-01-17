import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import warnings

warnings.filterwarnings('ignore')

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv("../input/Suicides in India 2001-2012.csv",sep=",", date_parser= 'Year')

data.info()
data.head()
data = data[data['Total'] != 0][data['State'] != 'Total (All India)'][data['State'] != 'Total (States)'][data['State'] != 'Total (Uts)'] 
data.head()
pd.crosstab(index=data.Gender, columns =data.Age_group)
data.groupby('State').sum()['Total'].plot("bar",figsize=(13,7),title ="State wise suicides frequency");
data.groupby('Year').sum()['Total'].plot("bar",figsize=(13,7),title ="Year wise suicides frequency");
data.groupby('Type_code').sum()['Total'].plot("bar",figsize=(13,7),title ="Type_Code wise suicides frequency");
data.groupby('Type').sum()['Total'].plot("barh",figsize=(14,15),title ="Type wise suicides frequency");
data.groupby('Gender').sum()['Total'].plot("bar",figsize=(13,7),title ="Gender wise suicides frequency");
data[data['Age_group'] != '0-100+'].groupby('Age_group').sum()['Total'].plot("bar",figsize=(13,7),title ="Age group wise suicides frequency");
pd.crosstab(index=data["Age_group"],  columns=[data["Type_code"],

                                      data["Gender"]],

                             margins=True)
l = []

for v in ['Male','Female']:

    d= data[data['Gender']==v].groupby('Type_code').sum()

    w = np.array(d['Total'].values)

    l.append(w)



df2 = pd.DataFrame(l, index=['Male','Female'])



df2.plot.bar(figsize=(12,8))

plt.title("Gender wise division among Type_code",fontsize=30,color='navy')

plt.legend(d.index)

plt.xlabel("Gender",fontsize=20,color='navy')

plt.ylabel("Suicide",fontsize=20,color='navy')
l = []

s = data.groupby('State').size().index

for v in s:

    d= data[data['State']==v].groupby('Type_code').size()

    w = np.array(d.values)

    print(v,d)

    l.append(w)

    

df2 = pd.DataFrame(l, index=s)

df2.plot.bar(figsize=(12,8),stacked =True)

plt.title("Type_code division among states",fontsize=30,color='navy')

plt.legend(d.index)

plt.xlabel("State",fontsize=20,color='navy')

plt.ylabel("Suicides Count",fontsize=20,color='navy')
l = []

s = data.groupby('State').size().index

for v in s:

    d= data[data['State']==v].groupby('Age_group').sum()

    w = np.array(d['Total'].values)

    l.append(w)

    

df2 = pd.DataFrame(l, index=s)

df2.plot(kind='barh',figsize=(12,8),stacked =True)

plt.title("Age_group wise suicides among states",fontsize=30,color='navy')

plt.legend(d.index)

plt.ylabel("State",fontsize=20,color='navy')

plt.xlabel("Suicides Count",fontsize=20,color='navy')
l = []

s = data.groupby('Year').size().index

for v in s:

    d= data[data['Year']==v].groupby('Type_code').size()

    w = np.array(d.values)

    l.append(w)

    

df2 = pd.DataFrame(l, index=s)

df2.plot.bar(figsize=(12,8),stacked =True)

plt.title("Type_code division among Years",fontsize=30,color='navy')

plt.legend(d.index)

plt.xlabel("State",fontsize=20,color='navy')

plt.ylabel("Suicides Count",fontsize=20,color='navy')
l = []

s = data.groupby('Type').size().index

for v in s:

    d= data[data['Type']==v].groupby('Gender').sum()

    w = np.array(d['Total'].values)

    l.append(w)

%matplotlib agg    

df2 = pd.DataFrame(l, index=s)

df2.plot(kind='barh',figsize=(15,20))

plt.title("Type division among gender",fontsize=30,color='navy')

plt.legend(d.index)

plt.ylabel("Causes",fontsize=20,color='navy')

plt.xlabel("Suicides Count",fontsize=20,color='navy')
l = []

s = data.groupby('Year').size().index

for v in s:

    d= data[data['Year']==v].groupby('Type_code').sum()

    w = np.array(d['Total'].values)

    l.append(w)

    

df2 = pd.DataFrame(l/l[0], index=s)

df2.plot.bar(figsize=(12,8),stacked =False)

plt.title("Type_code vs Year",fontsize=30,color='navy')

plt.legend(d.index)

plt.xlabel("Year",fontsize=20,color='navy')

plt.ylabel("Suicides Count",fontsize=20,color='navy')
l = []

s = data.groupby('Year').size().index

ds = data[data['Type_code'] == 'Education_Status']

for v in s:

    d= ds[ds['Year']==v].groupby('Type').sum()

    w = np.array(d['Total'].values)

    l.append(w)

    

df2 = pd.DataFrame(l/l[0], index=s)

df2.plot(figsize=(12,8),stacked =False)

plt.title("Type(Education_Status) vs Year",fontsize=30,color='navy')

plt.legend(d.index)

plt.xlabel("Year",fontsize=20,color='navy')

plt.ylabel("Suicides Count",fontsize=20,color='navy')
l = []

s = data.groupby('Year').size().index

ds = data[data['Type_code'] == 'Causes']

for v in s:

    d= ds[ds['Year']==v].groupby('Type').sum()

    w = np.array(d['Total'].values)

    l.append(w)

    

df2 = pd.DataFrame(l/l[0], index=s)

df2.plot(figsize=(12,8),stacked =False)

plt.title("Type(Causes) vs Year",fontsize=30,color='navy')

plt.legend(d.index)

plt.xlabel("Year",fontsize=20,color='navy')

plt.ylabel("Suicides Count",fontsize=20,color='navy')
l = []

s = data.groupby('Year').size().index

ds = data[data['Type_code'] == 'Social_Status']

for v in s:

    d= ds[ds['Year']==v].groupby('Type').sum()

    w = np.array(d['Total'].values)

    #print(v,d['Total'].values)

    l.append(w)



df2 = pd.DataFrame(l/l[0], index=s)

df2.plot(figsize=(12,8),stacked =False)

plt.title("Type(Social_Status) vs Year",fontsize=30,color='navy')

plt.legend(d.index)

plt.xlabel("Year",fontsize=20,color='navy')

plt.ylabel("Suicides Count",fontsize=20,color='navy')