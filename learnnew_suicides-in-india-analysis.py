import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data=pd.read_csv("../input/Suicides in India 2001-2012.csv")
data.info()
data.tail()
data=data[ (data['Total']>0) | (data['State']!='Total (All India)') | (data['State']!='Total (States)') | 
          (data['State']!='Total (Uts)')]
data.info()
data.groupby('State').sum()['Total'].plot("barh",figsize=(13,7),title ="State wise suicides frequency");
data.groupby('Year').sum()['Total'].plot("bar",figsize=(13,7),title ="Year wise suicides frequency");
data.groupby('Gender').sum()['Total'].plot("bar",figsize=(13,7),title ="Gender wise suicides frequency");
data.groupby('Age_group').sum()['Total'].plot("bar",figsize=(13,7),title ="Age wise suicides frequency");
for type_code in data['Type_code'].unique():
    print("{0}: {1}".format(type_code, data[data['Type_code'] == type_code].size))
ds=data[data['Type_code']=='Causes']
ds.groupby('Type').sum()['Total'].plot("barh",figsize=(13,7),title ="Causes wise suicides frequency");
ds1=data[data['Type_code']=='Education_Status']
ds1.groupby('Type').sum()['Total'].plot("barh",figsize=(13,7),title ="Education_Status wise suicides frequency");
ds2=data[data['Type_code']=='Professional_Profile']
ds2.groupby('Type').sum()['Total'].plot("barh",figsize=(13,7),title ="Professional_Profile wise suicides frequency");
ds3=data[data['Type_code']=='Social_Status']
ds3.groupby('Type').sum()['Total'].plot("barh",figsize=(13,7),title ="Social_Status wise suicides frequency");











