
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/2015.csv")
data
data.columns
data.dtypes
data.head(15)
data.tail(15)
data.info()
data.describe()
data.corr()
data.plot(kind='scatter', x='Family', y='Happiness Score',alpha = 0.5,color = 'red')  #as you see there is a strong relation between family and happiness score
plt.show()
data.plot(kind='scatter', x='Economy (GDP per Capita)',y='Happiness Score',alpha = 0.5,color = 'red')     #as you see there is a strong relation between family and happiness score
plt.show()
data["Economy (GDP per Capita)"].plot(kind='hist',bins = 50,figsize = (12,12) )   # you can see in this graph how Gdp is distrubuted
plt.show()
data["Family"].plot(kind='hist',bins = 50,figsize = (12,12) )   # you can see in this graph how Gdp is distrubuted
plt.show()
high_family=data["Family"]>1.3         # if family score is higher than 1.3 then it is a high family score. it is filtered
data[high_family]
low_family=data["Family"]<0.5          # if family score is lower than 0.5 then it is a low family score. it is filtered
data[low_family]
high_gdp=data["Economy (GDP per Capita)"]>1.35      # if gdp is higher than 1.35 then it is a high gdp. it is filtered
data[high_gdp]

low_gdp=data["Economy (GDP per Capita)"]<0.25         # if gdp is lower than 0.25 then it is a low gdp. it is filtered
data[low_gdp]
data[low_gdp].Generosity.mean()   #it shows what is low gdp's generosity mean
print(type(low_gdp))             # it shows what are their types
print(type(data.Generosity))
data[high_gdp].Generosity.mean()   #it shows what is high gdp's generosity mean. it is for the comparison between high and low gdp's generosity means 
y=data[(high_gdp)&(high_family)]    #countries are filtered by their gdp and family scores. as you can see if a country has high dgp and family scores it's happiness score is generally high
y['Happiness Score'].mean()
x=data[(low_gdp)&(low_family)]   #then as you see if they are low happiness score is low generally
x['Happiness Score'].mean()


data["high gdp"]=high_gdp           # there are some features added in dataframe
data["lowgdp"]=low_gdp
data["high family"]=high_family
data["low family"]=low_family
data ["high_gdp_and_family"]=data["high gdp"] & data ["high family"]   # countries are filtered by gdp and family and there is a feature added in dataframe. its type is boolean and it returns true or false
data
data['high_gdp_and_family'].value_counts(dropna = True)  # number of false and true
melted=pd.melt(frame=data,id_vars='Country',value_vars=['Region','Freedom'])  # this is an example for melting
melted
melted2=melted.pivot(index = 'Country', columns = 'variable',values='value')  # this is an example for pivoting
melted2
melted2.Freedom.mean()
high_freedom=melted2.Freedom>0.6        # there is a filter for freedom
melted2[high_freedom]['Region'].value_counts(dropna=False)  # it shows number of countries in different regions which has high freedom score

melted2['Region'].value_counts(dropna=False)   # this shows the regions have how many countries. we can say it is free west europe because there are 21 countries in west europe and their 12 one have high freedom score 
melted3=melted2.loc['Albania':'Angola',["Freedom"]]  # it is a slicing example
date_list = ["1923-03-03","1919-05-19","1923-10-29"] #it creates a list
datetime_object = pd.to_datetime(date_list) # then it makes the type of date_list datetime
melted3["date"]=datetime_object # it creates a feature with name date and add in datetime_object
melted3=melted3.set_index("date")
melted3
data.loc[1:10,"Family":"Generosity"] # another slicing example
def mult_2(n):
    return n*2
data.Generosity.apply(mult_2)
# it applies the function to data.Generosity
data_changed=data.set_index("Happiness Rank")# it changes index name and index to happiness rank
data_changed
data_changed2=data.set_index(["Happiness Rank","Freedom"])# there is a dataframe which has multiple index
data_changed2