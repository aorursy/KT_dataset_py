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
data = pd.read_csv('../input/2015.csv',encoding='iso-8859–1')

data.info()

data.describe()

data.head()
#Line plots



data['Happiness Score'].plot(kind = 'line', color = 'g',label = 'Happiness Score', linewidth = 1, alpha=0.8,grid=True,linestyle=':')

data['Health (Life Expectancy)'].plot(kind='line', color = 'red',label = 'Health',linewidth = 1, alpha = 0.8 ,grid = True, linestyle = ':')

plt.legend(loc= 'upper right')

plt.xlabel('Happiness')

plt.ylabel('Health')

plt.title('Line plot')

plt.show()



data['Economy (GDP per Capita)'].plot(kind = 'line', color = 'blue', label = 'Economy',linewidth = 2, alpha=0.8,grid=True,linestyle=':')

data['Health (Life Expectancy)'].plot(kind='line', color = 'black' , label = 'Health',linewidth=2,alpha=0.8,grid=True,linestyle= ':')

plt.legend(loc='upper right')

plt.xlabel('Economy')

plt.ylabel('Health')

plt.title('Line')

plt.show()

data['Freedom'].plot(kind = 'line',color = 'red', label = 'Freedom',linewidth = 2,alpha=0.8,grid=True,linestyle = ':')

data['Trust (Government Corruption)'].plot(kind = 'line', color = 'blue', label = 'Trust', linewidth = 2, alpha = 0.8, grid= True,linestyle = '-.')

plt.legend(loc = 'upper right')

plt.xlabel('Freedom')

plt.ylabel('Trust')

plt.title('line')

plt.show()

### Scatter plots



data.plot(kind = 'scatter', x ='Generosity', y = 'Economy (GDP per Capita)',alpha = 0.8,color= 'red',grid= True)

plt.xlabel('Generosity')

plt.ylabel('Economy (GDP per Capita)')

plt.show()

data.plot(kind= 'scatter', x = 'Freedom', y= 'Trust (Government Corruption)', alpha = 0.7,color= 'blue', grid=True)

plt.xlabel('Freedom')

plt.ylabel('Trust (Government Corruption)')

plt.show()
data.plot(kind = 'scatter',x = 'Standard Error', y ='Trust (Government Corruption)',alpha=0.7,color = 'g',grid=True)

plt.xlabel('Standard Error')

plt.ylabel('Trust (Government Corruption)')

plt.show()
# Histogram 

data.Family.plot(kind= 'hist',bins=50,figsize=(10,10), color = 'red', alpha=0.5,grid = True)

plt.xlabel('Family')

plt.ylabel('Frequency')

plt.title('Family')

plt.show()



data['Economy (GDP per Capita)'].plot(kind='hist',bins=50,figsize=(10,10),color='blue',alpha=0.8,grid=True)

plt.xlabel('Economy (GDP per Capita)')

plt.ylabel('Frequency')

plt.title('Economy (GDP per Capita')

plt.show()
a  = data['Trust (Government Corruption)'] > 0.5  #Qatar and Rwanda

data[a]



b = data['Economy (GDP per Capita)'] > 1.5 # Luxembourg, Singapore, Qatar and Kuwait

data[b]



c = data['Freedom'] > 0.66     # Switzerlan, Norway and Cambodia

data[c]



1.5 >1.2

x = data[np.logical_and(data['Happiness Rank']<5, data['Freedom']>0.6)] # Switzerland,Iceland,Denmark,Norway

x

meanfre = 0.428615

meantrust = 0.143422

y = data[np.logical_and(data['Freedom']>meanfre, data['Trust (Government Corruption)'] > meantrust )]

y

meanhealth = sum(data['Health (Life Expectancy)'])/len(data['Health (Life Expectancy)'])

z = data[np.logical_and(data['Happiness Score']>7, data['Health (Life Expectancy)'] < meanhealth)] # There is no such country. All countries have higher than 7 happiness score have more than mean helath score.

z







meanhappi = sum(data['Happiness Score'])/len(data['Happiness Score'])

data['Happiness_Score'] = ['Good' if i > 0.5 else 'Bad' for i in data.Freedom] # ??? TypeError: '>' not supported between instances of 'str' and 'float'





meaneco = sum(data['Economy (GDP per Capita)'])/len(data['Economy (GDP per Capita)'])

data['Mean economy comparison'] = ['Rich' if i > meaneco else 'Poor' for i in data['Economy (GDP per Capita)']]



meanfree = 0.428615

data['Free or not'] = ['Free' if i>meanfree else 'Not free' for i in data['Freedom']]



data.head()

print(data['Region'].value_counts(dropna=False))

print(data['Happiness_Score'].value_counts(dropna = False)) #Bad     105 ##Good     53

print(data['Mean economy comparison'].value_counts(dropna=False))  #Rich    87 ##Poor    71

print(data['Free or not'].value_counts(dropna=False)) #Free        83 ##Not free    75

data.boxplot(column = 'Happiness Score') #no outliers

data.boxplot(column= 'Trust (Government Corruption)')
data.boxplot(column =  'Freedom')

data.boxplot(column = 'Generosity')

#Tidy Data



new_data = data.head(30)



melted = pd.melt(frame=new_data, id_vars = 'Country',value_vars = ['Family','Freedom'])

melted





melted2 = pd.melt(frame=new_data,id_vars = 'Country',value_vars = ['Happiness Score','Generosity'])

melted2





#concat data 



data1 = data['Country'].head(20)

data2 = data['Region'].head(20)

data3 = data['Happiness Score'].head(20)

data4 = data['Family'].head(20)

data5 = data['Health (Life Expectancy)'].head(20)

data6 = data['Economy (GDP per Capita)'].head(20)



concat = pd.concat([data1,data2,data3], axis = 1)

concat



concat2 = pd.concat([data1,data2,data4],axis = 1)

concat2



concat3 =pd.concat([data1,data2,data5],axis = 1)

concat3



concat4 = pd.concat([data1,data2,data6],axis = 1)

concat4



concat4ort = sum(concat4['Economy (GDP per Capita)'])/len(concat4['Economy (GDP per Capita)'])

concat4ort



concat3ort = sum(concat3['Health (Life Expectancy)'])/len(concat3['Health (Life Expectancy)'])

concat3ort = 0.8822749999999997



concatort = sum(concat['Happiness Score'])/len(concat['Happiness Score'])

concatort = 7.252950000000001



concat2ort = sum(concat2['Family'])/len(concat2['Family'])

concat2ort = 1.271807



datafamort = sum(data['Family'])/len(data['Family'])

datafamort =  0.9910459493670887



data['First 30 family rate vs all family rate'] = ['Gelismis' if i > concat2ort else 'Gelismekte' if i > datafamort and i <concat2ort else 'Gelismemis' for i in data['Family']]

data.head(40)



dataecoort = sum(data['Economy (GDP per Capita)'])/len(data['Economy (GDP per Capita)'])

dataecoort = 0.8461372151898726



data['Country Economy'] = ['Rich' if i > concat4ort else 'Middle' if i < concat4ort and i > dataecoort else 'Poor' for i in data['Economy (GDP per Capita)']]

data.tail(40)



data.head(30)