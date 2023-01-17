# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import squarify

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

data = pd.read_csv("../input/startup_funding.csv")

del data['Remarks']

pd.set_option('display.max_columns',10)

pd.set_option('display.max_rows', 10)



# Any results you write to the current directory are saved as output.
print(data.columns)

print(data.info())

print(data.describe())
col = ['Date','StartupName','IndustryVertical','InvestorsName','InvestmentType','SubVertical','CityLocation','AmountInUSD']

df = pd.DataFrame(data, columns = col)

df = df.dropna(how = 'any')

df['AmountInUSD'] = df['AmountInUSD'].str.replace(',','').astype(float)

print(df.dtypes)
# Total amount of investment and number of times of investment

df_SuN = df.groupby('StartupName')['AmountInUSD'].agg(['sum','count']).sort_values('sum',ascending = False)

print(df_SuN.head(15))
#Total Investment Amount in USD by Industry Vertical.

df_IndV = df.groupby('IndustryVertical')['AmountInUSD'].agg(['sum']).sort_values('sum', ascending = False)

print(df_IndV.loc[df_IndV['sum'] > 100000000])



#Print and Visualize the total numbers of funding made in each industry

industry = df['IndustryVertical'].value_counts().head(10)

sns.barplot(industry.index, industry.values , alpha = 0.6)

plt.title("Total Funding made by IndustryVertical" , fontsize = 16)

plt.xticks ( rotation = 90)

plt.xlabel("Industry vertical of Startup", fontsize = 12)

plt.ylabel("Number of Funding made",fontsize = 12)
#Cleaning Data by replacing all the names with real city names

df['CityLocation'] = df['CityLocation'].replace('Pune / US','Pune')

df['CityLocation'] = df['CityLocation'].replace('New Delhi / US','New Delhi')

df['CityLocation'] = df['CityLocation'].replace('India / US','India')

df['CityLocation'] = df['CityLocation'].replace('USA/India','India')

df['CityLocation'] = df['CityLocation'].replace('Bangalore / Bangkok','Bangalore')

df['CityLocation'] = df['CityLocation'].replace('Bangalore / San Mateo','Bangalore')

df['CityLocation'] = df['CityLocation'].replace('Dallas / Hyderabad','Hyderabad')

df['CityLocation'] = df['CityLocation'].replace('Bangalore / Palo Alto','Bangalore')

df['CityLocation'] = df['CityLocation'].replace('Pune / Dubai','Pune')

df['CityLocation'] = df['CityLocation'].replace('Gurgaon / SFO','Gurgaon')

df['CityLocation'] = df['CityLocation'].replace('Hyderabad / USA','Hyderabad')

df['CityLocation'] = df['CityLocation'].replace('Noida / Singapore','Noida')

df['CityLocation'] = df['CityLocation'].replace('Mumbai / NY','Mumbai')

df['CityLocation'] = df['CityLocation'].replace('Mumbai / UK','Mumbai')

df['CityLocation'] = df['CityLocation'].replace('Delhi','New Delhi')

df['CityLocation'] = df['CityLocation'].replace('Hyderabad/USA','Hyderabad')

df['CityLocation'] = df['CityLocation'].replace('Bangalore / SFO','Bangalore')

df['CityLocation'] = df['CityLocation'].replace('Pune/Seattle', 'Pune')

df['CityLocation'] = df['CityLocation'].replace('US/India','India')

df['CityLocation'] = df['CityLocation'].replace('New York/ India','India')

df['CityLocation'] = df['CityLocation'].replace('SFO / Bangalore','Bangalore')

df['CityLocation'] = df['CityLocation'].replace('Mumbai / Global','Mumbai')

df['CityLocation'] = df['CityLocation'].replace('Bangalore / USA','Bangalore')

df['CityLocation'] = df['CityLocation'].replace('Bangalore/ Bangkok','Bangalore')



df_city = df.groupby('CityLocation')['AmountInUSD'].agg(['sum','count'])

df_city = df_city[df_city['count'] > 9 ]

print(df_city.sort_values('sum',ascending = False))

City = df['CityLocation'].value_counts().head(15)

sns.barplot(City.index , City.values, alpha = 0.6)

plt.xticks(rotation = 70)

plt.show()
#Maximum and Minimum Funding

print("Maximum funding startup : ", df['AmountInUSD'].sort_values().max())

print("Minimum funding startup : ", df['AmountInUSD'].sort_values().min())
#Total Number and Percentage of Values that are Null.

total = data.isnull().sum().sort_values(ascending = False)

Percent = ((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending = False)

df_null = pd.concat([total, Percent],axis = 1 ,keys = ['Total', 'Percent %']  )

print(df_null.head(10))
#Print and Visualize total number of funding made in a Startup

print('Total Funding of StartupName', len(df['StartupName'].unique()))

print(df['StartupName'].value_counts().head(10))

startupName = df['StartupName'].value_counts().head(20)

plt.figure(figsize = (15,8))

sns.barplot(startupName.index, startupName.values , alpha = 0.8)

plt.title('Numbers of Funding made per Startup')

plt.xlabel('StartupName')

plt.ylabel('Total number of Funding')

plt.xticks(rotation = 'vertical')

plt.show()

#Print and Visualize Total number of funding made in a Subvertical

print('Total number of Subvertical are', len(df['SubVertical'].unique()))

SubV = df['SubVertical'].value_counts().head(10)

print(SubV)

sns.barplot(SubV.index, SubV.values, alpha = 0.8)

plt.xticks(rotation = 30)

plt.title('Number of Funding made per Subverticals', fontsize = 20)

plt.xlabel('SubVerticals',fontsize = 16)

plt.ylabel('Number of Funding', fontsize = 16)

plt.show()
count = df['CityLocation'].value_counts().head(15)

squarify.plot(sizes = count.values, label = count.index, value = count.values)

plt.title('CITY LOCATION TOTAL FUNDING BY SIZE')

plt.show()
#Number of Investment in each Type of Graphs (Just for FUN)

Investment = df['InvestmentType'].value_counts()

print(Investment)

sns.barplot(Investment.index, Investment.values, alpha = 0.6)

plt.title('TYPES OF INVESTMENT WITH TOTAL NUMBER')

plt.xlabel('Types of Investment')

plt.ylabel('Numbers of Investment')

plt.xticks(rotation = 30)

labels = Investment.index

sizes = Investment.values

explode = [0,0.1,0]

fig1 ,ax1 = plt.subplots()

ax1.pie(sizes ,explode = explode ,labels = labels, autopct = '%1.1f%%', shadow = True,startangle = 90)

ax1.axis('equal')

plt.show()

df['AmountInUSD'] = np.log(df['AmountInUSD'])

sns.boxplot(x = 'InvestmentType', y = 'AmountInUSD' , data = df)

plt.show()