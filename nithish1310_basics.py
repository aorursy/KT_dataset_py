import numpy as np 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
gapminder_csv_url ='http://bit.ly/2cLzoxH'
fd = pd.read_csv("../input/startup_funding.csv")  #to reaad the csv file
gapminder = pd.read_csv("../input/startup_funding.csv")

fd.head(5)  #shows the first 5 DATASET
for i in fd.columns:
    print(i)
del fd['Remarks'] #del - command
fd.head(5)
d = pd.DataFrame(fd) 
leng=len(d.index)+1  #find length
print(leng,"ROWS are present in this DATASET")
name = gapminder['InvestorsName'].unique()   #stores unique values of INVESTORS NAME
amount = gapminder['AmountInUSD'].unique()   #stores unique values of INVESTORS NAME

print(amount)
#to find unique names of listed cities

uni_city = [] #array declaration
uni_city = gapminder['CityLocation'].unique() #to find unique

print(uni_city)
#eg:
fd[fd.CityLocation == 'Chennai'].head(5)
#to convert string to float

fd["AmountInUSD"] = fd["AmountInUSD"].apply(lambda x: float(str(x).replace(",",""))) #expression conversion is done using lambda
fd["AmountInUSD"] = pd.to_numeric(fd["AmountInUSD"]) #now those amount are converted to numeric format

fd.head(5)
#total

val = fd['AmountInUSD'].sum() #which sums up all the values in the row "AmountInUSD"
print("Total funding amount",val) #print total


#to calculate max value and to print the max row

max_invest=max(fd['AmountInUSD'])  #find max
print("maximum amount invested",max_invest)
max_index=fd['AmountInUSD'].idxmax()   #to assign max amount's index value
fd.iloc[[max_index]]   #print the row



min_invest=min(fd['AmountInUSD'])  
print("minimum amount invested",min_invest)
min_index=fd['AmountInUSD'].idxmin()   #to assign max amount's index value
fd.iloc[[min_index]]   #print the row

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
py.init_notebook_mode(connected=True)

%matplotlib inline

cityname = fd['CityLocation'].value_counts().head(10)
plt.figure(figsize=(15,8))
sns.barplot(cityname.index, cityname.values)
plt.xticks(rotation='vertical')
plt.xlabel('Cities Name')
plt.ylabel('Number of STARTUPS in each cities')

plt.show()
#fd.loc[fd['CityLocation'] == "Mumbai", 'AmountInUSD']

#total = fd.loc[fd['CityLocation'] == "Bangalore", 'StartupName'].sum()
x=0
for i in cityname.index:
    print("Number of STARTUPS in",i, "are",cityname.values[x])
    x=x+1
plt.scatter(cityname.index,cityname.values)
plt.xticks(rotation='vertical')
plt.xlabel('Cities Name')
plt.ylabel('Number of STARTUPS in each cities')

plt.show()
plt.pie(cityname.values, labels = cityname.index, autopct = "%.01f")
plt.show()
investname = fd['InvestorsName'].value_counts().head(5)
plt.figure(figsize=(15,8))
sns.barplot(investname.index, investname.values)
#plt.xticks(rotation='vertical')
plt.xlabel('Investors Name')
plt.ylabel('No. of Investments made')

plt.show()

x=0
for i in investname.index:
    print("Investments made by",i, "on",investname.values[x],"startups")
    x=x+1
plt.scatter(investname.index,investname.values)
plt.xticks(rotation='vertical')
plt.xlabel('Investors Name')
plt.ylabel('No. of Investments made')

plt.show()
plt.pie(cityname.values, labels = cityname.index, autopct = "%.01f")
plt.show()