import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
#Exploring the data from Times higher education
data  = pd.read_csv("timesData.csv")
print("The dimension of the table is ", data.shape)
data.head()
data.describe()
data.info()
#data pre-processing
#Dropping values
missing_data = data.dropna(inplace=True)
missing_data
#checking if the data has been cleaned
total_null = data.isna().sum().sort_values(ascending=False)
percent = (data.isna().sum()/data.isna().count()).sort_values(ascending=False)
missing_data = pd.concat([total_null, percent], axis=1, keys=['Total', 'Percent'])
missing_data
#Table to Show the top ranked universites, the country they are located and the number of students studying in them
tc = topclgs[['university_name','country','num_students']]
tc
#now since we have no missing values,we can continue the EDA.
#Our first question to be answered is- In which countries are the top ranked unis
topclgs = data.head(n = 30)
plt.figure(figsize = (20,10))
plt.hist(topclgs['country'],color = '#0504aa', edgecolor = 'black',
         bins = int(200/5))
# Add labels
plt.title('Histogram of Countries with highest ranked colleges (Top 30)',fontsize = 25)
plt.xlabel('Ranking/Country',fontsize = 15)
plt.ylabel('Frequency',fontsize = 15)
plt.show()
#Count of records displayed in the graph
df2 = topclgs.groupby("country")['world_rank'].count()
df3 = pd.DataFrame(df2)
df3
#Average Teaching Score based on the mean teaching score of the top universites in the respective country.
tc = topclgs[['country','teaching']]
df2 = tc.groupby("country")["teaching"].mean()
df3 = pd.DataFrame(df2)
df3
#Clearly the highest ranked universities according to Times are in the USA & UK.
#Now we shall try to find out countries with most international students enrolled.
tc = topclgs[['country','international_students']]
df1 = pd.DataFrame(tc)
df2 = df1.groupby("country")["international_students"].count()
plt.figure(figsize = (20,10))
ax = df2.plot.bar('country','international_students',rot=0,color='#4D67BF')
plt.xlabel('Country',fontsize=10)
plt.ylabel('Frequency',fontsize=10)
plt.title('International Students choice of Country',fontsize=20)
plt.show()
#Count of records displayed in the graph
df2 = topclgs.groupby("country")['international_students'].count()
df3 = pd.DataFrame(df2)
df3
#Next Criteria we will explore is the average research score of the top universities classified by the country.
tc = topclgs[['country','research']]
df2 = tc.groupby("country")["research"].mean()
df3 = pd.DataFrame(df2)
df3