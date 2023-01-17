import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #visulization
import os
import seaborn as sns
from sklearn.preprocessing import Imputer

df_athlete=pd.read_csv("../input/athlete_events.csv")
df_Noc=pd.read_csv("../input/noc_regions.csv")
df_Noc.head()
df_athlete.head()
#summarize all variables
df_athlete.info()
# how many each team share in competeion  and the null in columns
df_athlete.set_index(['Team','ID']).count(level='Team')
df_athlete.groupby('Team')['Medal']
df_athlete['Team'].value_counts()
df_athlete['Medal'].value_counts()
df_athlete.isnull().sum().sum()
df_athlete.isnull().any().any()
df_athlete.loc[:,df_athlete.isna().any()]
null_columns=df_athlete.columns[df_athlete.isnull().any()]
df_athlete[null_columns].isnull().sum()
#handle null in data set 
#get relation between the hieght and width it is show liner relationship
plt.plot(df_athlete['Height'],df_athlete['Weight'])
plt.xlabel='Height'
plt.ylabel='Weight'
plt.show()
#get the age distribution type
import matplotlib.mlab as mlab
np.min(df_athlete['Age'])
np.max(df_athlete['Age'])
mu=np.mean(df_athlete['Age'])
sigma=np.std(df_athlete['Age'])

bins=np.linspace(10,100,10)
plt.hist(df_athlete['Age'],bins,normed=1,alpha=0.5)
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--')
plt.title("Age Distribution")
plt.show()

#file the missing value in Age with impute 
values = df_athlete[['Age']].values
imputer = Imputer()
transformed_values = imputer.fit_transform(values)
# count the number of NaN values in each column
print(np.isnan(transformed_values).sum())
df_athlete['AgeImpute']=transformed_values
df_athlete.head()

#get the age distribution type again you will find it is almost same as previous before imputing
import matplotlib.mlab as mlab
np.min(df_athlete['Age'])
np.max(df_athlete['Age'])
mu=np.mean(df_athlete['Age'])
sigma=np.std(df_athlete['Age'])

bins=np.linspace(10,100,10)
plt.hist(df_athlete['Age'],bins,normed=1,alpha=0.5)
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--')
plt.title("Age Distribution")
plt.show()
df_athlete['Sport'].value_counts()
#imputing the height and weight but by grouping in sport because each sport have it is attribute of
#hieght and weight

df_athlete.groupby('Sport')['Height'].mean()
df_athlete.groupby('Sport')['Weight'].mean()
bySport=df_athlete.groupby('Sport')
# Write a function that imputes mean
def impute_mean(series):
    return series.fillna(series.mean())
df_athlete['Height2'] = bySport.Height.transform(impute_mean)
df_athlete['Weight2'] = bySport.Weight.transform(impute_mean)
df_athlete.head()

df_athlete['Height'].min()
df_athlete['Height'].max()

df_athlete['Weight'].min()
df_athlete['Weight'].max()
df_athlete['Year'].describe()
df_athlete['City'].unique()

bins=np.linspace(1896,2016,120)
plt.hist(df_athlete['Year'],bins)
plt.show()
#number of athletes participate each years
df_athlete['Year'].value_counts
df_athlete['Year'].describe()
YearFreq=pd.crosstab(index=df_athlete['Year'],columns="Count")
N=35
colors = np.random.rand(N)
area = (30 * np.random.rand(N))**2
plt.scatter(df_athlete['Year'].unique(),YearFreq['Count'],c=colors,s=area)
plt.show()

#Each city and number of particibate
CityFreq=pd.crosstab(index=df_athlete['City'],columns="Count")
CityFreq.sort_values

# it is show only 66 type of sport .table show Athletics is the most sport played
SportFreq=pd.crosstab(index=df_athlete['Sport'],columns=df_athlete['Year'], margins=True)
SportFreq
#number of athletes by year and sex
df_athlete.groupby(['Year','Sex']).count()['Sport'].unstack().plot()
#number of athletes by year and session
df_athlete.groupby(['Year','Season']).count()['Sport'].unstack().plot()

#only athletics during all period and we will see that is played only in summer
df_athlete['Year'].unique()
df_2=df_athlete[df_athlete['Sport']=="Athletics"]
df_2.groupby('Year').count()['Sport'].plot()
df_2.groupby(['Year','Season']).count()['Sport'].unstack().plot()
df_2.groupby(['Year','Sex']).count()['Sport'].unstack().plot()
# Get Teams Plot By Medal
df_athlete['Team'].unique()
TeamFreq=pd.crosstab(index=df_athlete['Medal'],columns=df_athlete['Team'],dropna=True)
TeamFreq
p1 = sns.heatmap(TeamFreq,linewidths=0,cmap='BuPu',annot=True)

labels = df_athlete['Team'].unique()
sizes = df_athlete.groupby('Team')['Medal'].count()

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

# plot Atheletic Sport Only to see how much medals owns each team
plt.plot(df_2['Team'].unique(),df_2.groupby('Team')['Medal'].count())
plt.show()

df_2.groupby(['Team','Medal']).count()['Sport'].unstack().plot()

