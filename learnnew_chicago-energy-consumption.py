#Importing the required modules
import numpy as np 
import pandas as pd 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df=pd.read_csv("../input/energy-usage-2010.csv")
df.head(10)
df.info()
#Counting the number of occurrence of each Area
df['COMMUNITY AREA NAME'].value_counts()
#Total enrgy consumption of each  Area
sdf=df.groupby(['COMMUNITY AREA NAME']).sum()
sdf.head(10)
#Area which comsumes more THERMS energy
sdf.nlargest(10, 'TOTAL THERMS')
lig=sdf.nlargest(10, 'TOTAL THERMS').index
print(lig)
dfl=df[df['COMMUNITY AREA NAME'].isin(lig)]
#Area which comsumes more energy
sdf.nlargest(10, 'TOTAL KWH')
lig=sdf.nlargest(10, 'TOTAL KWH').index
print(lig)
dfl=df[df['COMMUNITY AREA NAME'].isin(lig)]
#Area which comsumes less THERMS energy
sdf.nsmallest(10, 'TOTAL THERMS').iloc[:10]
lis=sdf.nsmallest(10, 'TOTAL THERMS').index
print(lis)
dfs=df[df['COMMUNITY AREA NAME'].isin(lis)]
#Area which comsumes less energy
sdf.nsmallest(10, 'TOTAL KWH').iloc[:10]
lis=sdf.nsmallest(10, 'TOTAL KWH').index
print(lis)
dfs=df[df['COMMUNITY AREA NAME'].isin(lis)]
fig, axs = plt.subplots(nrows=2,figsize=(20,10))
sns.countplot(x='BUILDING TYPE',data=dfs,hue='COMMUNITY AREA NAME', ax=axs[0])
axs[0].legend(loc='upper right')
axs[0].set_title("Area which consume less power")

sns.countplot(x='BUILDING TYPE',data=dfl,hue='COMMUNITY AREA NAME', ax=axs[1])
axs[1].legend(loc='upper right')
axs[1].set_title("Area which consume more power")
fig, axs = plt.subplots(nrows=2,figsize=(20,15))
sns.barplot(x=dfs['BUILDING TYPE'],y=dfs['TOTAL KWH'],hue=dfs['COMMUNITY AREA NAME'], ax=axs[0], ci=None)
sns.barplot(x='BUILDING TYPE',y='TOTAL KWH',data=dfl,hue='COMMUNITY AREA NAME', ax=axs[1], ci=None)

axs[0].legend(loc='upper right')
axs[0].set_ylim(0,700000)
axs[0].set_title("Area which consume less power")
axs[0].legend(loc=0)

axs[1].legend(loc=0)
axs[1].set_ylim(0,700000)
axs[1].set_title("Area which consume more power")
fig, axs = plt.subplots(nrows=2,figsize=(20,15))
sns.barplot(x='COMMUNITY AREA NAME',y='TOTAL POPULATION',data=dfs, ax=axs[0], ci=None)
sns.barplot(x='COMMUNITY AREA NAME',y='TOTAL POPULATION',data=dfl, ax=axs[1], ci=None)


axs[0].set_title("Area which consume less power")

axs[1].set_title("Area which consume more power")
#Getting energy consumption for each month
dft=pd.concat([dfs,dfl])
dft=dft.groupby(['COMMUNITY AREA NAME']).sum()
dft=dft.iloc[:,1:13]
dft
plt.figure(figsize=(20,6))
sns.clustermap(dft,cmap='coolwarm')
