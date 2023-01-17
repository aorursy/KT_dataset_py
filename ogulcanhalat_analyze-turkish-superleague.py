#First of all we're importing libraries.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('/kaggle/input/fifa19/data.csv')
#We can see column's name.

data.columns
data.head()
data.info()
data.describe()
data.corr()
f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(data.corr(),annot=True,linewidth=0.5,fmt='.2f',ax=ax)

plt.show()
# Line Plot

# color=color,label=label,linewidth=width of line,alpha=apacity,grid=grid,linestyle=style

data.Overall.plot(kind='line',color='r',label='Overall',linewidth=1,alpha=0.5,grid=True,linestyle='-.')

data.Age.plot(kind='line',color='g',label='Age',linewidth=1,alpha=0.5,grid=True,linestyle='--')

plt.legend(loc='upper right')

plt.xlabel('X Axis')

plt.ylabel('Y Axis')

plt.show()
#Scatter plot

#As we saw in the correlation table, the number of between Overall and Potential was too close to 1. 

#So the scatter better when there is correlation between two variables because of that we choose x='Overall' and y='Potential'.

data.plot(kind='scatter',x='Overall',y='Potential',color='b',figsize=(13,13))

plt.xlabel('Overall')

plt.ylabel('Potential')

plt.title('Colleration between Overall and Potential')

plt.show()
#Histogram

#Bins:Number of bar in figure

#data.plot(kind='hist',bins=50,figsize=(13,13))

data.Overall.plot(kind='hist',bins=50,figsize=(13,13))

plt.show()
dictionary={'spain':'madrid','usa':'vegas'}

print(dictionary.keys())

print(type(dictionary))
print(dictionary.values())
# Add new key and value.

dictionary['france']='paris'

dictionary
dictionary['spain']='barcelona' # Update new existing.

dictionary
print('france' in dictionary) #For check include
del dictionary['spain'] #For remove entry 

dictionary
tr_players=data[data['Nationality']=='Turkey']

tr_players.head()
lis=[1,2,3,4,5]

for i in lis:

    print(i)
#Enumerate index and value of list

for index,value in enumerate(lis):

    print(index,'',value)
#For pandas we can achieve index and value

for index,value in data[['Overall']][0:10].iterrows(): #iterrows() is method.

    print(index,'',value)
# Lambda Function: Faster way of writing function. For example:

square = lambda x:x**2

print(square(2))
#If we apply lambda function to our data for select turkish players as we did above

tr = lambda x:data[data['Nationality']==x][['Name','Club','Position','Overall','Potential','Age','Value']]

tr('Turkey')
num1=[5,10,15]

num2=[i**2 if i==10 else i-5 for i in num1]

print(num2)
#For Our Data

threshold=sum(data.Age)/len(data.Age)

data['Age_Level']=['Middle Age' if i>threshold else 'Young' for i in data.Age]

data.loc[:10,['Age_Level','Age','Overall','Name']]
#For example lets look frequency of club 

print((data['Club']).value_counts(dropna=False)) #if there are nan values that also be counted
data.describe()#ignore null entries
data.boxplot(column='Overall',by = 'Age')
df=data.head(2)

melted=pd.melt(frame=df,id_vars='Name',value_vars=['Club','Position'])

melted.head()
melted.pivot(index='Name',columns='variable',values='value')
data1=data.head()

data2=data.tail()

conc_data=pd.concat([data1,data2],axis=0,ignore_index=True) #axis=0, adds dataframe in row

conc_data
conc_data_col=pd.concat([data1,data2],axis=1,ignore_index=True)

conc_data_col
# Lets look at does pokemon data have nan value

# As you can see there are 18207 entries. However Club has 17966 non-null object so it has 241 null object.

data.info()
data['Club'].value_counts(dropna=False) #As you can see NaN Values equal 241.
#Let's drop non values

data['Club'].dropna(inplace=True)#inplace=True means we do not assign it to new variable
#Let's check with assert statement

assert data['Club'].notnull().all() #Returns nothing because we drop nan values
data['Club'].fillna('empty',inplace=True)
assert data['Club'].notnull().all()
print(data.index.name)

data.index.name='index_name'

data.head()
data['Name'][1]
data.Name[1]
data.loc[0:10,['Name']]
data[['Name','Club']]
# Difference between selecting columns: series and dataframes

print(type(data["Name"]))     # series

print(type(data[["Name"]]))   # data frames
# Slicing and indexing series

data.loc[1:10,"Name":"Club"]   # 10 and "Club" are inclusive
# Reverse slicing 

data.loc[10:1:-1,"Name":"Club"] 
# From something to end

data.loc[1:10,"Age":] 
# Creating boolean series

boolean = data.Overall > 85

data[boolean]
# Combining filters

first_filter = data.Overall > 85

second_filter = data.Age < data.Age.mean()

data[first_filter & second_filter]
def div(n):

    return n/2

data.Overall.apply(div)
# Or we can use lambda function

data.Overall.apply(lambda n : n/2 )
data.head()
print(data.index.name)

# lets change it

data.index.name = "index"

data.head()
# Overwrite index

# if we want to modify index we need to change all of them.

data.head()

# first copy of our data to data3 then change index 

data3 = data.copy()

# lets make index start from 1. It is not remarkable change but it is just example

data3.index = range(1,18208,1)

data3.head()
# lets read data frame one more time to start from beginning

data = pd.read_csv('/kaggle/input/fifa19/data.csv')

data.head()

# As you can see there is index. However we want to set one or more column to be index
# Setting index : Nationality is outer Club is inner index

data1 = data.set_index(["Nationality","Club"]) 

data1.head(100)
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df
# pivoting

df.pivot(index="treatment",columns = "gender",values="response")

df
df1 = df.set_index(["treatment","gender"])

df1

# lets unstack it
# level determines indexes

df1.unstack(level=0)
df1.unstack(level=1)
# change inner and outer level index position

df2 = df1.swaplevel(0,1)

df2
df
# df.pivot(index="treatment",columns = "gender",values="response")

pd.melt(df,id_vars="treatment",value_vars=["age","response"])
# We will use df

df
# according to treatment take means of other features

df.groupby("treatment").mean()   # mean is aggregation / reduction method

# there are other methods like sum, std,max or min
# we can only choose one of the feature

df.groupby("treatment").age.max()
# Or we can choose multiple features

df.groupby("treatment")[["age","response"]].min() 
data=pd.read_csv('/kaggle/input/fifa19/data.csv')

tr_players.describe()
tr_players.shape
#Also we can filter with functions.

def country(x):

    return data[data['Nationality']==x][['Name','Club','Position','Overall','Potential','Age','Value']]

tr=country('Turkey')

tr.head()
#As you can see tr and tr_players are same data.

tr.shape
#Now we need turkish superleague teams for filtering.Because of that I'am using set.

club=set(tr.Club)

type(club)

club

#In this set, you can see all teams.
superleague_teams=['Kasimpaşa SK','BB Erzurumspor','Beşiktaş JK','Göztepe SK','Trabzonspor','Galatasaray SK','Medipol Başakşehir FK','Alanyaspor','Yeni Malatyaspor','Çaykur Rizespor','Sivasspor','MKE Ankaragücü','Atiker Konyaspor','Bursaspor','Antalyaspor']
#We take all players in the Super League

tr_league=data[data['Club'].isin(superleague_teams)]

tr_league
tr_league.corr()
#We can see all club age's mean.

club_ages=tr_league.groupby('Club').Age.mean()

club_ages
club_overall=tr_league.groupby('Club').Overall.mean()

club_overall
#Now Let's look foreign players in Turkish Superleague

foreign_players=tr_league[tr_league['Nationality']!='Turkey'][['Name','Club','Position','Overall','Potential','Age','Value']]

foreign_players
foreign_players.Age.mean()
#Analyzing Club Data(Galatasaray SK )

def club(c):

    return data[data['Club']==c][['Name','Position','Nationality','Overall','Potential','Age','Value']]

GS=club('Galatasaray SK')

GS
GS.Age.mean()
GS[(GS['Nationality']!='Turkey')]
# Correlation heatmap

f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(tr_league.corr(),annot=True,linewidth=0.5,fmt='.2f',ax=ax)

plt.show()
#Line Plot

tr_league.Potential.plot(kind = 'line', color = 'g', label = 'Potential', linewidth = 1, alpha = 0.5,grid = True,linestyle = ':')

data.Overall.plot(color = 'r',label = 'Overall', linewidth = 1, alpha = 0.5, grid = True, linestyle = '--')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Analysing of Potential and Overall')

plt.show()
# Histogram Plot

tr_league.Age.plot(kind='hist',bins = 30, figsize=(15,15))

plt.show()
club_ages.plot(kind="bar",figsize=(18,18))
club_overall.plot(kind="bar",figsize=(18,18))