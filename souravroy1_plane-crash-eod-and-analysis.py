# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import matplotlib.pyplot as plt

import seaborn as sns

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/planecrashinfo_20181121001952.csv')

df.head()
# FIrst thing lets see with the first column 

df['Crash Year']= df['date'].str.split(',').str[1]

q= df['Crash Year'].value_counts()
df['Crash Year'].value_counts().tail(10)
x=q.index

y=q.values

sns.set(style="whitegrid")





plt.figure(figsize=(28,8))

ax = sns.barplot(x=x, y=y)

ax.set_title('Year wise distribution of plane crashes')

plt.xticks(rotation=55)

plt.show()
df['Crash Month']=df['date'].str.split(' ').str[0]
print("Percent accident based on month") 

df['Crash Month'].value_counts(normalize=True)*100
x=df['Crash Month'].value_counts().index

y=df['Crash Month'].value_counts().values

sns.set(style="whitegrid")

plt.figure(figsize=(20,8))

ax = sns.barplot(x=x, y=y)

ax.set_title('Moth wise distribution of crashes.')

plt.xticks(rotation=55)

plt.show()
df['time'].value_counts()
df['time']=df['time'].replace(['C ','Z','d ', 'c '], '', regex=True)
# Will work with time later .. a little complex cleaning is required
# Now I will work on what kind of Plane it was 

new= df['operator'].str.split("-", n = 1, expand = True)

df['Operator_type']=new[0]
df['Operator_type'].value_counts()
#I am recategorising them to military and non military planes based on text split. 

 

df['Operator_types']= np.where(df['Operator_type']=='Military ', 'Military', 'Non-Military/Personal')
df['Operator_types'].value_counts()
# pie chart of workers

fig = plt.figure(figsize=(6,6), dpi=100)

ax = plt.subplot(111)



df['Operator_types'].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=270, fontsize=12)

plt.show()

#Activities and rout seems to be combined together so I am now cleaning the rout column 

df['route'].value_counts()
# replacing ? to unknown to 

df['route']=np.where(df['route']=='?', 'Unknown Route',df['route'])
#List of activities involved other than moving from point a to b

l=df['route'].value_counts()

l[~l.index.str.contains(r"-", case = False)]
# I am taking the top activities and creating the graph. Ploting flight path will be overwellming

df['route_edited']= np.where(df['route'].isin(['Training','Sightseeing','Test flight','Test','Unknown Route']),df['route'], 'International flight Path/ Othe activity')

#df['route'].value_counts()[0:5,]

#df['route_edited'].value_counts()



x=df['route_edited'].value_counts().index

y=df['route_edited'].value_counts().values

sns.set(style="whitegrid")

plt.figure(figsize=(20,15))

ax = sns.barplot(x=x, y=y)

ax.set_title('Condition during accident')

plt.xticks(rotation=55)

plt.show()





#df['route_edited'].plot(kind="barh", color=sns.color_palette("deep", 3))
df.head()
df['ac_type'].value_counts()

# it seems the first name is the company name so lets see which company had most death







#There is one company whose name is split in 2 parts de Havilland

k=df[df['ac_type'].str.contains(r"de Havilland", case = False)]

o=k['ac_type'].value_counts().index
df['ac_type']= np.where(df['ac_type'].isin(o),'de_Havilland ', df['ac_type'])

#df['ac_type']=np.where(df['ac_type'])

new= df['ac_type'].str.split(" ", n = 1, expand = True)

df['ac_company']=new[0]
#df['ac_company']=new[0]

#q=df['ac_company'].apply(lambda x: x.map(x.value_counts()))

#df['ac_company'].where(df['ac_company'].apply(lambda x: x.map(x.value_counts()))>=2, "other")







df['ac_company']=df['ac_company'].where(df['ac_company'].replace(df['ac_company'].value_counts())>=10, "Other")

df['ac_company']=np.where(df['ac_company']=='?', 'Unknown Mf',df['ac_company'])
sns.set(style="whitegrid")



# Initialize the matplotlib figure

f, ax = plt.subplots(figsize=(10, 25))



# Plot the total crashes

sns.set_color_codes("pastel")

sns.barplot(x=df['ac_company'].value_counts().values, y=df['ac_company'].value_counts().index, label="Total", color="b")



#ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlabel="Manufacturing Company whose plane crashed")

sns.despine(left=True, bottom=True)
df.head()
# Now we will work with casualties
new= df['aboard'].str.split(" ", n = 1, expand = True)

df['Total_crew']=new[0]

df['Total_crew']=np.where(df['Total_crew']=='?',9999,df['Total_crew'])
new= df['fatalities'].str.split(" ", n = 1, expand = True)

df['Total_fatalities']=new[0]

df['Total_fatalities']=np.where(df['Total_fatalities']=='?',9999,df['Total_fatalities'])

df['Total_fatalities'].value_counts()

# If no wo crew and facultie are same well then its percent is 0 

df['Casualty_percent']=(pd.to_numeric(df['Total_crew'])- pd.to_numeric(df['Total_fatalities']))/pd.to_numeric(df['Total_crew'])*100
df[['Casualty_percent','Total_crew','Total_fatalities']].head(20)
df['Casualty_percent']= np.where(df['Total_crew']==df['Total_fatalities'], 100,df['Casualty_percent'])

df['Casualty_percent']=round(df['Casualty_percent'],0) # Setting the decimal place to 0, helps in plotting
df[['Casualty_percent','Total_crew','Total_fatalities']].head(20)
df['Casualty_percent'].value_counts()
# pie chart of workers

fig = plt.figure(figsize=(20,20))

ax = plt.subplot(111)







df['Casualty_percent'].value_counts().plot(kind='pie', ax=ax, autopct='%1.f%%', startangle=200, 

                                           fontsize=10)



plt.title('Casualty count in percent in airaccident ')

plt.show()
df.head()
p=df.groupby(['Crash Year'])[["Casualty_percent"]].mean()

p=p.reset_index()
x=p['Crash Year']

y=p['Casualty_percent']

sns.set(style="whitegrid")



plt.figure(figsize=(30,10))

ax = sns.barplot(x=x, y=y)

ax.set_title('% casualty over the years')

plt.xticks(rotation=90)

plt.show()