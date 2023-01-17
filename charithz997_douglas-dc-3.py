# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import collections as cs

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Read the data from the file

data_df=pd.read_csv('../input/Airplane_Crashes_and_Fatalities_Since_1908.csv')
#snapshot of the data included in the file

data_df.head()
#number of null entries in each column

print(data_df.isnull().sum())
#Drop attributes with significant null entries

data_df=data_df.drop(['Time','Flight #','Route','cn/In'],axis=1)
data_df.columns
data_df.head()
data_df['Registration'].isnull().sum()
#convert date column to datetime object

data_df['Date']=pd.to_datetime(data_df['Date'])
data_df['Date'].head()
#For convenience I have selected the crashes that have occured post 1950

post_1950=data_df[data_df['Date']>'1950-01-01']
#I require only crashes related to commerical airlines, therfore I have itterated through the Operator 

#and drop any entries with relevance to military and police

string2=['Military','Navy','Force','Police']

forces=[]



for index,row in post_1950.iterrows():

    string=row['Operator']

    string=str(string)

    if any(x in string for x in string2)==True:

        post_1950.drop([index],inplace=True)

    else:

        pass
#drop null entries in aboard and fatalities columns only

post_1950=post_1950.dropna(subset=['Aboard','Fatalities'])

print(post_1950.isnull().sum())

#create a new parameter called survival from the current attributes

survival=[]

for index,row in post_1950.iterrows():

       #post_1990['Survival']=post_1990.apply(lambda x: ((post_1990['Fatalities']/post_1990['Aboard'])*100), axis=1)

        this_aboard=row['Aboard']

        this_fatality=row['Fatalities']

        survival.append(((this_aboard-this_fatality)/this_aboard)*100)

        
post_1950['Survival']=(survival)

print(post_1950['Survival'])
survived_df=post_1950[post_1950['Survival']>0].loc[:,['Date','Operator','Location','Survival','Type','Fatalities','Aboard']]

#survived_df.groupby('Type').count().sort_values('Date',ascending=False)



#Douglas DC-3 seems to have crashed the most number of times so let us analyse statistics

doug_df=survived_df[survived_df['Type']=='Douglas DC-3']

dg=doug_df.groupby('Operator').sum()#.sort_values('Operator',ascending='False')

#reset the index convert current index to a column

dg=dg.reset_index(drop=False).sort_values('Operator',ascending=True)

dg
#scatter plot

x=dg['Operator'].values

Y=dg['Fatalities'].values

ab=dg['Aboard'].values

colors = np.random.rand(77)

X=list(range(len(x)))

#plt.scatter(X,Y,s,color='b')

plt.scatter(X, Y, s=ab, c=colors, alpha=0.7)

plt.xticks(X,x,rotation=90)

plt.show()

#len(Y)
location=[]

fatal=[]

for index,row in doug_df.iterrows():

    loc=row['Location']

    fat=row['Fatalities']

    loc=str(loc)

    #split seperates the string at the given notation into two lists

    word=loc.split(',')

    #strip removes white space in the starting or ending

    location.append(word[-1].strip())

    fatal.append(fat)

    


list1=list(zip(location,fatal))

#default dict maintains duplicate values of the same key

d = cs.defaultdict(list)

for k, v in list1:

    d[k].append(v)



d
#create a dic with location and total number of fatalities for that location

sumer={}

for value,key in d.items():

    sumer[value]=sum(d[value])
sumer
#import matplotlib.pyplot as plt

#Xuniques,X = np.unique([sumer.keys()], return_inverse=True)

x=list(range(len(sumer.keys())))

y=list(sumer.values())

plt.bar(x, y, width=1.5,color='g')

plt.xticks(x,list(sumer.keys()),rotation=90)

plt.show()
loc=range(len(sumer.keys()))

#plt.hist(sumer.keys())