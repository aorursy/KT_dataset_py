import os

import pandas as pd

import numpy as np

from pandas import DataFrame,Series

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv("../input/all.csv")

df.head()
state = list(df.State)

from collections import Counter

c = Counter(state)

st_name = list(c.keys())

st_name
x = df.columns.values

x
newdf = df.groupby('State').sum()

newdf.head(36)
popdf = newdf[['Persons','Males','Females']]

popdf.head()
newdf = df.groupby('State').sum()

newdf.reset_index()
plt.figure()

x = [i for i in range(35)]

#colors = np.random.rand(50)

plt.scatter(x = x,y=newdf.Persons,cmap='flag',s=newdf.Persons/200000,alpha = 0.5)

plt.xticks(x,st_name)

plt.xticks(rotation = 90)

plt.show()
ax = plt.figure(figsize=(100,200))

#ax =popdf.plot(kind = 'bar',color = ['Red','Yellow'],width = 1)

#plt.show()

ax =popdf.plot(kind = 'bar',cmap = 'Paired',width = 0.8,figsize=(30,10))



#ax.set_facecolor('black')

#
plt.figure()

popdf = newdf[['Males','Females']]

sum_df = popdf.sum()

labels =  'Males','Females'

sizes = [sum_df['Males'],sum_df['Females']]

colors = ['yellowgreen','lightcoral']

plt.pie(sizes,labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')

plt.show()
totlpop = df['Persons'].sum()
sharedf = df[['Persons','State']]

sharedf = sharedf.groupby('State').sum()

sharedf['Share'] = sharedf['Persons']*1000/totlpop

plotdf = sharedf['Share']
plt.figure()

xi = plotdf.plot(kind='bar',width=0.8,figsize=(30,10))
sexdf = newdf[['Sex.ratio..females.per.1000.males.','Sex.ratio..0.6.years.']]

sexdf.head()

ax = sexdf.plot(kind = 'bar',color = ['Red','black'],width = 0.8,figsize=(30,10))
newdf['totedu']= newdf['Total.Educated']/1000

sx_edudf = newdf[['Sex.ratio..females.per.1000.males.','totedu']]

axi = sx_edudf.plot(kind='bar',width=1,cmap='Set1',figsize=(30,10))
sx_edudf.plot(kind='kde')
ax = sexdf.plot(kind='bar',width=1,cmap='flag',figsize=(30,10))

#ax = sexdf.plot(kind='bar',width=1,color=['magenta','cyan'],stacked = True)

#ax.set_facecolor('cyan')
sum_df=newdf.sum()



labels = 'Graduate.and.Above','Below.Primary', 'Primary', 'Middle','Matric.Higher.Secondary.Diploma' 

sizes = [sum_df['Graduate.and.Above'],sum_df['Below.Primary'],sum_df['Primary'],sum_df['Middle'],sum_df['Matric.Higher.Secondary.Diploma']]

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','red']

explode = (0.1, 0, 0, 0,0)

plt.figure()

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=140)

 

plt.axis('equal')

plt.show()
evdf = df[['State','Persons','Safe.Drinking.water','Electricity..Power.Supply.','Electricity..domestic.','Electricity..Agriculture.','Primary.school','Middle.schools','Secondary.Sr.Secondary.schools','College','Medical.facility','Primary.Health.Centre','Primary.Health.Sub.Centre','Post..telegraph.and.telephone.facility','Bus.services','Paved.approach.road','Permanent.House']].groupby('State').sum()

evdf.reset_index()

#print(evdf.iloc[:,2])

for i in range(1,6):

    evdf.iloc[:,i]=evdf.iloc[:,i]/evdf['Persons']
evdf = evdf.drop('Persons',1)
evdf.head(50)
axi = evdf.plot(kind='bar',width=1,color = ['green','red','yellow','blue','black'],figsize=(30,10))
ndf = df.set_index(['State','District'])

state_name = 'AN'

arudf = ndf.loc[state_name,['Persons', 'Males', 'Females','Sex.ratio..females.per.1000.males.', 'Sex.ratio..0.6.years.','Persons..literate', 'Males..Literate', 'Females..Literate','Persons..literacy.rate', 'Males..Literatacy.Rate','Females..Literacy.Rate','Safe.Drinking.water','Electricity..Power.Supply.','Electricity..domestic.','Electricity..Agriculture.','Primary.school','Middle.schools','Secondary.Sr.Secondary.schools','College','Medical.facility','Primary.Health.Centre','Primary.Health.Sub.Centre','Post..telegraph.and.telephone.facility','Bus.services','Paved.approach.road','Permanent.House']]

arudf.head()
f1 = arudf.iloc[:,1]

x = []

for i in arudf.index:

    y = i.split(' ')

    if (len(y) == 7 ):

        y[1]+=y[2]

    x.append(y[1])

print(x)

m = [i for i in range(len(x))] 

#print(y)

plt.figure()

f1.plot(kind = 'bar',width = 0.8,cmap = 'hsv')

plt.show()

plt.xticks(m,x)
f2 = arudf.iloc[:,1:4]

f2.plot(kind = 'bar',width = 0.8,color= ['Red','yellow','black'])

plt.xticks(m,x)
f3 = arudf.iloc[:,5:8]

f3.plot(kind = 'bar',width = 0.8,cmap='hsv',figsize=(30,10))

plt.xticks(m,x)
f4 = arudf.iloc[:,[11]]

f4.plot(kind = 'bar',width = 0.8,cmap='Vega10',figsize=(30,10))

plt.xticks(m,x)
def isnumber(x):

    try:

        float(x)

        return True

    except:

        return False



arudf = arudf[arudf.applymap(isnumber)]
f5 = arudf.iloc[:,12:]

#f5.head()

f5.plot(kind = 'bar',width = 0.8,cmap='hsv',figsize=(30,10))

plt.xticks(m,x)
import os

import pandas as pd

import numpy as np

from pandas import DataFrame,Series

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv("../input/all.csv")

df.head()
state = list(df.State)

from collections import Counter

c = Counter(state)

st_name = list(c.keys())

st_name