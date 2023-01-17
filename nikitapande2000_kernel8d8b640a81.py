# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
file = pd.read_csv('../input/train.csv')
print(len(file))
print(file.describe)
print(file.columns.values)
#count the number of passengers that died and survived in titanic

nops = 0

for i in file['Survived']:

    if(i==1):

        nops = nops+1

print(nops)

nopd = 891-nops

print(891-nops)
#count number of males and females

nom = 0

for i in file['Sex']:

    if(i=='male'):

        nom = nom+1

print(nom)

nof = 891-nom

print(nof)

#count number of males and females

nom = 0

for i in file['Sex']:

    if(i=='male'):

        nom = nom+1

print(nom)

nof = 891-nom

print(nof)

#number of males survived

noms = 0

nomd = 0

list1 = file['Sex']

list2 = file['Survived']

for i in range(1,len(list1)):

    if(list1[i]=='male'):

        if(list2[i]==1):

            noms = noms+1

        else:

            nomd = nomd+1

print(noms)

print(nomd)  
#number of female survived

nofs = 0

nofd = 0

list1 = file['Sex']

list2 = file['Survived']

for i in range(1,len(list1)):

    if(list1[i]=='female'):

        if(list2[i]==1):

            nofs = nofs+1

        else:

            nofd = nofd + 1

print(nofs)

print(nofd)    
# find the survival percentage of male and female

permen = (noms*100)/nom

print(permen)



perfemale = (nofs*100)/nof

print(perfemale)
#passenger class

nopc1 = 0

nopc2 = 0

nopc3 = 0

for i in file['Pclass']:

    if(i==1):

        nopc1 = nopc1+1

    elif(i==2):

        nopc2 = nopc2+1

    else:

        nopc3 = nopc3+1

print(nopc1)

print(nopc2)

print(nopc3)
#how many people survived in each cls

list1 = file['Survived']

list2 = file['Pclass']

nopc1s = 0

nopc2s = 0

nopc3s = 0

for i in range(1,len(list1)):

    if(list1[i]==1):

        if(list2[i]==1):

            nopc1s = nopc1s + 1

        elif(list2[i]==2):

            nopc2s = nopc2s + 1

        else:

            nopc3s = nopc3s + 1

print(nopc1s)

print((nopc1s*100)/216)

print(nopc2s)

print((nopc2s*100)/184)

print(nopc3s)

print((nopc3s*100)/491)
#which person paid the max and min fare

import pandas as pd

file = pd.DataFrame.from_csv('../input/train.csv')



list1 = file['Fare']

list2 = file['Name']

max1 = max(file['Fare'])

min1 = min(file['Fare'])

print(max1)

print(min1)



print("\n")



for i in range(1,len(list1)):

    if(list1[i]==max1):

        print(list2[i])



print("\n")

for i in range(1,len(list1)):

    if(list1[i]==min1):

        print(list2[i])
embark = file['Embarked']

embarks = set(embark)

embark2 = set(embark)



ems = 0

emc = 0

emq = 0

emnan = 0

diff = embarks.intersection(embark2)

print(diff)

#print(len(diff))



for i in range(1,891):

    if(embark[i] == 'S'):

        ems = ems + 1

    elif(embark[i] == 'C'):

        emc= emc+ 1

    elif(embark[i] == 'Q'):

        emq= emq+ 1

    else:

        emnan = emnan + 1

print(emc)

print(emq+1)

print(ems)

print(emnan)

print("\n")
#error in above cell---verification cell

emb = file.groupby(['Embarked']).count()

print(emb)
#how many males survived in first classs

passcls = file['Pclass']

sex1 = file['Sex']

survive = file['Survived']

ms1 = 0

ms2 = 0

ms3 = 0

fs1 = 0

fs2 = 0

fs3 = 0

for i in range(1,len(survive)):

    

    if(survive[i] == 1):

        if(passcls[i]==1):

            if(sex1[i]=="male"):

                ms1 = ms1 + 1

            elif(sex1[i]=='female'):

                fs1 = fs1 +1

        elif(passcls[i]==2):

            if(sex1[i]=="male"):

                ms2 = ms2 + 1

            elif(sex1[i]=='female'):

                fs2 = fs2 +1 

        elif(passcls[i]==3):

            if(sex1[i]=="male"):

                ms3 = ms3 + 1

            elif(sex1[i]=='female'):

                fs3 = fs3 +1 

print(ms1)

print(fs1)  

print(ms2)

print(fs2)

print(ms3)

print(fs3)

print(ms1/fs1)

print(ms2/fs2)

print(ms3/fs3)
age = file['Age']

survive = file['Survived']

minage = min(age)

maxage = max(age)

print(minage)

print(maxage)



k1 = 0

k2 = 0

k3 = 0

k4 = 0

k5 = 0



ks1 = 0

ks2 = 0

ks3 = 0

ks4 = 0

ks5 = 0



for i in range(1,len(age)):

    if(age[i]<=10):

        k1 = k1+1

    elif(age[i]>10 and age[i]<30):

        k2 = k2+1

    elif(age[i]>=30 and age[i]<50):

        k3 = k3+1

    elif(age[i]>=50 and age[i]<70):

        k4 = k4+1

    elif(age[i]>=70):

        k5 = k5+1

k = 891 - k1-k2-k3-k4-k5    

for i in range(1,len(age)):

    if(survive[i] == 1):

        if(age[i]<=10):

            ks1 = ks1+1

        elif(age[i]>10 and age[i]<30):

            ks2 = ks2+1

        elif(age[i]>=30 and age[i]<50):

            ks3 = ks3+1

        elif(age[i]>=50 and age[i]<70):

            ks4 = ks4+1

        elif(age[i]>=70):

            ks5 = ks5+1



print("\n")

print(k1)

print(k2)

print(k3)

print(k4)

print(k5)

print(k)

print("\n")



ks = nops - ks1-ks2-ks3-ks4-ks5



print(ks1)

print(ks2)

print(ks3)

print(ks4)

print(ks5)

print(ks)

print("\n")



print((ks1*100)/k1)

print((ks2*100)/k2)

print((ks3*100)/k3)

print((ks4*100)/k4)

print((ks5*100)/k5)
g1 = file.groupby(['Survived']).count()

print(g1)
g2 = file.groupby(['Pclass']).count()

print(g2)
#create a bar graph which represents number of males and females died and survived

from matplotlib import pyplot as py

print(noms)

print(nomd)

print(nofs)

print(nofd)



x=[noms,nofs]

print(x)

y = [nomd,nofd]

print(y)



py.title("survival graph")

py.xlabel("people survived")

py.ylabel("people died")

py.bar(x,y,color = 'b',align="center")

py.plot(x,y,'_',color = 'g')

py.show()


grouped = file.groupby(['Pclass','Sex'])['Survived'].sum()

print(grouped)


groupedage = file.groupby(['Age'])['Survived'].sum()

print(groupedage)
grouped = file.groupby(['Sex'])['Survived'].sum()

print(grouped)
embark = file.Embarked.unique()

print(embark)
grouped = file.groupby(['Embarked','Sex'])['Survived'].sum()

print(grouped)
#find out the name of the persons those survived in titanic

grouped = file.groupby(['Name'])['Survived'].sum()

print(grouped)


#find out the names of the people who are having age = 42

age42 = file[['Age','Name']].groupby(['Age']).get_group(42)

print(age42)
#find out the number of people survived with age 26

age26 = file[['Age']].groupby(['Age']).get_group(26).count()

print(age26)
#how many males and females survived in each age

age42 = file.groupby(['Age','Sex'])['Survived'].count()

print(age42)
#count the number of people survived those are having family with them

family = file[['Survived','SibSp']].groupby(['SibSp'])['Survived'].sum()

print(family)


#Find the record of the people those are having allen in their name

allen = file[file.Name.str.contains('Allen')]

print(allen)
#count the number of females whose title was miss

miss = file[file.Name.str.contains('Miss')].groupby('Sex').count()

print(miss)
capt = file[file.Name.str.contains('Capt')].count()

print(capt)


Dev = file[file.Name.str.contains('D')].count()

print(Dev)


mister = file[file.Name.str.contains('Mr')].count()

print(mister)
cabins = file.Cabin.unique()

print(cabins)

print(len(cabins))


cabs = file[['Cabin','Survived']].groupby(['Cabin'])['Survived'].count()

print(cabs)

print(len(cabs))
#plot bar graph for died and survived as per each passenger class

from matplotlib import pyplot as py

x = [1,2,3]

y = [nopc1s,nopc2s,nopc3s]

print(y)

print(x)

py.ylabel('Pclass')

py.xlabel('Passenger Survived')

py.bar(y,x,color = 'g')

py.show()
cabs1 = file[['Cabin','Survived']].groupby(['Survived'])['Cabin'].count()

print(cabs1)
import pandas as pd

from matplotlib import pyplot as py

file = pd.DataFrame.from_csv('../input/train.csv')

gsurvive = file.groupby(['Sex'])['Survived'].sum()

y = list(gsurvive)

print(y)



#gsex = file['Sex'].groupby(['Sex'])['Survived']

#x1 = file.Sex.unique()

#x = list(x1)

x = [1,2]

print(x)

py.bar(x,y)

py.show()
#plot graph for survival of all passenger class

x = [1,2,3]

y = [nopc1s,nopc2s,nopc3s]



print(x)

print(y)

py.bar(x,y,align = 'center')

py.show()
x = file.Age.unique()

age = list(x)



groupedage = file.groupby(['Age'])['Survived'].sum()

ages = list(groupedage)

print(ages)

print(len(ages))



x = file.Age.unique()

#x= file['']

age = list(x)

print(age)

print(len(age))



py.plot(age[:-1],ages,'1')

py.show()
x = [10,30,50,70,80]

y = [ks1,ks2,ks3,ks4,ks5]

print(x)

print(y)



py.bar(x,y,align='center')

py.show()
cabs = file[['Cabin','Survived']].groupby(['Cabin'])['Survived'].count()

print(len(cabs))

cabsl = list(cabs)

print(cabsl)

cabn = file.Cabin.unique()

cabname = list(cabn)



y=[]



for i in range(1,len(cabname)):

    y.append(i)

print(y)

py.bar(y,cabsl,align = 'center')

py.show()
ems = file['Embarked']

emsl = list(ems)



survive = file['Survived']

survivel = list(survive)



cs = 0

ss = 0

qs = 0

cd = 0

sd = 0

qd = 0

for i in range(1,len(survivel)):

    if(emsl[i]=='C'):

        if(survive[i]==1):

            cs = cs+1

        else:

            cd = cd+1

    elif(emsl[i]=='S'):

        if(survive[i]==1):

            ss = ss+1

        else:

            sd = sd+1

    elif(emsl[i]=='Q'):

        if(survive[i]==1):

            qs = qs+1

        else:

            qd = qd+1



xd = [cd,sd,qd]

xs = [cs,ss,qs]

y1 = [1,2,3]

y=[4,5,6]



print(xs)

print(y1)

print(xd)

print(y)



py.bar(y1,xs,align = 'center')

py.bar(y,xd,align = 'center',color = 'g')

py.show()