# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
df=pd.read_csv("../input/married-at-first-sight/mafs.csv")
df.head()
df.isnull().sum()
df.isna().sum()
print(df['Location'].value_counts().keys())
print(df['Location'].value_counts())

tempLocDict=dict()
tempLocDict={'NYC&NJ':'New York City and Northern New Jersey',
             'DC':'Washington D.C.',
             'NC':'Charlotte, North Carolina',
             'PA':'Philadelphia, Pennsylvania',
             'MA':'Boston, Massachusetts',
             'IL':'Chicago, Illinois',
             'SF':'South Florida',
             'GA':'Atlanta, Georgia',
             'TX':'Dallas, Texas'}

tempLocDict1=dict()
for key,value in tempLocDict.items():
    temp=key
    tempLocDict1[value]=key

    
print(tempLocDict1)

locationlst=[]
for i in range(len(df['Location'])):
    temp1=tempLocDict1[df['Location'][i]]
    df['Location'][i]=temp1
df['Location'].value_counts().keys()
#Average age of married and divorced couples

print((df['Age'].groupby(df['Status'])).mean())


#Average 'age' gender wise: 
print(df['Age'].groupby(df['Gender']).mean())

print(df['Age'].min())
print(df['Age'].max())
#Average age of patients:
ageGrouping=['Below 30','between 30-35','Above 35']
ageCat=[]

for i in range(len(df['Age'])):
    if df['Age'][i]<=30:
        df['Age'][i]=ageGrouping[0]
    elif df['Age'][i]>30 and df['Age'][i]<=35:
        df['Age'][i]=ageGrouping[1]
    else:
        df['Age'][i]=ageGrouping[2]
        
        
print(df['Age'].unique())   
ageDict={'Below 30':0 ,'between 30-35':0 ,'Above 35':0}

for i in range(len(df['Status'])):
    if df['Status'][i]=='Divorced':
        ageDict[df['Age'][i]]+=1
        
        
print(ageDict)

keyList=list(ageDict.keys())
valList=list(ageDict.values())

import matplotlib.pyplot as plt

plt.figure()
plt.bar(keyList,valList)
plt.title('Age distribution over divorced cases')
plt.ylabel('Num of Divorced Cases')
plt.xlabel('Age segments')
plt.show()
        
#Extract the names for the experts:

startIdx=9

#Names for the doctors:
namesDocs=[]

for i in range(startIdx,len(df.columns)):
    namesDocs.append(df.columns[i])

print(namesDocs)
#alias names:

aliasNames=[]
tempLst=[]

import re
for i in namesDocs:
    if i[0]=='D':
        tempLst=re.split('([A-Z])',i[1:])
        aliasNames.append('Dr.'+''.join(tempLst[1:3])+' '+''.join(tempLst[3:]))
    else:
        tempLst=re.split('([A-Z])',i)
        aliasNames.append(''.join(tempLst[1:3])+' '+''.join(tempLst[3:5])+' '+''.join(tempLst[5:]))
        

        
print(aliasNames)
#Number of Separated Cases that every expert
#had to deal:
sepCases=dict()

#Number of married cases that every expert had to
#deal:
marrCases=dict()

#Number of cases who decided 
#to stay together after meeting 
#the expert:
stayCases=dict()

for i in range(len(namesDocs)):
    
    #By Status
    statusList=list(df[namesDocs[i]].groupby(df['Status']).sum().keys())
    sumStatus=list(df[namesDocs[i]].groupby(df['Status']).sum())
    tempStatus=[statusList[0],statusList[1]]
    sepCases.update({aliasNames[i]:sumStatus[0]})
    marrCases.update({aliasNames[i]:sumStatus[1]})
    
    #By Gender
    gendList=list(df[namesDocs[i]].groupby(df['Gender']).sum().keys())
    sumGender=list(df[namesDocs[i]].groupby(df['Gender']).sum())
    tempGender=[gendList[0],gendList[1]]
    
    #By Age distn:
    ageLst=list(df[namesDocs[i]].groupby(df['Age']).sum().keys())
    sumAge=list(df[namesDocs[i]].groupby(df['Age']).sum())
    tempAge=[ageLst[0],ageLst[1],ageLst[2]]
    plt.figure()
    plt.bar(tempAge,sumAge)
    plt.ylabel('Num of cases for each category')
    plt.xlabel('Age Categories')
    plt.title('For'+' '+aliasNames[i])
    plt.show()
    
    #By Location
    plt.figure()
    tempLst=[]
    tempLocation=list(df[namesDocs[i]].groupby(df['Location']).sum().keys())
    sumLocation=list(df[namesDocs[i]].groupby(df['Location']).sum())
    plt.bar(tempLocation,sumLocation)
    plt.ylabel('Num of caeses')
    plt.xlabel('Locations')
    plt.title('For'+' '+aliasNames[i])
    plt.show()
    
    #Decision to stay together: 
    tempLst=[]
    tempDecision=list(df[namesDocs[i]].groupby(df['Decision']).sum().keys())
    tempDecision=['Didn\'t stay','stay together']
    sumDecision=list(df[namesDocs[i]].groupby(df['Decision']).sum())
    stayCases.update({aliasNames[i]:sumDecision[1]})
    
    finListCats=[*tempStatus,*tempDecision,*tempGender]
    finListVals=[*sumStatus,*sumDecision,*sumGender]
    
    plt.figure()
    plt.bar(finListCats,finListVals)
    plt.ylabel('Num of caeses')
    plt.xlabel('Various catergories')
    plt.title('For'+' '+aliasNames[i])
    plt.show()
    
    

print(sepCases)
print(stayCases)
namLst=[]
caseLst=[]
tempLst=[]

for key,value in stayCases.items():
    tempLst=re.split(' ',key)
    namLst.append(tempLst[0])
    caseLst.append(value)

plt.figure()
plt.bar(namLst,caseLst)
plt.title('Experts vs. Num of Cases where couples decided to stay together')
plt.ylabel('Num of Cases')
plt.xlabel('Doctors/Experts')
plt.show()

namLst=[]
caseLst=[]
tempLst=[]

for key,value in sepCases.items():
    tempLst=re.split(' ',key)
    namLst.append(tempLst[0])
    caseLst.append(value)

plt.figure()
plt.bar(namLst,caseLst)
plt.title('Experts vs. the num of divorce cases that they had to deal')
plt.ylabel('Num of Cases')
plt.xlabel('Doctors/Experts')
plt.show()
import matplotlib.pyplot as plt

plt.bar(df['Location'].value_counts().keys(),df['Location'].value_counts())
plt.xlabel('locations')
plt.ylabel('Number of couples')
plt.show()
temp=set(df['Location'])
tempPrime=dict()
for keys in temp:
    tempPrime.update({keys:0})    
print(tempPrime)    

tempPrime1=dict()
tempPrime1=tempPrime.copy()
print(tempPrime1)
temp1=''
for i in range(len(df['Status'])):
    temp1=df['Location'][i]
    
    if df['Status'][i]=='Married':
        tempPrime[temp1]+=1
    else:
        tempPrime1[temp1]+=1

print(tempPrime)
print(tempPrime1)
#For the married number:
tempPrime
tempLst1=[]
tempLst2=[]

import matplotlib.pyplot as plt
tempLst1=list(tempPrime.keys())
tempLst2=list(tempPrime.values())
plt.bar(tempLst1,tempLst2)
plt.title('Distribution of married couples across various regions')
plt.show()
#For the divorced cases:
tempLst2=[]
tempLst1=[]

import matplotlib.pyplot as plt
tempLst1=list(tempPrime1.keys())
tempLst2=list(tempPrime1.values())
plt.bar(tempLst1,tempLst2)
plt.title('Distribution of divorced cases across various regions')
plt.show()
removWords=['of','from','the','and','not','an','then','.','best','minimum','maximam','to','learn',
            'for','with','in','by',' ',';','a',':','','on','using','basics','1','2','3','4','cc','own','you','step','become','&',
           'how','de','and','from']
occuDict_divorced=dict()
occuDict_married=dict()

for j in range(len(df['Occupation'])):
    tempOccu=''
    tempOccu=(df['Occupation'][j]).lower()
    if df['Status'][j]=='Divorced':
        if tempOccu not in occuDict_divorced.keys():
            occuDict_divorced.update({tempOccu:1})
        else:
            occuDict_divorced[tempOccu]+=1        
    else:
        
        if tempOccu not in occuDict_married.keys():
            occuDict_married.update({tempOccu:1})
        else:
            occuDict_married[tempOccu]+=1
        
print(occuDict_married)
print(occuDict_divorced)
print(len(occuDict_divorced))
print(len(occuDict_married))
finCats={'healthCare':['nurse','health','healthcare'],
'technical':['software','technician','engineer'],
'managerial':['manager','executive','businessman','director','owner','president'],
'sales':['sales',],
'consulting':['consultant','analyst','scientist'],
'realEstate':['real estate','realtor'],
'teacher':['teacher','tutor','mentor','coach']}

removWords=['of','from','the','and','not','an','then','.','best','minimum','maximam','to','learn',
            'for','with','in','by',' ',';','a',':','','on','using','basics','1','2','3','4','cc','own','you','step','become','&',
           'how','de','complete','design']
marriedDivorced=[occuDict_divorced.keys(),occuDict_married.keys()]

occuCat=dict()
occuCatMarried={'healthCare':0,'technical':0,'managerial':0,'sales':0,'consulting':0,'realEstate':0,'teacher':0}
occuCatDivorced=occuCatMarried.copy()
flag=0
for count,tempPrime in enumerate(marriedDivorced):
    for temp in tempPrime:
        tempSplits=re.split(' ',temp)
        for i in tempSplits:
            if flag!=0:
                break
            for key,val in finCats.items():
                #print((f'yet to be matched:{val,tempSplits}'))
                if i not in removWords and set([i]).intersection(set(val)):
                    if count==0:
                        #print(f'Matched{(i,val,key)}')
                        occuCatDivorced[key]+=1
                        flag=1
                        break
                    else:
                        #print(f'Matched{(i,val,key)}')
                        occuCatMarried[key]+=1
                        flag=1
                        break
                        
        flag=0        
        
            
print(occuCatMarried)
print(occuCatDivorced)
#Distribution of major Occupation categories across married and divorced couples: 

plt.bar(occuCatMarried.keys(),occuCatMarried.values())
plt.title('Distribution of major occupation categories across married couples')
plt.ylabel('Numbers')
plt.xlabel('Profession categories')
plt.show()

plt.bar(occuCatDivorced.keys(),occuCatDivorced.values())
plt.title('Distribution of major occupation categories across divorced couples')
plt.ylabel('Numbers')
plt.xlabel('Profession categories')
plt.show()
print(namesDocs)
print(aliasNames)
namesDict=dict()
for names,alias in zip(namesDocs,aliasNames):
    namesDict.update({names:re.split(' ',alias)[0]})
    
print(namesDict)    
finCats.items()
occuCat={'healthCare':0,'technical':0,'managerial':0,'sales':0,'consulting':0,'realEstate':0,'teacher':0}

occuCat_expert={'Dr.Pepper':occuCat,'Dr.Logan':occuCat,'Dr.Joseph':occuCat,'Chaplain':occuCat,'Pastor':occuCat,
                'Rachel':occuCat,'Dr.Jessica':occuCat,'Dr.Viviana':occuCat}

for names,shortNames in namesDict.items():
    for i in range(len(df['Occupation'])):
        tempExpert=df[names][i]
        tempSplit=[]
        tempSplit=re.split(' ',df['Occupation'][i].lower())
        for key,value in finCats.items():
            if flag!=0:
                break
            for j in tempSplit:
                if j not in removWords and set([j]).intersection(set(value)) and tempExpert==1:
                    occuCat[key]+=1
                    occuCat_expert[shortNames]=occuCat
                    flag=1
                    break
            flag=0 
    print(f'For {shortNames}:{occuCat_expert[shortNames]}')
    occuCat={'healthCare':0,'technical':0,'managerial':0,'sales':0,'consulting':0,'realEstate':0,'teacher':0}
    
for key,val in occuCat_expert.items():
    tempDict=occuCat_expert[key]
    
    plt.figure()
    plt.bar(tempDict.keys(),tempDict.values())
    plt.title(f'Distribution of customers across different profession for {key}')
    plt.ylabel('Number')
    plt.show()
