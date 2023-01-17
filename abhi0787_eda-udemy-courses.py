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
df=pd.read_csv('../input/udemy-courses/udemy_courses.csv')
df.head()
df.columns
df['level'].value_counts()
import matplotlib.pyplot as plt
import re

levList=list(df['level'].value_counts().keys())
print(levList)
tempSplit=[]

[tempSplit.append(re.split(' ',i)[0]) for i in levList] 

plt.bar(tempSplit,list(df['level'].value_counts()))
plt.xlabel('Level of courses')
plt.ylabel('#Num of Courses')
plt.title('Distribution of courses across different difficulty Levels')
plt.show()

valList=list(df['is_paid'].value_counts())
keyList=list(df['is_paid'].value_counts().keys())

print(valList)
print(keyList)

keyList=['Paid courses','Unpaid courses']

plt.bar(keyList,valList)
plt.ylabel('#Num of Courses')
plt.title('Distribution of paid/unpaid courses')
plt.show()

df['subject'].value_counts()
def subSplitter(subList):

    catList=['Web','Business','Musical','Graphics']
    catDict=dict()
    [catDict.update({subList[i]:re.split(' ',subList[i])[0]}) for i in range(len(subList))]

    print(catDict)
    subCategories=list(catDict.values())
    
    return subCategories
subList=list(df['subject'].value_counts().keys())

plt.bar(subSplitter(subList),list(df['subject'].value_counts()))

plt.xlabel('Different subjects')
plt.ylabel('#Num of Courses')
plt.title('Distribution of courses across subjects')
plt.show()
temp=df['num_subscribers'].mean()
print(f'Average number of subscribers is: {temp}')
temp=df['num_reviews'].mean()
print(f'Average number of reviews is :{temp}')
temp=df['num_lectures'].mean()
print(f'Average number of lectures is: {temp}')
temp=df['content_duration'].mean()
print(f'Average content duration is: {temp}')
temp=df['price'].mean()
print(f'Average price is: {temp}')
def levSplitting(levList):
    tempSplit=dict()
    [tempSplit.update({i:re.split(' ',i)[0]}) for i in levList] 
    tempVals=list(tempSplit.values())
    return tempVals
### Categorical features ###
catFeat={'level':'Various levels','subject':'Various Subjects','is_paid':'Whether it was paid?'}

#Numeric features: Subscribers,reviews,lectures,duraiton,price
### Distribution of number of subscribers across various categories ######

for key,values in catFeat.items():
    
    #Number of subscribers across various levels:
    lstSubLev=list(df['num_subscribers'].groupby(df[key]).mean())

    levList=list(df['num_subscribers'].groupby(df[key]).mean().keys())
    tempSplit=[]
    
    if  isinstance(levList[0],bool):
        tempSplit.append(str(levList[0]))
        tempSplit.append(str(levList[1]))
    else:
        tempSplit=levSplitting(levList)

    #Distribution of subscribers across levels:
    plt.figure()
    plt.bar(tempSplit,lstSubLev)
    plt.xlabel(values)
    plt.ylabel('Average number of Subscribers')
    if key=='is_paid':
        plt.title(f'Distribution of average number of subscribers in the above categories vs. if they were priced?')
    else:    
        plt.title(f'Distribution of average number of subscribers vs {values}')
    plt.show()
    
    

### Distribution of number of reviews across various categories ###

for key,values in catFeat.items():
    
    lstSubLev=list(df['num_reviews'].groupby(df[key]).mean())

    levList=list(df['num_reviews'].groupby(df[key]).mean().keys())
    tempSplit=[]
    
    if  isinstance(levList[0],bool):
        tempSplit.append(str(levList[0]))
        tempSplit.append(str(levList[1]))
    else:
        tempSplit=levSplitting(levList)

    plt.figure()
    plt.bar(tempSplit,lstSubLev)
    plt.xlabel(values)
    plt.ylabel('Average number of reviews')
    if key=='is_paid':
        plt.title(f'Distribution of average number of reviews in the above categories vs Free/Priced')
    else:    
        plt.title(f'Distribution of average number of reviews vs {values}')
    plt.show()
    
    
### Distribution of number of Lectures across various categories ###

for key,values in catFeat.items():
    
    lstSubLev=list(df['num_lectures'].groupby(df[key]).mean())

    levList=list(df['num_lectures'].groupby(df[key]).mean().keys())
    tempSplit=[]
    
    if  isinstance(levList[0],bool):
        tempSplit.append(str(levList[0]))
        tempSplit.append(str(levList[1]))
    else:
        tempSplit=levSplitting(levList)

    plt.figure()
    plt.bar(tempSplit,lstSubLev)
    plt.xlabel(values)
    plt.ylabel('Average number of Lectures')
    if key=='is_paid':
        plt.title(f'Distribution of average number of lectures in the above categories vs Free/Priced')
    else:    
        plt.title(f'Distribution of average number of lectures vs {values}')
    plt.show()
    
 
### Distribution of average content hours across various categories ###

for key,values in catFeat.items():
    
    lstSubLev=list(df['content_duration'].groupby(df[key]).mean())

    levList=list(df['content_duration'].groupby(df[key]).mean().keys())
    tempSplit=[]
    
    if  isinstance(levList[0],bool):
        tempSplit.append(str(levList[0]))
        tempSplit.append(str(levList[1]))
    else:
        tempSplit=levSplitting(levList)

    plt.figure()
    plt.bar(tempSplit,lstSubLev)
    plt.xlabel(values)
    plt.ylabel('Average number of hours')
    if key=='is_paid':
        plt.title(f'Distribution of average number of hours in the above categories vs Free/Priced')
    else:    
        plt.title(f'Distribution of average number of hours vs {values}')
    plt.show()
    
### Distribution of average price across various levels ###

for key,values in catFeat.items():
    
    lstSubLev=list(df['price'].groupby(df[key]).mean())

    levList=list(df['price'].groupby(df[key]).mean().keys())
    tempSplit=[]
    
    if  isinstance(levList[0],bool):
        tempSplit.append(str(levList[0]))
        tempSplit.append(str(levList[1]))
    else:
        tempSplit=levSplitting(levList)

    plt.figure()
    plt.bar(tempSplit,lstSubLev)
    plt.xlabel(values)
    plt.ylabel('Average cost')
    plt.title(f'Distribution of average cost vs {values}')
    plt.show()
    
removWords=['of','from','the','and','not','an','then','.','best','minimum','maximam','to','learn',
            'for','with','in','by',' ',';','a',':','','on','using','basics','1','2','3','4','cc','own','you','step','become','&',
           'how','de','complete','design']
wordDict_priced=dict()
wordDict_nonPriced=dict()

for j in range(len(df['course_title'])):
    courTitle=(df['course_title'][j]).lower()
    if df['is_paid'][j]==True:
        temp=[]
        temp=re.split(' ',courTitle)
        for i in temp:
            if i not in removWords:
                if i not in wordDict_priced.keys():
                    wordDict_priced.update({i:1})
                else:
                    wordDict_priced[i]+=1
            else:
                continue
    else:
        temp=[]
        temp=re.split(' ',courTitle)
        for i in temp:
            if i not in removWords:
                if i not in wordDict_nonPriced.keys():
                    wordDict_nonPriced.update({i:1})
                else:
                    wordDict_nonPriced[i]+=1
            else:
                continue
        
sortedDict_priced=sorted(wordDict_priced.items(),key=lambda x: x[1], reverse=True)
cutPt=300
[sortedDict_priced[i] for i in range(1,cutPt)]
sortedDict_nonPriced=sorted(wordDict_nonPriced.items(),key=lambda x: x[1], reverse=True)
cutPt2=80
[sortedDict_nonPriced[i] for i in range(1,100)]
finance=['trading','forex','financial','finance','trade','business','stock','stocks','accounting','bitcoin','tax','equity']


computerSc=['web','photoshop','javascript','wordpress','website','accounting','adobe','html','html5','php','css','bootstrap','jquery','html5',
'angular','websites','developer','app','graphic','programming','Excel','coding','js','rails','ruby','python','angularjs','code',
           'ajax','mysql','robot','asp.net','api','django','json','animation','xml']

music=['guitar','piano','harmonica','drum','jazz','rhythm']


topicDict={'Computer Science':0,'Finance':0,'Music':0}
topicDictFree=topicDict.copy()

tempIdx=[sortedDict_priced,sortedDict_nonPriced]
tempIdxTopicDict=[topicDict,topicDictFree]

for idx in range(len(tempIdx)):
    for i in range(cutPt):
        
        tempTop=tempIdx[idx][i][0]
        tempVal=tempIdx[idx][i][1]
    
        if tempTop in computerSc:
            tempIdxTopicDict[idx]['Computer Science']+=tempVal
        elif tempTop in finance:
            tempIdxTopicDict[idx]['Finance']+=tempVal
        elif tempTop in music:
            tempIdxTopicDict[idx]['Music']+=tempVal
       
    
#For priced courses
print(tempIdxTopicDict[0])

#For free courses
print(tempIdxTopicDict[1])
#For Priced courses
plt.figure()
plt.bar(tempIdxTopicDict[0].keys(),tempIdxTopicDict[0].values())
plt.title('Distribution of Priced courses within these three majors')
plt.ylabel('Number of courses')
plt.show()

#For Free courses
plt.figure()
plt.bar(tempIdxTopicDict[1].keys(),tempIdxTopicDict[1].values())
plt.title('Distribution of free courses within these three majors')
plt.ylabel('Number of courses')
plt.show()
codingCourses=['javascript','html','html5','php','css','bootstrap','angular','js','rails','ruby','python','angularjs','mysql',
               'asp.net','django']

javaScrptCours=['javascript','angular','js','angularjs','react']
webFrameworks=['html','html5','php','css','bootstrap','asp.net','django']
rubyRails=['rails','ruby']
python=['python']
mySQL=['mysql']


### For priced  and Free courses:

topicDict={'Java':0,'Web tools':0,'Ruby on Rails':0,'Python':0,'SQL':0}
topicDictFree=topicDict.copy()

tempIdx=[sortedDict_priced,sortedDict_nonPriced]
tempIdxTopicDict=[topicDict,topicDictFree]

for idx in range(len(tempIdx)):
    for i in range(cutPt):
        
        tempTop=tempIdx[idx][i][0]
        tempVal=tempIdx[idx][i][1]
    
        if tempTop in javaScrptCours:
            tempIdxTopicDict[idx]['Java']+=tempVal
        elif tempTop in webFrameworks:
            tempIdxTopicDict[idx]['Web tools']+=tempVal
        elif tempTop in rubyRails:
            tempIdxTopicDict[idx]['Ruby on Rails']+=tempVal
        elif tempTop in python:
            tempIdxTopicDict[idx]['Python']+=tempVal
        elif tempTop in mySQL:
            tempIdxTopicDict[idx]['SQL']+=tempVal
    
#For priced courses
print(tempIdxTopicDict[0])

#For free courses
print(tempIdxTopicDict[1])
#For priced courses
plt.figure()
plt.bar(tempIdxTopicDict[0].keys(),tempIdxTopicDict[0].values())
plt.title('Distribution of Priced coding/scripting/programming courses')
plt.ylabel('Number of courses')
plt.show()

#For free courses
plt.figure()
plt.bar(tempIdxTopicDict[1].keys(),tempIdxTopicDict[1].values())
plt.title('Distribution of Free coding/scripting/programming courses')
plt.ylabel('Number of courses')
plt.show()
javaScrptCours="javascript,angular,js,angularjs,react"
webFrameworks="html,html5,php,css,bootstrap,asp.net,django"
rubyRails="rails,ruby"
python='python'
mySQL='mysql,'

courseMatch={javaScrptCours:'Java',webFrameworks:'Web tools',rubyRails:'Ruby+Rails',python:'Python&Web',mySQL:'SQL'}
priceDict={'Java':0,'Web tools':0,'Ruby+Rails':0,'Python&Web':0,'SQL':0}
subsDict=priceDict.copy()
revDict=priceDict.copy()
titleList=[]
for i in range(len(df['course_title'])):
    
    tempPrice=[]
    tempPrice=df['price'][i]
    tempSubs=df['num_subscribers'][i]
    tempRev=df['num_reviews'][i]
    
    chngdTitle=(df['course_title'][i]).lower().replace('[!,?,.,\',[0-9]]','')
    titleList.append(chngdTitle)
    tempTitle=re.split(' ',chngdTitle)
    for j in courseMatch.keys():
        tempLst=[]
        tempLst.append(re.split(',',j)[0])
        if set(tempLst).intersection(set(tempTitle)):
            priceDict[courseMatch[j]]+=tempPrice
            subsDict[courseMatch[j]]+=tempSubs
            revDict[courseMatch[j]]+=tempRev

            
print(priceDict)
print(subsDict)
print(revDict)
#For num. of Subscribers:
plt.figure()
plt.bar(subsDict.keys(),subsDict.values())
plt.title('Approx. distribution of number of subscribers for the various category of coding/programming courses')
plt.ylabel('Number of subscribers')
plt.show()

#For num. of Reviews:
plt.figure()
plt.bar(revDict.keys(),revDict.values())
plt.title('Approx. distribution of number of reviews for the various category of coding/programming courses ')
plt.ylabel('Number of reviews')
plt.show()

#For total price:
plt.figure()
plt.bar(priceDict.keys(),priceDict.values())
plt.title('Approx. distribution of cost of course for the various category of coding courses')
plt.ylabel('Total cost')
plt.show()
### Priced  and Free courses ###
#Web tools based courses ###

webFrameworks=['html','html5','php','css','bootstrap','asp.net','django']

topicDict={'HTML':0,'HTML5':0,'PHP':0,'CSS':0,'Bootstrap':0,'ASP.NET':0,'Django':0}
topicDictFree=topicDict.copy()

tempIdx=[sortedDict_priced,sortedDict_nonPriced]
tempIdxTopicDict=[topicDict,topicDictFree]

for idx in range(len(tempIdx)):
    for i in range(cutPt):
        
        tempTop=tempIdx[idx][i][0]
        tempVal=tempIdx[idx][i][1]
        
        if tempTop=='html':
            tempIdxTopicDict[idx]['HTML']+=tempVal
        elif tempTop=='html5':
            tempIdxTopicDict[idx]['HTML5']+=tempVal
        elif tempTop=='php':
            tempIdxTopicDict[idx]['PHP']+=tempVal
        elif tempTop =='css':
            tempIdxTopicDict[idx]['CSS']+=tempVal
        elif tempTop == 'bootstrap':
            tempIdxTopicDict[idx]['Bootstrap']+=tempVal
        elif tempTop == 'asp.net':
            tempIdxTopicDict[idx]['ASP.NET']+=tempVal
        elif tempTop == 'django':
            tempIdxTopicDict[idx]['Django']+=tempVal    
            
    
#For priced courses
print(tempIdxTopicDict[0])

#For free courses
print(tempIdxTopicDict[1])
#For priced courses
plt.figure()
plt.bar(tempIdxTopicDict[0].keys(),tempIdxTopicDict[0].values())
plt.title('Approx. distribution of Priced courses on Web or Front end development')
plt.ylabel('Number of courses')
plt.show()

#For free courses
plt.figure()
plt.bar(tempIdxTopicDict[1].keys(),tempIdxTopicDict[1].values())
plt.title('Approx. distribution of Free courses on Web or Front end development')
plt.ylabel('Number of courses')
plt.show()
#With respect to Price:

plt.figure()
plt.scatter(df['num_reviews'],df['price'])
plt.title('Number of reviews a course received vs. Price of the course')
plt.show()

plt.figure()
plt.scatter(df['num_lectures'],df['price'])
plt.title('Number of Lectures a course had vs. Price of the course')
plt.show()

plt.figure()
plt.scatter(df['num_subscribers'],df['price'])
plt.title('Number of Subscribers a course had vs. Price of the course')
plt.show()

plt.figure()
plt.scatter(df['content_duration'],df['price'])
plt.title('Number of Hours of content a course had vs. Price of the course')
plt.show()
plt.figure()
plt.scatter(df['num_reviews'],df['num_subscribers'])
plt.title('Number of Reviews a course had vs. Number of subscribers')
plt.show()
plt.figure()
plt.scatter(df['num_lectures'],df['num_reviews'])
plt.title('Number of Lectures vs. Number of reviews')
plt.show()
