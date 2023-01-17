import os
import pandas as pd

input_io_dir="../input"

original_train_data=pd.read_csv(input_io_dir+"/train.csv")
original_test_data=pd.read_csv(input_io_dir+"/test.csv")
print('original_train_data',original_train_data.shape)
print('original_test_data',original_test_data.shape)
original_train_data.head()
print('Training data --------------')
print(original_train_data.info())
print('Test data ------------------')
print(original_test_data.info())
original_train_data.describe()
original_test_data.describe()
def ExploreCategoricalVariable(dataSet,variableName):
    print('Variable:'+variableName)
    print(dataSet[variableName].value_counts()/len(dataSet[variableName]))
    print('')

print('----------------------- Training set')
ExploreCategoricalVariable(original_train_data,'Sex')
ExploreCategoricalVariable(original_train_data,'Pclass')
ExploreCategoricalVariable(original_train_data,'Embarked')
print('----------------------- Test set')
ExploreCategoricalVariable(original_test_data,'Sex')
ExploreCategoricalVariable(original_test_data,'Pclass')
ExploreCategoricalVariable(original_test_data,'Embarked')
%matplotlib inline
import matplotlib.pyplot as plt
fig, axarr = plt.subplots(4, 2, figsize=(12, 8))

original_train_data['Age'].hist(ax=axarr[0][0])
original_test_data['Age'].hist(ax=axarr[0][1])
original_train_data['Fare'].hist(ax=axarr[1][0])
original_test_data['Fare'].hist(ax=axarr[1][1])
original_train_data['Parch'].hist(ax=axarr[2][0])
original_test_data['Parch'].hist(ax=axarr[2][1])
original_train_data['SibSp'].hist(ax=axarr[3][0])
original_test_data['SibSp'].hist(ax=axarr[3][1])
original_train_data.corr()
original_train_data.groupby('Sex')['Survived'].sum().plot.bar(stacked=True)
original_train_data.groupby('Embarked')['Survived'].sum().plot.bar(stacked=True)
original_train_data.head()
import numpy as np

def extractTitleFromNameForExploring(name):
    pos_point=name.find('.')
    if pos_point == -1: return ""
    wordList=name[0:pos_point].split(" ")
    if len(wordList)<=0: return ""
    title=wordList[len(wordList)-1]
    return title

# Get a list with different titles
training_titleList=np.unique(original_train_data['Name'].apply(lambda x: extractTitleFromNameForExploring(x)))
for title in training_titleList:
    training_titleSet=original_train_data[original_train_data['Name'].apply(lambda x: title in x)]
    # Evaluate survival rate for each subset
    survivalRate=float(len(training_titleSet[training_titleSet['Survived']==1]))/float(len(training_titleSet))
    print('Title['+title+'] count:'+str(len(training_titleSet))+' survival rate:'+str(survivalRate))
# Let's check test data set values - just to confirm the training will consider all potential values
test_titleList=np.unique(original_test_data['Name'].apply(lambda x: extractTitleFromNameForExploring(x)))
for title in test_titleList:
    test_titleSet=original_test_data[original_test_data['Name'].apply(lambda x: title in x)]
    print('Title['+title+'] count:'+str(len(test_titleSet)))
import numpy as np

def multipleReplace(text, wordDic):
    for key in wordDic:
        if text.lower()==key.lower():
            text=wordDic[key]
            break
    return text

def normaliseTitle(title):
    wordDic = {
    'Mlle': 'Miss',
    'Ms': 'Mrs',
    'Mrs':'Mrs',
    'Master':'Master',
    'Mme': 'Mrs',
    'Lady': 'Nobility',
    'Countess': 'Nobility',
    'Capt': 'Army',
    'Col': 'Army',
    'Dona': 'Other',
    'Don': 'Other',
    'Dr': 'Other',
    'Major': 'Army',
    'Rev': 'Other',
    'Sir': 'Other',
    'Jonkheer': 'Other',
    }     
    title=multipleReplace(title,wordDic)
    return title
def extractTitleFromName(name):
    pos_point=name.find('.')
    if pos_point == -1: return ""
    wordList=name[0:pos_point].split(" ")
    if len(wordList)<=0: return ""
    title=wordList[len(wordList)-1]
    normalisedTitle=normaliseTitle(title)
    return normalisedTitle

# Get a list with different titles
titleList=np.unique(original_train_data['Name'].apply(lambda x: extractTitleFromName(x)))
for title in titleList:
    titleSet=original_train_data[original_train_data['Name'].apply(lambda x: title in extractTitleFromName(x))]
    # Evaluate survival rate for each subset
    survivalRate=float(len(titleSet[titleSet['Survived']==1]))/float(len(titleSet))
    print('Title['+title+'] count:'+str(len(titleSet))+' survival rate:'+str(survivalRate))
original_train_data['IsAlone']=(original_train_data["SibSp"]+original_train_data["Parch"]).apply(lambda x: 0 if x>0 else 1)
original_train_data['FamilySize']=original_train_data["SibSp"]+original_train_data["Parch"]+1

original_train_data.corr()
import numpy as np
total=original_train_data.groupby('IsAlone')['PassengerId'].count()
survived=original_train_data[original_train_data['Survived']==1].groupby('IsAlone')['PassengerId'].count()
notSurvived=original_train_data[original_train_data['Survived']==0].groupby('IsAlone')['PassengerId'].count()
df=pd.concat([total, survived,notSurvived], axis=1, sort=True)
df.fillna(0,inplace=True)
df.columns=['Total','Survived','NotSurvived']
df=df.astype('int64')
print(df)
df.loc[:,['Survived','NotSurvived']].plot.bar(stacked=True,figsize=(20,8))
print("FamilySize value distribution")
print(original_train_data['FamilySize'].value_counts()/len(original_train_data))
import numpy as np
total=original_train_data.groupby('FamilySize')['PassengerId'].count()
survived=original_train_data[original_train_data['Survived']==1].groupby('FamilySize')['PassengerId'].count()
notSurvived=original_train_data[original_train_data['Survived']==0].groupby('FamilySize')['PassengerId'].count()
df=pd.concat([total, survived,notSurvived], axis=1, sort=True)
df.fillna(0,inplace=True)
df.columns=['Total','Survived','NotSurvived']
df=df.astype('int64')
print(df)
df.loc[:,['Survived','NotSurvived']].plot.bar(stacked=True,figsize=(20,8))
original_train_data['NoCabin']=original_train_data['Cabin'].isnull().apply(lambda x: 1 if x is True else 0)
original_train_data.corr()
total=original_train_data.groupby('NoCabin')['PassengerId'].count()
survived=original_train_data[original_train_data['Survived']==1].groupby('NoCabin')['PassengerId'].count()
notSurvived=original_train_data[original_train_data['Survived']==0].groupby('NoCabin')['PassengerId'].count()
df=pd.concat([total, survived,notSurvived], axis=1, sort=True)
df.fillna(0,inplace=True)
df.columns=['Total','Survived','NotSurvived']
df=df.astype('int64')
print(df)
df.loc[:,['Survived','NotSurvived']].plot.bar(stacked=True,figsize=(20,8))
import numpy as np
# Group data to detect sharing of cabins - excluding missing  values
cabinList=original_train_data[original_train_data['Cabin'].notnull()==True].groupby('Cabin')['PassengerId'].count()
cabinList=cabinList.reset_index()
cabinList.columns=['Cabin','Count']
print('Distribution of people per cabin - not considering those with missing cabin')
print(cabinList['Count'].value_counts())

# Add new column to indicate number of people a passenger is sharing with
# -1 means there is no data to compute the feature
def extractCabinSharedWithFeature(name):
    if (str(name)!='nan'):
        row=cabinList.loc[cabinList['Cabin'] == name]
        count=row['Count']-1
        return count
    else:
        return -1

original_train_data['CabinSharedWith']=original_train_data['Cabin'].apply(lambda x: extractCabinSharedWithFeature(x)).astype(int)
# Let's now analyse this new column
total=original_train_data[original_train_data['CabinSharedWith']!=-1]['PassengerId'].count()
survived=original_train_data[(original_train_data['CabinSharedWith']!=-1) & (original_train_data['Survived']==1)].groupby('CabinSharedWith')['PassengerId'].count()
notSurvived=original_train_data[(original_train_data['CabinSharedWith']!=-1) & (original_train_data['Survived']==0)].groupby('CabinSharedWith')['PassengerId'].count()
survivedPercent=survived/total
notSurvivedPercent=notSurvived/total
print('Survivor distribution by feature CabinSharedWith')
print(survivedPercent)
print('NotSurvivor distribution by feature CabinSharedWith')
print(notSurvivedPercent)
df=pd.concat([survived,survivedPercent,notSurvived,notSurvivedPercent], axis=1, sort=True)
df.fillna(0,inplace=True)
df.columns=['Survived','SurvivedPercent','NotSurvived','NotSurvivedPercent']
df.loc[:,['Survived','NotSurvived']].plot.bar(stacked=True,figsize=(20,8))
original_train_data.corr()
original_train_data['Ticket'].head(10)
def getTicketType(name, normalise):
    item=name.split(' ')
    itemLength=len(item)
    if itemLength>1:
        ticketType=""
        for i in range(0,itemLength-1):
            ticketType+=item[i].upper()
    else:
        ticketType="NORMAL"
    if normalise==True:
        ticketType= ticketType.translate(str.maketrans('','','./'))
    return ticketType

# Let's list what we have - first view without normalising
training_itemList=[]
for ticket in original_train_data['Ticket']:
    training_itemList.append(getTicketType(ticket,False))
ticketTypeList=np.unique(training_itemList)
print("Ticket type values: no normalisation")
print(ticketTypeList)
training_itemList=[]
for ticket in original_train_data['Ticket']:
    training_itemList.append(getTicketType(ticket,True))
ticketTypeList=np.unique(training_itemList)
print("Ticket type values: normalisation")
print(ticketTypeList)
pd.set_option('display.max_columns', None)
original_train_data['TicketType']=original_train_data['Ticket'].apply(lambda x: getTicketType(x,True))
total=pd.DataFrame(original_train_data.groupby('TicketType')['PassengerId'].count())
total.columns=['Total']
survived=pd.DataFrame(original_train_data[original_train_data['Survived']==1].groupby('TicketType')['PassengerId'].count())
survived.columns=['Survived']
notSurvived=pd.DataFrame(original_train_data[original_train_data['Survived']==0].groupby('TicketType')['PassengerId'].count())
notSurvived.columns=['NotSurvived']
# Let's merge all ticket type in the same list
df_all=total
df_all=df_all.merge(survived,left_index=True, right_on="TicketType")
df_all=df_all.merge(notSurvived,left_on='TicketType',left_index=True, right_on="TicketType")
df_all['Ratio']=df_all['Survived']/df_all['Total']
df_all.loc[:,['Ratio']].plot.bar(figsize=(20,8))
df_all

original_train_data[original_train_data['TicketType']=='FCC']