

import pandas as pd

import numpy as np

import seaborn as sb

import matplotlib.pyplot as mpl

#from sklearn.cross_validation import train_test_split as tts

import re

from nltk.tokenize import word_tokenize as wt
filename='../input/train.csv'

try:

    train=pd.read_csv(filename)

    print("Dataset %s successfully loaded"%filename)

except Exception as k:

    print(k)

    raise



train.shape

train.head()

train.tail()

train.describe()
#distribution of the target variable---------------------------------------------------------------------------------------------

features=[k for k in train]

sb.countplot(train['%s'%features[1]])

positives=np.sum(train['Survived'])/len(train['Survived'])

print('%s percent positives'%(round(positives,3)*100))
#Handling missing values---------------------------------------------------------------------------------------------------------

print('Missing Values in each feature')

for feat in features:

    print('%s: %s'%(feat,train[feat].isnull().sum()))

#for Age, I will fill the missing values with the mean age as the age

#has an approximate bell shape with a skew to the right

def missing(data):

    data['Age']=data['Age'].fillna(data['Age'].mean())

    #there are only 2 missing values for enbarked I fill them with the prominent class 'S'

    print(list(set(data['Embarked'])))

    data['Embarked']=data['Embarked'].fillna('S')



missing(train)
#Dummy variables----------------------------------------------------------------------------------------------------------------

print('DataType for each feature')

for feat in features:

    print('%s: %s'%(feat,type(train[feat][0])))



train['Sex']=train['Sex'].map({'male':1,'female':0})





train['Embarked']=train['Embarked'].map({'S':0,'C':1,'Q':2})

#----------------------------------feature Engineering---------------------------------------------------------------------------

#from the name of the passenger, we can engineer a new feature, the title(Mr,Mrs,Miss,Dr)---------------------------------------

def strCleaner(text):

    return re.sub('[-!$%^&*\{\}\[\]#\(\)\'\"\:\;,.?\/]',' ',str(text)).upper()



def Passangertitle(k):

    toked=wt(k)

    titles=['MR','MRS','MISS','DR','COL','LADY','MASTER','SIR','MAYOR','DON','DUKE','REV']

    if toked[1] in titles:

        return toked[1]

    elif toked[2] in titles:

        return toked[2]

    else:

        return 'NONE'



try_=train['Name'].apply(strCleaner)

train['title']=try_.apply(Passangertitle)

sb.countplot(train['title'])
def reduce_titles(title):

    if title not in ['MR','MRS','MISS','MASTER','DR']:

        return 'OTHER'

    else:

        return title



train['title']=train['title'].apply(reduce_titles)

train['title']=train['title'].map({'MR':0,'MRS':1,'MISS':2,'MASTER':3,'DR':4,'OTHER':5})



del train['Name']
#Cabin has a lot of missing values but is still worth investigating

print(train['Cabin'].head())

print(train['Cabin'].describe())

train['Cabin']=train['Cabin'].apply(strCleaner)
#               ----strip first character of the cabin instead------

train['Cabin_']=list(map(lambda x:x[0],train['Cabin']))

sb.countplot(train['Cabin_'])



# Group similar behaving cabins to one (G,A,F,T) and have the rest form dummies (N,C,E,B,D)

def reduce_cabin(cab):

    if cab not in ['N','C','E','B','D']:

        return 'Z'

    else:

        return cab

        

train['Cabin_']=train['Cabin_'].apply(reduce_cabin)

train['Cabin']=train['Cabin_'].map({'Z':0,'N':1,'C':2,'E':3,'B':4,'D':5})

del train['Cabin_']


kk=sb.FacetGrid(train,col='Cabin',hue='Survived')

kk.map(mpl.scatter,'Age','Fare')

kk.add_legend()
#Ticket doesnt seem to have any obvious pattern, I will look at the length of ticket#

train['Ticket_len']=train['Ticket'].apply(len)

del train['Ticket']

#we an get the family_size from combining the SibSp and Parch fields

train['Family_size']=train['SibSp']+train['Parch']

sb.countplot(train['Family_size'])
#Validating age and fare--------------------------------------------------------------------------------------------------------------

def visualize_var(variable):

    minimum=round(np.min(train[variable]),1)

    median=round(np.median(train[variable]),1)

    mode=round(train[variable].mode()[0],1)

    maximum=round(np.max(train[variable]),1)

    variance=round(np.std(train[variable]),1)

    mpl.hist(train[variable],bins=50,range=(-5,100))

    mpl.ylabel('Count')

    mpl.title('%s Distribution\nmin:%s    median:%s     mode:%s     max:%s    std_dev:%s'%(variable,minimum,median,mode,maximum,variance))

    mpl.xlabel(variable)

    mpl.show()

    

visualize_var('Fare')
visualize_var('Age')


#sex & Title across (age and fair)

kk=sb.FacetGrid(train,row='Sex',col='title',hue='Survived')

kk.map(mpl.scatter,'Age','Fare')

kk.add_legend()

kk=sb.FacetGrid(train,col='Embarked',hue='Survived')

kk.map(mpl.scatter,'Age','Fare')

kk.add_legend()
sb.violinplot('Cabin','Fare',hue='Survived',data=train,split=True)