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
df_rows = pd.read_csv("../input/train.csv")
df_rows.head()
df_rows.describe()
survived_sex = df_rows[df_rows['Survived']==1]['Sex'].value_counts()
print(survived_sex)
dead_sex = df_rows[df_rows['Survived']==0]['Sex'].value_counts()
print(dead_sex)
df = pd.DataFrame([survived_sex,dead_sex],index = ['Survived','Dead'])
df.plot(kind='bar',stacked=True, figsize=(15,8))
import matplotlib.pyplot as plt
x = np.random.random_integers(1, 100, 5)
print(x)
plt.hist(x, bins=10)
plt.ylabel('No of times')
plt.show()
#How about survival and age 
figure = plt.figure(figsize=(15,8))
plt.hist([df_rows[df_rows['Survived']==1]['Age'],df_rows[df_rows['Survived']==0]['Age']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()
#What do you think about fare?
figure = plt.figure(figsize=(15,8))
plt.hist([df_rows[df_rows['Survived']==1]['Fare'],df_rows[df_rows['Survived']==0]['Fare']], stacked=True, color = ['g','r'],
         bins = 30,label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()
plt.figure(figsize=(15,8))
ax = plt.subplot()
ax.scatter(df_rows[df_rows['Survived']==1]['Age'],df_rows[df_rows['Survived']==1]['Fare'],c='green',s=40)
ax.scatter(df_rows[df_rows['Survived']==0]['Age'],df_rows[df_rows['Survived']==0]['Fare'],c='red',s=40)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.legend(('survived','dead'),scatterpoints=1,loc='upper right',fontsize=15,)
ax = plt.subplot()
ax.set_ylabel('Average fare')
df_rows.groupby('Pclass').mean()['Fare'].plot(kind='bar',figsize=(15,8), ax = ax)
#What are you priors about survival and class of a passenger
survived_embark = df_rows[df_rows['Survived']==1]['Embarked'].value_counts()
dead_embark = df_rows[df_rows['Survived']==0]['Embarked'].value_counts()
df = pd.DataFrame([survived_embark,dead_embark])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(15,8))
#What are you priors about survival and class of a passenger
survived_embark = df_rows[df_rows['Survived']==1]['Pclass'].value_counts()
dead_embark = df_rows[df_rows['Survived']==0]['Pclass'].value_counts()
df = pd.DataFrame([survived_embark,dead_embark])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(15,8))
#Let us look at both training and test dataset now
def get_combined_data():
    # reading train data
    train = pd.read_csv('../input/train.csv')
    
    # reading test data
    test = pd.read_csv('../input/test.csv')

    # extracting and then removing the targets from the training data 
    targets = train.Survived
    train.drop('Survived',1,inplace=True)
    

    # merging train data and test data for future feature engineering
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index',inplace=True,axis=1)
    
    return combined
combined = get_combined_data()
combined.shape
combined.head()
#let us now try and get some more information
def get_titles():

    global combined
    
    # we extract the title from each name
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # a map of more aggregated titles
    Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
    
    # we map each title
    combined['Title'] = combined.Title.map(Title_Dictionary)
get_titles()
combined.head()
#Playing with ages a bit more creatively
grouped = combined.groupby(['Sex','Pclass','Title'])
grouped.median()
def process_age():
    
    global combined
    
    # a function that fills the missing values of the Age variable
    
    def fillAges(row):
        if row['Sex']=='female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return 30
            elif row['Title'] == 'Mrs':
                return 45
            elif row['Title'] == 'Officer':
                return 49
            elif row['Title'] == 'Royalty':
                return 39

        elif row['Sex']=='female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return 20
            elif row['Title'] == 'Mrs':
                return 30

        elif row['Sex']=='female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return 18
            elif row['Title'] == 'Mrs':
                return 31

        elif row['Sex']=='male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 41.5
            elif row['Title'] == 'Officer':
                return 52
            elif row['Title'] == 'Royalty':
                return 40

        elif row['Sex']=='male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return 2
            elif row['Title'] == 'Mr':
                return 30
            elif row['Title'] == 'Officer':
                return 41.5

        elif row['Sex']=='male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 26
    
    combined.Age = combined.apply(lambda r : fillAges(r) if np.isnan(r['Age']) else r['Age'], axis=1)
    
process_age()
combined.info()

def process_names():
    
    global combined
    # we clean the Name variable
    combined.drop('Name',axis=1,inplace=True)
    
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(combined['Title'],prefix='Title')
    combined = pd.concat([combined,titles_dummies],axis=1)
    
    # removing the title variable
    combined.drop('Title',axis=1,inplace=True)
process_names()
combined.head()
def process_fares():
    
    global combined
    # there's one missing fare value - replacing it with the mean.
    combined.Fare.fillna(combined.Fare.mean(),inplace=True)
process_fares()
combined.head()
def process_embarked():
    
    global combined
    # two missing embarked values - filling them with the most frequent one (S)
    combined.Embarked.fillna('S',inplace=True)
    
    # dummy encoding 
    embarked_dummies = pd.get_dummies(combined['Embarked'],prefix='Embarked')
    combined = pd.concat([combined,embarked_dummies],axis=1)
    combined.drop('Embarked',axis=1,inplace=True)
process_embarked()
combined.head()
def process_cabin():
    
    global combined
    
    # replacing missing cabins with U (for Uknown)
    combined.Cabin.fillna('U',inplace=True)
    
    # mapping each Cabin value with the cabin letter
    combined['Cabin'] = combined['Cabin'].map(lambda c : c[0])
    
    # dummy encoding ...
    cabin_dummies = pd.get_dummies(combined['Cabin'],prefix='Cabin')
    
    combined = pd.concat([combined,cabin_dummies],axis=1)
    
    combined.drop('Cabin',axis=1,inplace=True)
process_cabin()
combined.head()
combined.info()
def process_sex():
    
    global combined
    # mapping string values to numerical one 
    combined['Sex'] = combined['Sex'].map({'male':1,'female':0})
process_sex()
combined.head()
def process_pclass():
    
    global combined
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(combined['Pclass'],prefix="Pclass")
    
    # adding dummy variables
    combined = pd.concat([combined,pclass_dummies],axis=1)
    
    # removing "Pclass"
    
    combined.drop('Pclass',axis=1,inplace=True)
process_pclass()
combined.dtypes
def process_ticket():
    
    global combined
    
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t : t.strip() , ticket)
        ticket = list(filter(lambda t : not t.isdigit(), ticket))
        if len(ticket) > 0:
            return ticket[0]
        else: 
            return 'XXX'
    

    # Extracting dummy variables from tickets:

    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'],prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies],axis=1)
    combined.drop('Ticket',inplace=True,axis=1)
process_ticket()
def process_family():
    
    global combined
    # introducing a new feature : the size of families (including the passenger)
    combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
    
    # introducing other features based on the family size
    combined['Singleton'] = combined['FamilySize'].map(lambda s : 1 if s == 1 else 0)
    combined['SmallFamily'] = combined['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)
    combined['LargeFamily'] = combined['FamilySize'].map(lambda s : 1 if 5<=s else 0)
process_family()
combined.head()
def scale_all_features():
    
    global combined
    
    features = list(combined.columns)
    features.remove('PassengerId')
    combined[features] = combined[features].apply(lambda x: x/x.max(), axis=0)
    
    print('Features scaled successfully !')
combined.dtypes
scale_all_features()
combined.head()
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
 #Split data to validation and train
train = pd.read_csv('../input/train.csv')
    
    # reading test data
test = pd.read_csv('../input/test.csv')
m = len(train)
n = len(test)

x_train = combined[0:m]
x_test = combined[m:m + n]

train_percent = 0.80
validate_percent = 0.20


m = len(x_train)
y_train = train['Survived']
x_train = x_train[:int(train_percent * m)]
x_validation = x_train[int(validate_percent * m):]
y_train = y_train[:int(train_percent * m)]
y_validation = y_train[int(validate_percent * m):]
def get_result(predicted):
    print("F1_Score: " + str(f1_score(y_validation, predicted, average='macro')))
    print("accuracy: " + str(accuracy_score(y_validation, predicted)))
    print("AUC: " + str(roc_auc_score(y_validation, predicted)))
    print("recall: " + str(recall_score(y_validation, predicted)))
    return
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

# Random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc = rfc.fit(x_train, y_train)
#y_predicted_validation_rfc = rfc.predict(x_validation)
y_prediction_test_rfc = rfc.predict(x_test)

print("- Random forest -")
get_result(y_predicted_validation_rfc)
y_prediction_test_rfc             
x_test.head()
raw_data = {'PassengerId' : x_test.PassengerId, 'Survived' : y_prediction_test_rfc}
df = pd.DataFrame(raw_data, columns = ['PassengerId', 'Survived'])

df.head()
df.to_csv('submission.csv', encoding='utf-8', mode = 'w', index=False)
data = pd.read_csv('submission.csv')
data.head()