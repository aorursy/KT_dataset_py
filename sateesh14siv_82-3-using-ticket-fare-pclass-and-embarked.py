import pandas as pd
import numpy as np
train, test = pd.read_csv('../input/train.csv'), pd.read_csv('../input/test.csv')
train.head()
# Men who have the tag Master are classified as boys.
boys = train[train['Name'].str.contains('Master')]
females = train[train['Sex'] == 'female']

boy_or_female = boys | females
#Creating a new feature called Surname
train['Surname'] = [train.iloc[i]['Name'].split(',')[0] for i in range(len(train))]
test['Surname'] = [test.iloc[i]['Name'].split(',')[0] for i in range(len(test))]
# Survival rate of the boy-female groups grouped by surname.
boy_femSurvival = train.loc[boy_or_female.index].groupby('Surname')['Survived'].mean().to_frame()
boy_femSurvival
# Boys survive in groups where all boys and females survive.
# Females die in the groups where all boys and females die.
boysSurvived = list(boy_femSurvival[boy_femSurvival['Survived'] == 1].index) # List of families (surname) where all boys and females survived.
femDie = list(boy_femSurvival[boy_femSurvival['Survived'] == 0].index) # List of families where all boys and females died.
def createFeatures(frame):
    frame['Boy'] = frame['Name'].str.contains('Master').astype(int)
    frame['Female'] = (frame['Sex']=='female').astype(int) 
    return frame

def makePredictions(row):
    if row['Boy'] and (row['Surname'] in boysSurvived):
        return 1
        
    if row['Female']:
        if row['Surname'] in femDie:
            return 0
        else:
            return 1
            
    else:
        return 0
test = createFeatures(test)
test.head()
pred = test.apply(makePredictions, axis = 1)
sub = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':pred})
sub.to_csv('SurnameModel.csv', index = False)
boy_count = 0
femCount = 0
for index, row in sub.iterrows():
    if test.iloc[index]['Sex'] == 'female' and row[1] == 0:
        print('femPassenger:',row[0])
        femCount = femCount + 1
    elif test.iloc[index]['Sex'] == 'male' and row[1] == 1:
        print('boy:',row[0])
        boy_count = boy_count + 1
print('BoyCount:',boy_count)
print('FemCount',femCount)
print(100*train.isnull().sum().sort_values(ascending = False)/len(train))
test.isnull().sum().sort_values(ascending = False)
train.drop('Cabin', axis = 1, inplace = True)
test.drop('Cabin', axis = 1, inplace = True)
train[train['Embarked'].isnull() == True]
train[train['Ticket'] == '113572']
train[train['Surname'] == 'Icard']
ticketTrain = train.set_index('Ticket')
train.groupby(['Ticket', 'Surname', 'Name'])['Survived'].mean().to_frame()
frame = train.groupby(['Surname', 'Ticket','Fare','Pclass', 'Name', 'Embarked'])['Survived'].mean().to_frame()
frame
frame.loc['Andersson']
test['Embarked'].fillna('S', inplace = True)
test.isnull().sum().sort_values(ascending = False)
test[test['Fare'].isnull()]
test[test['Ticket'] == '3701']
fill = test[test['Pclass']==3]['Fare'].mean()
test['Embarked'].fillna(fill, inplace = True)
train['GroupID'] = train.apply(lambda row : str(row[8][:-1]) + '-' + str(row[2]) + '-' + str(row[9]) + str(row[10]), axis = 1)
test['GroupID'] = test.apply(lambda row : str(row[7][:-1]) + '-' + str(row[1]) + '-' + str(row[8]) + str(row[9]), axis = 1)
train
GroupTable = train.pivot_table(index = 'GroupID', values = 'Survived', aggfunc=[len, np.mean])
GroupTable
print('Number of unique surnames:',len(train['Surname'].unique()))
print('Number of groups (Families/ Friends ..):',len(train['GroupID'].unique()))
test.head()
boy_femSurvival = train.loc[boy_or_female.index].groupby('GroupID')['Survived'].mean().to_frame()
boy_femSurvival.head()
# These are the groups in which all the boys and females survived.
boysSurvived = list(boy_femSurvival[boy_femSurvival['Survived'] == 1].index)
print(len(boysSurvived))
boysSurvived
# These are the groups in which all the females and boys died.
femDie = list(boy_femSurvival[boy_femSurvival['Survived'] == 0].index)
print(len(femDie))
femDie
def makePredictions2(row):
    if row['Boy'] and (row['GroupID'] in boysSurvived):
        return 1
        
    if row['Female']:
        if row['GroupID'] in femDie:
            return 0
        else:
            return 1
            
    else:
        return 0
pred2 = test.apply(makePredictions2, axis = 1)
sub = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':pred2})
sub.to_csv('TicketModel.csv', index = False)
boy_count = 0
femCount = 0
for index, row in sub.iterrows():
    if test.iloc[index]['Sex'] == 'female' and row[1] == 0:
        print('femPassenger:',row[0])
        femCount = femCount + 1
    elif test.iloc[index]['Sex'] == 'male' and row[1] == 1:
        print('boy:',row[0])
        boy_count = boy_count + 1
print('BoyCount:',boy_count)
print('FemCount',femCount)
print('Total number of females:',len(train[train['Sex'] == 'female']) + len(test[test['Sex'] == 'female']))
print('Percentage of females that survived in the train set:', len(train[(train['Sex']=='female') & (train['Survived']==1)])*100/len(train[train['Sex'] == 'female']))
print('Percentage of females that survived in the test set:', 100*(len(test[test['Sex'] == 'female'])-13)/len(test[test['Sex'] == 'female']))
print('Total number of males:',len(train[train['Sex'] == 'male']) + len(test[test['Sex'] == 'male']))
print('Percentage of males that survived in the train set:', len(train[(train['Sex']=='male') & (train['Survived']==1)])*100/len(train[train['Sex'] == 'male']))
print('Percentage of males that survived in the test set:', 100*9/len(test[test['Sex'] == 'male']))