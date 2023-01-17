import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# Local notebook
# train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')

# Cloud notebook
train = pd.read_csv(os.path.join('../input', 'train.csv'))
test = pd.read_csv(os.path.join('../input', 'test.csv'))
train.info()
train.head()
train = train.drop(['PassengerId'], axis = 1)
train.head()
# fijar normalize como True hace que el resultado sea porcentual (o relativo a la cantidad de observaciones de la muestra).
train['Survived'].value_counts(normalize = True)
train['Survived'].value_counts()
sns.countplot(train['Survived'])
train['Pclass'].value_counts()
train['Survived'].groupby(train['Pclass']).mean()
sns.countplot(train['Pclass'], hue=train['Survived'])
train['Name'].head()
train['Name_Title'] = train['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
train['Name_Title'].value_counts()
train['Name_Title2'] = train['Name_Title'].replace(['Lady.','Mme.','Countess.','Capt.','Col.','Don.',
                                                    'Major.', 'Sir.', 'Jonkheer.', 'Dona.', 'Ms.',
                                                    'Mlle.', 'the'], 'other')
sns.countplot(train['Name_Title2'], hue=train['Survived'])
train['Survived'].groupby(train['Name_Title2']).mean()
train['Name_Len'] = train['Name'].apply(lambda x: len(x))
train['Survived'].groupby(pd.qcut(train['Name_Len'],7)).mean()
pd.qcut(train['Name_Len'],7).value_counts()
sns.countplot(pd.qcut(train['Name_Len'],7), hue=train['Survived'])
sns.boxplot(x=train['Survived'], y=train['Name_Len'], palette="Set1")
train['ParenthesisInName'] = train['Name'].apply(lambda x: "(" in x)
train['ParenthesisInName'].value_counts()
train['Survived'].groupby(train['ParenthesisInName']).mean()
sns.countplot(train['ParenthesisInName'], hue=train['Survived'])
train['QuoteInName'] = train['Name'].apply(lambda x: "\"" in x)
train['QuoteInName'].value_counts()
train['Survived'].groupby(train['QuoteInName']).mean()
sns.countplot(train['QuoteInName'], hue=train['Survived'])
train['Sex'].value_counts(normalize=True)
# this will show the percentage of women in the dataset
(train[train.Survived > 0])['Sex'].value_counts(normalize=True)
# this will show the percentage of women in survivors
train['Survived'].groupby(train['Sex']).mean()
# Percentage in survivors per gender
sns.countplot(train['Sex'], hue=train['Survived'])
sns.countplot(train['ParenthesisInName'], hue=train['Sex'])
print('Proportion of parenthesis in male\'s name')
print((train[train.Sex == 'male'])['ParenthesisInName'].value_counts(normalize=True))
print('Proportion of parenthesis in female\'s name')
print((train[train.Sex == 'female'])['ParenthesisInName'].value_counts(normalize=True))
(train[train.Sex == 'female'])['Survived'].groupby(train['ParenthesisInName']).mean()
sns.countplot((train[train.Sex == 'female'])['ParenthesisInName'], hue=train['Survived'])
sns.countplot(train['QuoteInName'], hue=train['Sex'])
train['Survived'].groupby(train['Age'].isnull()).count()
train['Survived'].groupby(train['Age'].isnull()).mean()
sns.countplot(train['Age'].isnull(), hue=train['Survived'])
sns.countplot(pd.qcut(train['Age'],6), hue=train['Survived'])
train['MinorThan18'] = train.Age < 18
train['Survived'].groupby(train['MinorThan18']).mean()
sns.countplot(train['MinorThan18'], hue=train['Survived'])
train['MinorThan15'] = train.Age < 15
train['Survived'].groupby(train['MinorThan15']).mean()
sns.countplot(train['MinorThan15'], hue=train['Survived'])
train['MinorThan10'] = train.Age < 10
train['Survived'].groupby(train['MinorThan10']).mean()
sns.countplot(train['MinorThan10'], hue=train['Survived'])
pd.qcut(train['Age'],6).value_counts()
train['Survived'].groupby(train['SibSp']).mean()
train['SibSp'].value_counts()
sns.countplot(train['SibSp'], hue=train['Survived'])
train['Survived'].groupby(train['Parch']).mean()
train['Parch'].value_counts()
sns.countplot(train['Parch'], hue=train['Survived'])
train['FamSize'] = train['SibSp'] + train['Parch']
train['Survived'].groupby(train['FamSize']).mean()
train['FamSize'].value_counts()
sns.countplot(train['FamSize'], hue=train['Survived'])
train['ChildOverSibling'] = train['Parch'] - train['SibSp']
train['Survived'].groupby(train['ChildOverSibling']).mean()
train['ChildOverSibling'].value_counts()
sns.countplot(train['ChildOverSibling'], hue=train['Survived'])
train['Ticket_Len'] = train['Ticket'].apply(lambda x: len(x))
train['Ticket_Len'].value_counts()
train['Survived'].groupby(train['Ticket_Len']).mean()
train['LetraTicket'] = train['Ticket'].apply(lambda x: str(x)[0])
train['LetraTicket'].value_counts()
train['Survived'].groupby(train['LetraTicket']).mean()
pd.qcut(train['Fare'], 5).value_counts()
train['Survived'].groupby(pd.qcut(train['Fare'], 3)).mean()
sns.countplot(pd.qcut(train['Fare'], 3), hue=train['Survived'])
pd.crosstab(pd.qcut(train['Fare'], 3), columns=train['Pclass'])
sns.boxplot(x=train['Pclass'], y=train['Fare'], palette="Set1")
train['Fare_per_person'] = train['Fare']/(train['FamSize']+1)
pd.qcut(train['Fare_per_person'], 3).value_counts()
train['Survived'].groupby(pd.qcut(train['Fare_per_person'], 3)).mean()
sns.countplot(pd.qcut(train['Fare_per_person'], 3), hue=train['Survived'])
train['Survived'].groupby(train['Cabin'].isnull()).mean()
train['Cabin_Letter'] = train['Cabin'].apply(lambda x: str(x)[0])
train['Cabin_Letter'].value_counts()
train['Survived'].groupby(train['Cabin_Letter']).mean()
sns.countplot(train['Cabin_Letter'], hue=train['Survived'])
train['Survived'].groupby(train['Embarked'].isnull()).mean()
train['Embarked'].value_counts()
train['Embarked'].value_counts(normalize=True)
train['Survived'].groupby(train['Embarked']).mean()
sns.countplot(train['Embarked'], hue=train['Survived'])
sns.countplot(train['Embarked'], hue=train['Pclass'])
def names(data):
    data['Title'] = data['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
    data['Title'] = data['Title'].replace(['Lady.','Mme.','Countess.','Capt.','Col.','Don.',
                                             'Major.', 'Sir.', 'Jonkheer.', 'Dona.', 'Ms.',
                                             'Mlle.', 'the'], 'otro')
    data['NameLen'] = data['Name'].apply(lambda x: len(x))
    data['ParenthesisInName'] = data['Name'].apply(lambda x: "(" in x)
    data['QuoteInName'] = data['Name'].apply(lambda x: "\"" in x)
    data = data.drop(['Name'], axis = 1)  # The name is useless
    return data

def replace_titles(data):
    title = data['Title']
    if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Countess', 'Mme']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms']:
        return 'Miss'
    elif title == 'Dr':
        if data['Sex']=='Male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return title

def namesVar(data):
    data['Title'] = data['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
    data['Title'] = data.apply(replace_titles, axis=1)
    data['NameLen'] = data['Name'].apply(lambda x: len(x))
    data['ParenthesisInName'] = data['Name'].apply(lambda x: "(" in x)
    data['QuoteInName'] = data['Name'].apply(lambda x: "\"" in x)
    data = data.drop(['Name'], axis = 1)  # The name is useless
    return data
# The first function fills which the mean of each person's class and title and creates a flag
def nansAge(data):
    data['Null_Age'] = data['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
    temp = train.groupby(['Title', 'Pclass'])['Age']
    data['Age'] = temp.transform(lambda x: x.fillna(x.mean()))
    data['MinorThan18'] = data['Age'].apply(lambda x: 1 if x < 18 else 0)
    data['MinorThan15'] = data['Age'].apply(lambda x: 1 if x < 15 else 0)
    data['MinorThan10'] = data['Age'].apply(lambda x: 1 if x < 10 else 0)
    return data

# The second method just fills the nulls with -1
def nansAgeVar(data):
    data['MinorThan18'] = data['Age'].apply(lambda x: 1 if x < 18 else 0)
    data['MinorThan15'] = data['Age'].apply(lambda x: 1 if x < 15 else 0)
    data['MinorThan10'] = data['Age'].apply(lambda x: 1 if x < 10 else 0)
    data.Age[data.Age.isnull()] = -1
    return data
def distFamily(data):
    data['FamSize'] = data['Parch'] + data['SibSp']
    data['ChildOverSiblingSum'] = data['Parch'] - data['SibSp']
    return data
def tickets(data):
    data['TicketLetter'] = data['Ticket'].apply(lambda x: str(x)[0])
    data['TicketLetter'] = data['TicketLetter'].apply(lambda x: str(x))
    
    # Now, if the first letter was one those that were more incident than 'A' in the training data set
    # then we leave it. If it appeared in the training dataset but were less indicent than 'A', then we
    # change it as a 'low_ticket'. If it didn't appeared in the training dataset, we set it as 'other'.
    data['TicketLetter'] = np.where((data['TicketLetter']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']), data['TicketLetter'],
                               np.where((data['TicketLetter']).isin(['W', '4', '7', '6', 'L', '5', '8']),
                                        'Low_ticket', 'Other_ticket'))
    
    # Finally, we get the number of characters in the ticket number.
    data['TicketLen'] = data['Ticket'].apply(lambda x: len(x))
    data = data.drop(['Ticket'], axis = 1)
    return data
def cabin(data):
    data['CabinLetter'] = data['Cabin'].apply(lambda x: str(x)[0])
    data = data.drop(['Cabin'], axis = 1)
    return data
def fillEmbarked(data):
    data['Embarked'] = data['Embarked'].fillna('S')
    return data
def farePerPerson(data):
    data['FarePerPerson'] = data['Fare']/(data['FamSize']+1)
    return data
def dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'TicketLetter', 'CabinLetter', 'Title', 'ParenthesisInName', 'QuoteInName']):
    for column in columns:
        train[column] = train[column].apply(lambda x: str(x))
        test[column] = test[column].apply(lambda x: str(x))
        good_cols = ([column+'_'+i for i in train[column].unique() if i in test[column].unique()])[:-1]
        train = pd.concat((train, pd.get_dummies(train[column], prefix = column)[good_cols]), axis = 1)
        test = pd.concat((test, pd.get_dummies(test[column], prefix = column)[good_cols]), axis = 1)
        del train[column]
        del test[column]
    return train, test
def setData(data):
    data = data.drop(['PassengerId'], axis = 1)
    data = nansAge(data)
    data = distFamily(data)
    data = tickets(data)
    data = cabin(data)
    data = fillEmbarked(data)
    data = farePerPerson(data)
    return data
def fillFare(data):
    temp = train.groupby(['Pclass'])['Fare']
    data['Fare'] = temp.transform(lambda x: x.fillna(x.mean()))
    return data
# For the local notebook
# train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')

# For the online notebook
train = pd.read_csv(os.path.join('../input', 'train.csv'))
test = pd.read_csv(os.path.join('../input', 'test.csv'))
test = fillFare(test)
train = names(train)
test = names(test)

train = setData(train)
test = setData(test)

train, test = dummies(train, test)
len(train.columns)
train.columns
from sklearn.ensemble import RandomForestClassifier

precisiones = []
for i in range(2, 300, 2):
    rf = RandomForestClassifier(criterion='entropy', 
                                n_estimators=i,
                                min_samples_split=10,
                                min_samples_leaf=2,
                                max_features=15,
                                oob_score=True,
                                random_state=1,
                                n_jobs=-1)

    rf.fit(train.iloc[:, 1:], train.iloc[:, 0])
    precisiones.append(rf.oob_score_)
plt.plot(range(2, 300, 2), precisiones)
rf = RandomForestClassifier(criterion='entropy', 
                             n_estimators=150,
                             min_samples_split=10,
                             min_samples_leaf=2,
                             max_features=15,
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1)

rf.fit(train.iloc[:, 1:], train.iloc[:, 0])
print("%.4f" % rf.oob_score_)
pd.concat((pd.DataFrame(train.iloc[:, 1:].columns, columns = ['variable']), 
           pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:10]
# Se predice el asunto
predictions = rf.predict(test)
predictions = pd.DataFrame(predictions, columns=['Survived'])
predictions.head()
#test2 = pd.read_csv('test.csv')
test2 = pd.read_csv(os.path.join('../input', 'test.csv'))
predictions = pd.concat((test2.iloc[:, 0], predictions), axis = 1)
predictions.to_csv('y_predicted.csv', sep=",", index = False)
predictions.head()
