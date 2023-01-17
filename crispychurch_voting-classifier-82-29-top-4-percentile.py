# Importing required libaries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/gender_submission.csv')

train_data.head()
train_data.isnull().sum()
test_data.isnull().sum()
train_data['Embarked'] = train_data['Embarked'].fillna('S')
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())
train_data['Embarked'].isnull().sum() + test_data['Fare'].isnull().sum()
sns.barplot(x=train_data['Pclass'], y=train_data['Survived'])
sns.barplot(x=train_data['Sex'], y=train_data['Survived'])
sns.pointplot(x=train_data['Age'], y=train_data['Survived'])
sns.barplot(x=train_data['SibSp'], y=train_data['Survived'])
sns.barplot(x=train_data['Parch'], y=train_data['Survived'])
train_data = train_data.join(train_data['Name'].str.split(',', 1, expand=True).rename(columns={0:'LastName', 1:'FName'}))
train_data = train_data.join(train_data['FName'].str.split('.', 1, expand=True).rename(columns={0:'Title', 1:'FirstName'}))
train_data['Title'] = train_data['Title'].str.strip()
train_data.drop(['Name', 'FName'], axis=1, inplace=True)
train_data.head()
test_data = test_data.join(test_data['Name'].str.split(',', 1, expand=True).rename(columns={0:'LastName', 1:'FName'}))
test_data = test_data.join(test_data['FName'].str.split('.', 1, expand=True).rename(columns={0:'Title', 1:'FirstName'}))
test_data['Title'] = test_data['Title'].str.strip()
test_data.drop(['Name', 'FName'], axis=1, inplace=True)
test_data.head()
# Initialising with -1s
family_ratio_train = [-1]*len(train_data)
family_ratio_test = [-1]*len(test_data)

friends_ratio_train = [-1]*len(train_data)
friends_ratio_test = [-1]*len(test_data)

for i in range(len(train_data)):
    #print('i = '+str(i))
    family_survive_list = []
    friends_survive_list = []
    for j in range(len(train_data)):
        if ((train_data['LastName'][i] == train_data['LastName'][j]) & (train_data['Fare'][i] == train_data['Fare'][j])):
            family_survive_list.append(train_data['Survived'][j])
        elif (train_data['Ticket'][i] == train_data['Ticket'][j]):
            friends_survive_list.append(train_data['Survived'][j])
    if len(family_survive_list) > 1:
        family_ratio_train[i] = np.mean(family_survive_list)
    else:
        family_ratio_train[i] = -1
    if len(friends_survive_list) > 1:
        friends_ratio_train[i] = np.mean(friends_survive_list)
    else:
        friends_ratio_train[i] = -1

i = 0
j = 0

for i in range(len(test_data)):
    #print('i = '+str(i))
    family_survive_list = []
    friends_survive_list = []
    for j in range(len(train_data)):
        if ((test_data['LastName'][i] == train_data['LastName'][j]) & (test_data['Fare'][i] == train_data['Fare'][j])):
            family_survive_list.append(train_data['Survived'][j])
        elif (test_data['Ticket'][i] == train_data['Ticket'][j]):
            friends_survive_list.append(train_data['Survived'][j])
    if len(family_survive_list) > 1:
        family_ratio_test[i] = np.mean(family_survive_list)
    else:
        family_ratio_test[i] = -1
    if len(friends_survive_list) > 1:
        friends_ratio_test[i] = np.mean(friends_survive_list)
    else:
        friends_ratio_test[i] = -1

train_data['Family Survival Ratio'] = family_ratio_train
test_data['Family Survival Ratio'] = family_ratio_test

train_data['Friends Survival Ratio'] = friends_ratio_train
test_data['Friends Survival Ratio'] = friends_ratio_test
# Creating the FamilySize variable as suggested before
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
# Filling the empty Cabin rows with U, for unknown.
train_data['Cabin'] = train_data['Cabin'].fillna('U')
test_data['Cabin'] = test_data['Cabin'].fillna('U')
# Choosing only the first character of the Cabin string, to denote Deck.
train_data['Cabin'] = train_data['Cabin'].str[0]
test_data['Cabin'] = test_data['Cabin'].str[0]
# Rearranging columns and creating a joint variable so that fewer lines of code are required
train_data = train_data[['Pclass','Title','Sex','Age','FamilySize','Family Survival Ratio',
                         'Friends Survival Ratio','Fare','Cabin','Embarked','Survived']]

test_data = test_data[['Pclass','Title','Sex','Age','FamilySize','Family Survival Ratio',
                       'Friends Survival Ratio','Fare','Cabin','Embarked']]

total_data = [train_data, test_data]
# Lets take a look at train_data first
train_data.head()
for dataset in total_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'the Countess', 'Sir', 'Don','Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Dr', 'Rev', 'Capt', 'Col', 'Major'], 'Occ')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Occ": 5, "Rare": 6}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    
    dataset['Sex'] = dataset['Sex'].map( {"female": 1, "male": 2} )
    
    cabin_mapping = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,"T":8,"U":9}
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
    
    dataset['Embarked'] = dataset['Embarked'].map( {'C': 1, 'Q': 2, 'S': 3} ).astype(int)
    
    dataset.loc[ dataset['Fare'] <= 10, 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 10) & (dataset['Fare'] <= 40), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 40) & (dataset['Fare'] <= 80), 'Fare'] = 3
    dataset.loc[ dataset['Fare'] > 80, 'Fare'] = 4
train_data.head()
# Just some steps for a smoother implementation
y = train_data['Survived'].values

train_data = pd.DataFrame(train_data, columns = ['Pclass','Title','Sex','Age','FamilySize',
                                                 'Family Survival Ratio','Friends Survival Ratio',
                                                 'Fare','Embarked'])    

test_data = pd.DataFrame(test_data, columns = ['Pclass','Title','Sex','Age','FamilySize',
                                                 'Family Survival Ratio','Friends Survival Ratio',
                                                 'Fare','Embarked'])    
X_total = pd.concat([train_data, test_data])
# Filling in missing values for Age
from fancyimpute import KNN
X_select = X_total[['Title', 'Age', 'FamilySize']]
X_complete = KNN(k=50).complete(X_select)

X_complete[:,1] = np.trunc(X_complete[:,1])

X_total['Age'] = X_complete[:,1]
# Categorising age
complete_data = pd.DataFrame(X_total,columns=X_total.columns)

X_total = complete_data.values

def categ_age(x):
    if x<=8:
        return 1
    elif x<=25:
        return 2
    elif x<=40:
        return 3
    elif x<=50:
        return 3
    else:
        return 4

for i in range(len(X_total)):
    X_total[i,3] = categ_age(X_total[i,3])
# Scaling feature values
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_total)

X = X_scaled[:len(train_data),:]
X_final = X_scaled[len(train_data):,:]
# Fitting the Voting Classifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
estimators_list = []

dt = DecisionTreeClassifier()
estimators_list.append(('Decision Tree', dt))

rf = RandomForestClassifier(n_estimators=3001)
estimators_list.append(('Random Forest', rf))

ada = AdaBoostClassifier(n_estimators=1000,learning_rate=0.00001)
estimators_list.append(('AdaBoost', ada))

grad_boost = GradientBoostingClassifier(max_features='sqrt',n_estimators=1000,learning_rate=0.00001)
estimators_list.append(('Gradient Boosting', grad_boost))

knn = KNeighborsClassifier(n_neighbors=19)
estimators_list.append(('KNN', knn))

logreg = LogisticRegression()
estimators_list.append(('Logistic Regression', logreg))

svc = SVC(kernel='rbf', C=1, gamma=0.1, probability=True)
estimators_list.append(('SVC', svc))

ensemble = VotingClassifier(estimators_list, voting='soft')
# Fitting the model on the train data
ensemble.fit(X,y)
#  Predicting for the test data
y_final_pred = ensemble.predict(X_final)
# Submission file
submission['Survived'] = y_final_pred
submission.to_csv('voting_submission.csv', index=False)