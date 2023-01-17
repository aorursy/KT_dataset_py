import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

#import train and test data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
dataset = pd.concat([train, test], ignore_index = True)
#Retrieve Passenger ID from test set, used for submission
PassengerId = test['PassengerId']
dataset = dataset.fillna(np.nan)
dataset.isnull().sum()
#Check missing values in train set
train.info()
train.isnull().sum()
# check the first five information of the train set
train.head()
# Check the data types of every column
train.dtypes
# Generate descriptive statistics that summarize the central tendency, dispersion and shape of a dataset's distribution
train.describe()
sns.barplot(x="Sex", y="Survived", data=train, palette='Set3')
print("Percentage of females that could survive: %.2f" %(train['Survived'][train['Sex'] == 'female'].value_counts(normalize = True)[1]*100))
print("Percentage of females that could survive: %.2f" %(train['Survived'][train['Sex'] == 'male'].value_counts(normalize = True)[1]*100))
sns.barplot(x='Pclass', y='Survived', data=train, palette='Set3')
print("Percentage of Pclass = 1, survived probability: %.2f" %(train['Survived'][train['Pclass']==1].value_counts(normalize = True)[1]*100))
print("Percentage of Pclass = 2, survived probability: %.2f" %(train['Survived'][train['Pclass']==2].value_counts(normalize = True)[1]*100))
print("Percentage of Pclass = 3, survived probability: %.2f" %(train['Survived'][train['Pclass']==3].value_counts(normalize = True)[1]*100))

sns.barplot(x="SibSp", y="Survived", data=train, palette='Set3')
sns.barplot(x="Parch", y="Survived", data=train, palette='Set3')
age = sns.FacetGrid(train, hue="Survived",aspect=2)
age.map(sns.kdeplot,'Age',shade= True)
age.set(xlim=(0, train['Age'].max()))
age.add_legend()
fare = sns.FacetGrid(train, hue="Survived",aspect=2)
fare.map(sns.kdeplot,'Fare',shade= True)
fare.set(xlim=(0, 200))
fare.add_legend()
dataset['Title'] = dataset['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
Title_Dict = {}
Title_Dict.update(dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
Title_Dict.update(dict.fromkeys(['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
Title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
Title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
Title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
Title_Dict.update(dict.fromkeys(['Master','Jonkheer'], 'Master'))
dataset['Title'] = dataset['Title'].map(Title_Dict)
sns.barplot(x="Title", y="Survived", data=dataset, palette='Set3')
dataset['FamilySize']=dataset['SibSp']+dataset['Parch']+1
sns.barplot(x="FamilySize", y="Survived", data=dataset, palette='Set3')
# Based on the family size, classify them into three groups
def Family_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif (s > 7):
        return 0
dataset['FamilyLabel']=dataset['FamilySize'].apply(Family_label)
sns.barplot(x="FamilyLabel", y="Survived", data=dataset, palette='Set3')
dataset['Cabin'] = dataset['Cabin'].fillna('Unknown')
dataset['Deck']= dataset['Cabin'].str.get(0)
sns.barplot(x="Deck", y="Survived", data=dataset, palette='Set3')
Ticket_Count = dict(dataset['Ticket'].value_counts())
dataset['TicketGroup'] = dataset['Ticket'].apply(lambda x:Ticket_Count[x])
sns.barplot(x='TicketGroup', y='Survived', data=dataset, palette='Set3')
# Classify the TicketGroup into three kinds
def Ticket_Label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif (s > 8):
        return 0

dataset['TicketGroup'] = dataset['TicketGroup'].apply(Ticket_Label)
sns.barplot(x='TicketGroup', y='Survived', data=dataset, palette='Set3')
# Fill the missing age value, use feature Pclass, Sex and Title and random forest regressor model to predict 
age = dataset[['Age','Pclass','Sex','Title']]
age = pd.get_dummies(age)
# print(age)
known_age = age[age.Age.notnull()].as_matrix()
null_age = age[age.Age.isnull()].as_matrix()
x = known_age[:, 1:]
y = known_age[:, 0]
rf = RandomForestRegressor(n_jobs=-1)
rf.fit(x, y)
predictedAge = rf.predict(null_age[:, 1:])
dataset.loc[(dataset.Age.isnull()),'Age'] = predictedAge
dataset[dataset['Embarked'].isnull()]
C = dataset[(dataset['Embarked']=='C') & (dataset['Pclass'] == 1)]['Fare'].median()
print(C)
S = dataset[(dataset['Embarked']=='S') & (dataset['Pclass'] == 1)]['Fare'].median()
print(S)
Q = dataset[(dataset['Embarked']=='S') & (dataset['Pclass'] == 1)]['Fare'].median()
print(Q)
dataset['Embarked'] = dataset['Embarked'].fillna('C')
dataset[dataset['Fare'].isnull()]
fare=dataset[(dataset['Embarked'] == "S") & (dataset['Pclass'] == 3)].Fare.median()
dataset['Fare']=dataset['Fare'].fillna(fare)
dataset['Surname']=dataset['Name'].apply(lambda x:x.split(',')[0].strip())
Surname_Count = dict(dataset['Surname'].value_counts())
dataset['FamilyGroup'] = dataset['Surname'].apply(lambda x:Surname_Count[x])
Female_Child_Group=dataset.loc[(dataset['FamilyGroup']>=2) & ((dataset['Age']<=12) | (dataset['Sex']=='female'))]
Male_Adult_Group=dataset.loc[(dataset['FamilyGroup']>=2) & (dataset['Age']>12) & (dataset['Sex']=='male')]
Female_Child=pd.DataFrame(Female_Child_Group.groupby('Surname')['Survived'].mean().value_counts())
Female_Child.columns=['GroupCount']
Female_Child
Male_Adult=pd.DataFrame(Male_Adult_Group.groupby('Surname')['Survived'].mean().value_counts())
Male_Adult.columns=['GroupCount']
Male_Adult
Female_Child_Group=Female_Child_Group.groupby('Surname')['Survived'].mean()
Dead_List=set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)
print(Dead_List)
Male_Adult_List=Male_Adult_Group.groupby('Surname')['Survived'].mean()
Survived_List=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)
print(Survived_List)
train=dataset.loc[dataset['Survived'].notnull()]
test=dataset.loc[dataset['Survived'].isnull()]
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Sex'] = 'male'
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Age'] = 60
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Title'] = 'Mr'
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Sex'] = 'female'
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Age'] = 5
test.loc[(test['Surname'].apply(lambda x:x in Survived_List)),'Title'] = 'Miss'
dataset = pd.concat([train, test])
dataset=dataset[['Survived','Pclass','Sex','Age','Fare','Embarked','Title','FamilyLabel','Deck','TicketGroup']]
dataset=pd.get_dummies(dataset)
trainset=dataset[dataset['Survived'].notnull()]
testset=dataset[dataset['Survived'].isnull()].drop('Survived',axis=1)
X = trainset.as_matrix()[:,1:]
Y = trainset.as_matrix()[:,0]
pipe=Pipeline([('select',SelectKBest(k=20)), 
               ('classify', RandomForestClassifier(random_state = 10, max_features = 'sqrt'))])

param_test = {'classify__n_estimators':list(range(20,50,2)), 
              'classify__max_depth':list(range(3,60,3))}
gsearch = GridSearchCV(estimator = pipe, param_grid = param_test, scoring='accuracy', cv=10)
gsearch.fit(X,Y)
print(gsearch.best_params_, gsearch.best_score_)
select = SelectKBest(k = 20)
clf = RandomForestClassifier(random_state = 10, warm_start = True, 
                                  n_estimators = 26,
                                  max_depth = 6, 
                                  max_features = 'sqrt')
pipeline = make_pipeline(select, clf)
pipeline.fit(X, Y)
cv_score = model_selection.cross_val_score(pipeline, X, Y, cv= 10)
print("CV Score : Mean - %.7g | Std - %.7g " % (np.mean(cv_score), np.std(cv_score)))
predictions = pipeline.predict(testset)
submission = pd.DataFrame({"PassengerId": PassengerId, "Survived": predictions.astype(np.int32)})
submission.to_csv("submission.csv", index=False)
