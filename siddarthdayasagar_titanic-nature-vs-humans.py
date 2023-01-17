import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

train_df=pd.read_csv('../input/titanic/train.csv')
train_df.head()
test_df=pd.read_csv('../input/titanic/test.csv')
test_df.head()
train_df.info()

test_df.info()
train_df = train_df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1)
test_df    = test_df.drop(['Name','Ticket'], axis=1)
train_df.head()
test_df.head()
train_df.describe()
#VISUALIZATION COLUMN WISE
#FARE
plt.figure(figsize=(10,7))
sns.barplot(train_df['Survived'],train_df['Fare'])

plt.figure(figsize=(10,7))
sns.barplot(train_df['Survived'],train_df['Age'])

sns.regplot(train_df['Age'],train_df['Fare'])
plt.figure(figsize=(14,6))
sns.countplot(x='Survived',hue='Sex',data=train_df)
plt.title('people category by Sex and Survival')
plt.figure(figsize=(14,6))
sns.heatmap(train_df.corr(),annot=True)
plt.title("Correlation of the following data");
total = train_df.isnull().sum().sort_values(ascending=False)
percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)
#FILLING ALL MISSING VALUES
#FARE
data = [train_df, test_df]

for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)
    
#AGE
data = [train_df, test_df]

for dataset in data:
    dataset['Age'] = dataset['Age'].fillna(0)
    dataset['Age'] = dataset['Age'].astype(int)
train_df['Embarked'].value_counts()
train_df['Embarked'].fillna(value='S',inplace=True)
embark = {"S": 1, "Q": 2,"C":3}
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(embark)

genders = {"male": 0, "female": 1}
data = [train_df, test_df]

for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)
from sklearn.model_selection import train_test_split,KFold
from sklearn.linear_model import LogisticRegression
from sklearn import svm
train_df['female']=train_df['Sex']=='female'

X = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].values

y = train_df['Survived']

X_train,X_test,y_train,y_test=train_test_split(X,y)
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

model0 = RandomForestClassifier(n_estimators=100)
model0.fit(X_train, y_train)

Y_prediction = model0.predict(X_test)

NATURE=model0.score(X_train, y_train)
print(NATURE)
    

from sklearn.neural_network import MLPClassifier

model1=MLPClassifier(max_iter=1500,hidden_layer_sizes=(500,1000,500),alpha=1e-5,solver='adam',random_state=26)
model1.fit(X_train,y_train)

y_prediction=model1.predict(X_test)

HUMANS=model1.score(X_train,y_train)
print(HUMANS)
winner = pd.DataFrame({
    'Model': ['FOREST','NEURONS'],
    'Score': [NATURE,HUMANS]})
win_model = winner.sort_values(by='Score', ascending=False)
win_model = winner.set_index('Score')
win_model.head(9)
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X_test = pd.get_dummies(test_df[features])
predictions = model0.predict(X_test)

output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
sub=pd.read_csv('./submission.csv')
sub.head()
