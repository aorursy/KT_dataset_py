import pandas as pd
from pandas import Series,DataFrame
train_df = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


train_df.head()
train_df.describe(include='all')
import seaborn as sns
sns.countplot('Survived',data=train_df)

train_df[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
grid = sns.FacetGrid(train_df, row='Pclass', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex','Survived', alpha=.5, ci=None)
grid.add_legend()
import matplotlib.pyplot as plt
%matplotlib inline
grid = sns.FacetGrid(train_df, col='Survived', row='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
import matplotlib.pyplot as plt
%matplotlib inline
grid = sns.FacetGrid(train_df, row='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
df = pd.DataFrame(train_df,columns=['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])




df = df.drop(['Cabin','PassengerId','Ticket'], axis = 1)

df.head()
test_df.head()
test = pd.DataFrame(test_df,columns=['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked'])



test = test.drop(['Cabin','Ticket'], axis = 1)

test.head()
median = df['Age'].mean()
df['Age']= df['Age'].fillna(median)
sns.countplot('Embarked',data=train_df)
import matplotlib.pyplot as plt
plt.subplot(121)
sns.boxplot('Pclass', 'Fare', 'Survived', df, orient='v')

df.groupby(['Embarked']).mean()
df['Embarked'] = df['Embarked'].fillna('S')
df.Age.isnull().any()
df.Embarked.isnull().any()
df.info()
median_test = test['Age'].median()
test['Age']= test['Age'].fillna(median_test)
fare = test['Fare'].median()
test['Fare'] = test['Fare'].fillna(fare)
test.info()
total_data = [df,test]
for dataset in total_data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
for dataset in total_data:
    dataset['Title'] = dataset['Title'].replace(['Capt','Col','Countess','Don','Dona','Dr','Jonkheer','Lady','Major','Master','Rev','Sir'], 'Honor')

    dataset['Title'] = dataset['Title'].replace(['Mlle', 'Miss','Ms','Mme','Mrs','Mr'],'Common')
    
df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
grid = sns.FacetGrid(df, row='Title', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex','Survived', alpha=.5, ci=None)
grid.add_legend()

df.groupby(['Title']).mean()

test.head()
for dataset in total_data:
    dataset['Title'] = dataset['Title'].map( {'Honor': 1, 'Common': 0} ).astype(int)


df = df.drop('Name', axis = 1)
test = test.drop('Name',axis = 1)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df['Gender'] = le.fit_transform(df.Sex)

test['Gender'] = le.fit_transform(test.Sex)

df = df.drop('Sex', axis = 1)
test = test.drop('Sex',axis = 1)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df['Embarked'] = le.fit_transform(df.Embarked)

test['Embarked'] = le.fit_transform(test.Embarked)
df.head()
test.head()
total_data = [df,test]
X_train = df.drop("Survived", axis=1)
Y_train = df["Survived"]
X_test  = test.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log
# get Correlation Coefficient for each feature using Logistic Regression
coeff_df = DataFrame(df.columns.delete(0))
coeff_df.columns = ['Features']
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])

# preview
coeff_df
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
clf = DecisionTreeClassifier(random_state=0)
clf = clf.fit(X_train, Y_train)
print(dict(zip(X_train.columns, clf.feature_importances_)))
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred2 = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest
feature = ['Pclass', 'Age', 'SibSp', 'Parch','Fare','Embarked','Title','Gender']
prediction = clf.predict(test[feature])
prediction

submission_randomforest = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':prediction})
filename = 'Titanic Prediction Beginner.csv'

submission_randomforest.to_csv(filename, index = False)

print('Saved file:' + filename)