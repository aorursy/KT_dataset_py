import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline 
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head(3)
train_df.info()
train_df.describe()
drop_elements = ['Name','Parch','Ticket','Cabin']
train_df = train_df.drop(drop_elements,axis=1)
train_df.info()
train_df['Age'].fillna(train_df['Age'].mean(), inplace=True)
test_df['Age'].fillna(test_df['Age'].mean(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].mean(), inplace=True)
train_df.info()
# train_df.dropna(inplace=True)
# test_df.dropna(inplace=True)
sns.pairplot(train_df)
sns.barplot('Sex','Survived',data=train_df)
sns.barplot(x='Pclass', y='Survived', hue='Sex', data=train_df, palette='coolwarm')
sns.countplot('Pclass',hue='Sex',data=train_df)
sns.factorplot('Sex', 'Survived', hue='Pclass', size=4, aspect=2, data=train_df)
# plot
fig = plt.figure(figsize=(15,5))
sns.factorplot('Embarked','Survived', hue='Sex',data=train_df, size=4,aspect=3)
fig = plt.figure(figsize=(15,5))
sns.distplot(train_df['Age'])
sns.heatmap(train_df.corr(),annot=True)
from sklearn.model_selection import train_test_split
train_df.columns
# X = train_df[['Pclass','Age', 'SibSp', 'Fare']]

# y = train_df['Survived']

X_train = train_df[['Pclass','Age', 'SibSp', 'Fare']]

y_train = train_df["Survived"]

X_test  = test_df[['Pclass','Age', 'SibSp', 'Fare']]
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix
lm = LinearRegression()
lm.fit(X_train,y_train)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
X_test.info()
test_df.head(3)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)

pred = lm.predict(X_test) #Linear Regression
dt_pred = clf.predict(X_test)
rfc_pred = rfc.predict(X_test)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
#         "Survived": dt_pred #decision tree
#         "Survived": pred # linear Regression
        "Survived": rfc_pred # Random Forest
    })
submission.to_csv('submission_3.csv', index=False)
