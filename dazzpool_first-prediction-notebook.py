import pandas as pd
import seaborn as sns
sns.set()
train = pd.read_csv('../input/titanic/train.csv')
type(train)
train.info()
train.shape
train.head()
train.describe()
train['Sex'].value_counts()
sns.countplot(train['Sex'])
train['Survived'].value_counts(dropna=False)
sns.countplot(train['Survived'])
sns.countplot(train['Survived'], hue = 'Sex', data = train)
train['Survived'].value_counts(dropna=False)
sns.countplot(train['Embarked'])
sns.countplot(train['Embarked'], hue = 'Survived', data = train)
train.isna().sum()
sns.heatmap(train.isna())
# Remove the whole row:
train.dropna()
train['Age'].mean()
# Using median to replace our missing values:
train['Age'].fillna(train['Age'].median(), inplace = True)
# There's a slight difference in the age.
train['Age'].mean()
# Let's check if we're still having NA values or not
sns.heatmap(train.isna())
# Let's look how our features are corelated with each other
train.corr()
sns.heatmap(train.corr(), annot = True)
X = train[['Pclass', 'Fare', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]
y = train[['Survived']]
sex = X['Sex']
sex_dummy = pd.get_dummies(sex)
sex_dummy_final = sex_dummy.iloc[:,0:1]
X['Sex'] = sex_dummy_final
embarked = X['Embarked']
embarked_dummies = pd.get_dummies(embarked)
embarked_dummies_final = embarked_dummies.iloc[:,0:2]
X[['Cherbourg', 'Queenstown']] = embarked_dummies_final
X.drop('Embarked', axis=1, inplace=True)
pclass = X['Pclass']
pclass_dummies = pd.get_dummies(pclass)
pclass_dummies_final = pclass_dummies.iloc[:,0:2]
X[['pclass1', 'pclass2']] = embarked_dummies_final
X.drop('Pclass', axis=1, inplace=True)
X
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
#Fitting our model
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
y_pred
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)