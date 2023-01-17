import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('../input/titanic/train.csv')
df.head()
# View Summary Table
df.describe()
df.dtypes
# Check Missing values
df.isnull().sum()
fig = plt.gcf()
fig.set_size_inches(13, 9)
sns.countplot(x="Survived", hue="Sex", data=df, palette="rocket")
fig = plt.gcf()
fig.set_size_inches(13, 9)
sns.countplot(x="Survived", hue="Pclass", data=df, palette="rocket")
fig = plt.gcf()
fig.set_size_inches(30, 13)
sns.countplot(x="Age", hue="Survived", data=df, palette="deep")
fig = plt.gcf()
fig.set_size_inches(9, 13)
sns.countplot(x="Embarked", hue="Survived", data=df, palette="deep")
df.head() # Just to remind us with the data
# Fill the 2 Missing values in Embarked column with 'S' since it's the most common value
values = {'Embarked': 'S', 'Cabin': 'U0'}
df.fillna(value=values, inplace=True)
df['Deck'] = df['Cabin'].str.extract(pat = '([A-Z])') # Extract Passenger Deck from the Cabin column into a new feature
df.head()
# Now Let's convert the categorical values in Deck column into numeric
from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
enc.fit(df[['Deck']])
df[['Deck']] = enc.fit_transform(df[['Deck']])
df.head()
# Again Let's convert the categorical values in Embarked column into numeric
enc = OrdinalEncoder()
enc.fit(df[['Embarked']])
df[['Embarked']] = enc.fit_transform(df[['Embarked']])
df.head()
# Apply Ordinal Encoding to Sex column
encoding = {'Sex':{'male': 1, 'female':0}}
df.replace(encoding, inplace=True)
values = {'Age': df['Age'].mean()}
df.fillna(value=values, inplace=True)
df['Fare'] = df['Fare'].astype(int)
# Extract Title from Name Column
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['Title'] = df['Title'].replace('Mme', 'Mrs')
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace('Ms', 'Miss')
enc = OrdinalEncoder()
enc.fit(df[['Title']])
df[['Title']] = enc.fit_transform(df[['Title']])
df.head()
bins = [0, 10, 18, 30, 40, 50, 60, 70, 120]
labels = ['0-10', '10-18', '18-29', '30-39', '40-49', '50-59', '60-69', '70+']
df['Agebin'] = pd.cut(df.Age, bins, labels = labels,include_lowest = True)
df.head()
encoding = {"Agebin": {'0-10': 0, '10-18': 1, '18-29': 2, '30-39': 3, '40-49': 4, '50-59': 5, '60-69': 6, '70+': 7}}

# Apply Ordinal Encoding on preferred_foot column
df.replace(encoding, inplace=True)
# Creating new feature for ranking Age and Passenger Class
df['Age_Class'] = df['Age'] * df['Pclass']
df['Fair_Deck'] = df['Fare'] * df['Deck']
df['Pclass_Deck'] = df['Pclass'] * df['Deck']
columns = ['Survived', 'Pclass', 'Sex','SibSp', 'Parch', 'Fare', 'Embarked', 'Deck']
x_test = df[df['Age'].isnull()]
x_test = x_test[columns]

x_train = df[df['Age'].notnull()]
x = x_train
y_train = x_train['Age'].values
x_train = x_train[columns]
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)
pred[pred < 0] = 1 # Replace negative number with 1
pred
x_test['Age'] = pred.astype(int)
titanic_df = pd.concat([x, x_test])
titanic_df
titanic_df.isnull().sum()
df.head()
y_train = df['Survived'].values
X_train = df.drop(['Survived', 'Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

model_GB = GradientBoostingClassifier()

parameters = {
        'learning_rate': [0.001, 0.002, 0.01, 0.02, 0.1, 0.2],
        'n_estimators':[100],
        'min_samples_split': [5,10, 20],
        'min_samples_leaf': [5, 10, 20],
        'max_depth': [5, 10, 15, 20]
}

# Using GridSearch for tuning the HyperParameters
grid_GB = GridSearchCV(model_GB, parameters, cv=20)
grid_GB.fit(X_train, y_train)
print('Best Score: ', grid_GB.best_score_, '\nBest Parameters: ', grid_GB.best_params_)
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

RF = RandomForestClassifier(n_estimators=200)
RF.fit(X_train, y_train)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

DT = DecisionTreeClassifier()
parameters = {
        'max_depth': [5, 10, 15, 20],
        'min_samples_split': [5,10, 15, 20],
        'min_samples_leaf': [5, 10, 15, 20]
}

grid_DT = GridSearchCV(DT, parameters, cv=20)
grid_DT.fit(X_train, y_train)
print('Best Score: ', grid_DT.best_score_, '\nBest Parameters: ', grid_DT.best_params_)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier()
parameters = {
        'n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10]
}

grid_knn = GridSearchCV(knn, parameters, cv=20)
grid_knn.fit(X_train, y_train)
print('Best Score: ', grid_knn.best_score_, '\nBest Parameters: ', grid_knn.best_params_)
test_df = pd.read_csv('../input/titanic/test.csv')
test_df
values = {'Embarked': 'S', 'Cabin': 'U0'}
test_df.fillna(value=values, inplace=True)
test_df['Deck'] = test_df['Cabin'].str.extract(pat = '([A-Z])') # Extract Passenger Deck from the Cabin column into a new feature
test_df.head()
enc = OrdinalEncoder()
enc.fit(test_df[['Deck']])
test_df[['Deck']] = enc.fit_transform(test_df[['Deck']])
test_df.head()
enc = OrdinalEncoder()
enc.fit(test_df[['Embarked']])
test_df[['Embarked']] = enc.fit_transform(test_df[['Embarked']])
test_df.head()
encoding = {'Sex':{'male': 1, 'female':0}}
test_df.replace(encoding, inplace=True)
values = {'Age': test_df['Age'].mean()}
test_df.fillna(value=values, inplace=True)
test_df['Fare'] = test_df['Fare'].fillna(0)
test_df['Fare'] = test_df['Fare'].astype(int)
test_df['Title'] = test_df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test_df['Title'] = test_df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test_df['Title'] = test_df['Title'].replace('Mme', 'Mrs')
test_df['Title'] = test_df['Title'].replace('Mlle', 'Miss')
test_df['Title'] = test_df['Title'].replace('Ms', 'Miss')

enc = OrdinalEncoder()
enc.fit(test_df[['Title']])
test_df[['Title']] = enc.fit_transform(test_df[['Title']])
test_df.head()
bins = [0, 10, 18, 30, 40, 50, 60, 70, 120]
labels = ['0-10', '10-18', '18-29', '30-39', '40-49', '50-59', '60-69', '70+']
test_df['Agebin'] = pd.cut(test_df.Age, bins, labels = labels,include_lowest = True)

encoding = {"Agebin": {'0-10': 0, '10-18': 1, '18-29': 2, '30-39': 3, '40-49': 4, '50-59': 5, '60-69': 6, '70+': 7}}

# Apply Ordinal Encoding on preferred_foot column
test_df.replace(encoding, inplace=True)
# Creating new feature for ranking Age and Passenger Class
test_df['Age_Class'] = test_df['Age'] * test_df['Pclass']
test_df['Fair_Deck'] = test_df['Fare'] * test_df['Deck']
test_df['Pclass_Deck'] = test_df['Pclass'] * test_df['Deck']
dff = test_df.copy()
X_test = test_df.drop(['Name', 'Ticket', 'PassengerId', 'Cabin'], axis=1)
OHE = pd.get_dummies(test_df.Embarked)
test_df = pd.concat([test_df,OHE], axis=1)
del test_df['Embarked']

# Apply Ordinal Encoding to Sex column
encoding = {'Sex':{'male': 1, 'female':0}}
test_df.replace(encoding, inplace=True)

dff = test_df.copy()
test_df
columns = ['Pclass', 'Sex','SibSp', 'Parch', 'Fare', 'Cabin', 'S', 'C', 'Q']
x_test = test_df[test_df['Age'].isnull()]
x_test = x_test[columns]

x_train = test_df[test_df['Age'].notnull()]
x = x_train
y_train = x_train['Age'].values
x_train = x_train[columns]
model = LinearRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)
pred[pred < 0] = 1 # Replace negative number with 1
pred
x_test['Age'] = pred.astype(int)
final_df = pd.concat([x, x_test])
final_df
# Using Gradient Boosting Classifier
pred = grid_GB.predict(X_test)
pred
# Using Random Forest Classifier
pred = RF.predict(X_test)
pred
# Using Decision Tree Classifier
pred = grid_DT.predict(X_test)
pred
# Using Random Forest Classifier
pred = grid_knn.predict(X_test)
pred
dff['Survived'] = pred.astype(int)
dff.head()
final_df = dff[['PassengerId', 'Survived']]
final_df.head()
final_df.to_csv('sixth_submission.csv',index=False)
