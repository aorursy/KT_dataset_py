import pandas as pd 
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style('dark')
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
test=pd.read_csv('/kaggle/input/titanic/test.csv')
test.head()
train.info()
test.info()
train["Survived"].value_counts(normalize=True)
g=sns.countplot(x=train["Survived"]).set_title("Survivor Count")
a=sns.countplot(train["Sex"]).set_title("Passenger count by Sex")
plt.title("Survival Rate by Sex")
b=sns.barplot(x='Sex',y='Survived',data=train).set_ylabel("Survival Rate")
train.groupby('Pclass').Survived.mean()
a = sns.countplot(x='Pclass', hue='Survived', data=train).set_title('Survivors by class')
plt.title("Survival Rate by Pclass")
b = sns.barplot(x='Pclass', y='Survived', data=train).set_ylabel('Survival Rate')
train.groupby(['Pclass','Sex']).Survived.mean()
plt.title("Survival Rate by Pclass and Sex")
g=sns.barplot(x='Pclass',y='Survived',hue='Sex',data=train).set_ylabel("Survival Rate")
fig, axarr = plt.subplots(1,2,figsize=(12,6))
axarr[0].set_title('Age distribution')
f = sns.distplot(train['Age'], color='red', bins=40, ax=axarr[0])
axarr[1].set_title('Age distribution for the two subpopulations')
g = sns.kdeplot(train['Age'].loc[train['Survived'] == 1], 
                shade= True, ax=axarr[1], label='Survived').set_xlabel('Age')
g = sns.kdeplot(train['Age'].loc[train['Survived'] == 0], 
                shade=True, ax=axarr[1], label='Not Survived')
g = sns.swarmplot(y='Sex', x='Age', hue='Survived', data=train).set_title('Survived by age and sex')
train.Fare.describe()
f = sns.distplot(train.Fare, color='b').set_title('Fare distribution')
fare_ranges = pd.qcut(train.Fare, 4, labels = ['Low', 'Mid', 'High', 'Very high'])
g = sns.barplot(x=fare_ranges, y=train.Survived).set_ylabel('Survival rate')
a = sns.swarmplot(x='Sex', y='Fare', hue='Survived',data=train).set_title('Survived by fare and sex')
sns.countplot(train['Embarked']).set_title('Passengers count by boarding point')
p = sns.countplot(x = 'Embarked', hue = 'Survived', data = train).set_title('Survivors by boarding point')
g = sns.countplot(data=train, x='Embarked', hue='Pclass').set_title('Pclass count by boarding point')
train['Title'] = train['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
test['Title'] = test['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
train['Title'].value_counts()
test['Title'].value_counts()
train['Title'].replace(['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dona'], 'Miss', inplace=True)
test['Title'].replace(['Mme', 'Ms', 'Lady', 'Mlle', 'the Countess', 'Dona'], 'Miss', inplace=True)

train['Title'].replace(['Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer'], 'Mr', inplace=True)
test['Title'].replace(['Major', 'Col', 'Capt', 'Don', 'Sir', 'Jonkheer'], 'Mr', inplace=True)
train.groupby('Title').Survived.mean()
plt.title('Survival rate by Title')
g = sns.barplot(x='Title', y='Survived', data=train).set_ylabel('Survival rate')
fig, axarr = plt.subplots(1,2,figsize=(12,6))
a = sns.countplot(train['SibSp'], ax=axarr[0]).set_title('Passengers count by SibSp')
axarr[1].set_title('Survival rate by SibSp')
b = sns.barplot(x='SibSp', y='Survived', data=train, ax=axarr[1]).set_ylabel('Survival rate')
fig, axarr = plt.subplots(1,2,figsize=(12,6))
a = sns.countplot(train['Parch'], ax=axarr[0]).set_title('Passengers count by Parch')
axarr[1].set_title('Survival rate by Parch')
b = sns.barplot(x='Parch', y='Survived', data=train, ax=axarr[1]).set_ylabel('Survival rate')
g  = sns.catplot(x="Parch",y="Survived",data=train, height = 8)
g = g.set_ylabels("Survival Percentage")
g  = sns.catplot(x="SibSp",y="Survived",data=train, height = 8)
g = g.set_ylabels("Survival Percentage")
train['Fam_size'] = train['SibSp'] + train['Parch'] + 1
test['Fam_size'] = test['SibSp'] + test['Parch'] + 1
plt.title('Survival rate by family size')
g = sns.barplot(x=train.Fam_size, y=train.Survived).set_ylabel('Survival rate')
g  = sns.catplot(x="Fam_size",y="Survived",data=train, height = 8)
g = g.set_ylabels("Survival Percentage")
train['Fam_type'] = pd.cut(train.Fam_size, [0,1,4,7,11], labels=['Solo', 'Small', 'Big', 'Very big'])
test['Fam_type'] = pd.cut(test.Fam_size, [0,1,4,7,11], labels=['Solo', 'Small', 'Big', 'Very big'])
plt.title('Survival rate by family type')
g = sns.barplot(x=train.Fam_type, y=train.Survived).set_ylabel('Survival rate')
y = train['Survived']
features = ['Pclass', 'Sex', 'Fare', 'Title', 'Embarked', 'Fam_type']
X = train[features]
X.head()
numerical_cols = ['Fare']
categorical_cols = ['Pclass', 'Sex', 'Title', 'Embarked', 'Fam_type']

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='median')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder())
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Bundle preprocessing and modeling code 
titanic_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', RandomForestClassifier(random_state=0, 
                                                               n_estimators=500, max_depth=5))
                             ])

# Preprocessing of training data, fit model 
titanic_pipeline.fit(X,y)

print('Cross validation score: {:.3f}'.format(cross_val_score(titanic_pipeline, X, y, cv=10).mean()))
X_test = test[features]
X_test.head()
predictions = titanic_pipeline.predict(X_test)
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print('Your submission was successfully saved!')