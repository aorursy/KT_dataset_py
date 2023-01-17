# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

# machine learning
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

# other
import warnings
warnings.filterwarnings('ignore')
train_df = pd.read_csv("../input/train.csv")
train_df.head()
train_df.info()
train_df.describe()
test_df = pd.read_csv("../input/test.csv")
test_df.info()
test_df.describe()
#Pclass
sns.factorplot('Pclass', 'Survived', data = train_df, size=3, aspect=3)
# Sex
def get_gender(passenger):
  age, sex = passenger
  return 'child' if age < 16 else sex

train_df['Gender'] = train_df[['Age','Sex']].apply(get_gender, axis=1)
test_df['Gender'] = test_df[['Age','Sex']].apply(get_gender, axis=1)
sns.barplot(x='Gender', y='Survived', hue='Pclass', data=train_df)
# Age
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
# Fare
sns.boxplot(x='Pclass', y='Fare', hue='Survived', data=train_df)
# SibSp&Parch
train_df['Family'] =  train_df["Parch"] + train_df["SibSp"]
train_df['Family'].loc[train_df['Family'] > 0] = 1
train_df['Family'].loc[train_df['Family'] == 0] = 0

test_df['Family'] =  test_df["Parch"] + test_df["SibSp"]
test_df['Family'].loc[test_df['Family'] > 0] = 1
test_df['Family'].loc[test_df['Family'] == 0] = 0

g=sns.barplot(x='Family', y='Survived', data=train_df)
g.set_xticklabels(["Alone", "With Family"])
# Embarked
sns.factorplot('Embarked', 'Survived', data = train_df, size=3, aspect=3)
data = [train_df, test_df]
for dataset in data:
    mean = dataset["Age"].mean()
    std = dataset["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = dataset["Age"].astype(int)
    
train_df["Age"].isnull().sum()
test_df["Age"].isnull().sum()
train_df['Embarked'].describe()
data = [train_df, test_df]
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
train_new = train_df.drop(['PassengerId', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1)
test_new = test_df.drop(['Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1)
data = [train_new, test_new]
for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)
gender = {"male": 0, "female": 1, "child": 2}
data = [train_new, test_new]

for dataset in data:
    dataset['Gender'] = dataset['Gender'].map(gender)
port = {"S": 0, "C": 1, "Q": 2}
data = [train_new, test_new]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(port)
train_new.info()
test_new.info()
# Define the training and test sets
X_train = train_new.drop("Survived",axis=1)
Y_train = train_new["Survived"]
X_test  = test_new.drop("PassengerId",axis=1).copy()
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)

round(logreg.score(X_train, Y_train), 4)
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)

round(random_forest.score(X_train, Y_train), 4)
knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(X_train, Y_train)  
Y_pred = knn.predict(X_test)  

round(knn.score(X_train, Y_train), 4)
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)

round(linear_svc.score(X_train, Y_train), 4)
from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
importances = pd.DataFrame({'feature':X_train.columns,
                            'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)
outcome = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': Y_pred
})
outcome.to_csv('titanic.csv', index=False)