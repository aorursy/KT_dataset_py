# some routine imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB



import warnings

warnings.filterwarnings('ignore')
# load data and Exploratory Data Analysis (EDA)

my_train_data = pd.read_csv("../input/titanic/train.csv", index_col=0)

my_test_data = pd.read_csv("../input/titanic/test.csv", index_col=0)

my_train_data.head(2)
my_train_data.info()
my_test_data['Survived'] = 0

my_all_data = pd.concat([my_train_data,my_test_data], ignore_index=True)

my_all_data.describe()
print(my_train_data.Age.count(), my_test_data.Age.count(), my_all_data.Age.count())
missing = my_all_data.isnull().sum()

missing_pct = missing/len(my_all_data)*100

print(pd.DataFrame(data=[missing, missing_pct]))

      

missing_table = pd.DataFrame(data=[missing, missing_pct],

                            index=['missing','% missing'])

missing_table.T # convinient way to rotate the data
# deal with missing data

# drop Cabin; mode for Embarked and Fare; investigate Age

my_all_data = my_all_data.drop('Cabin', axis = 1)
my_all_data.Embarked = my_all_data.Embarked.fillna('S')

my_all_data.Embarked.value_counts(dropna=False)
# exploring Fare distribution

fig, (axis1, axis2) = plt.subplots(2,1,figsize=(8,10))

axis1.set_title('Normal')

sns.distplot(my_all_data.Fare.dropna(), kde=False, ax=axis1)

# print(np.log2(0))

temp = np.log2(my_all_data.Fare.dropna().loc[my_all_data.Fare != 0])

temp.sort_values()

axis2.set_title('Log2')

sns.distplot(temp, kde=False, ax=axis2)
nona = my_all_data

sex_embark_pclass = nona.Sex.map(str)+"_"+nona.Embarked+"_"+nona.Pclass.map(str)

print(sex_embark_pclass.nunique(), sex_embark_pclass.count(), len(my_all_data))

sex_embark_pclass.value_counts()
fig, (axis1, axis2) = plt.subplots(2,1,figsize=(8,10))

axis1.set_title('Age before impute')

sns.distplot(my_all_data.Age.dropna(), kde=False, ax=axis1)

axis2.set_title('Age after impute')

old_ages = my_all_data.Age.copy()

my_all_data['AgeGroup'] = sex_embark_pclass

m_copy = my_all_data.copy()

print(m_copy.Age.isnull().sum())



m_copy.Age = m_copy.Age.fillna(m_copy.groupby('AgeGroup').Age.transform('mean'))

print(m_copy.Age.isnull().sum())

sns.distplot(m_copy.Age, kde=False, ax=axis2)

my_all_data.Age = m_copy.Age

print(my_all_data.Age.isnull().sum())
plt.figure(figsize=(8,5))

sns.distplot(old_ages.dropna(), bins = 32, kde=False, color='red')



sns.distplot(m_copy.Age, kde=False)
my_all_data['FamilySize'] = my_all_data.Parch + my_all_data.SibSp + 1

fsize_sur = my_all_data.loc[:890,['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()

sns.barplot(x='FamilySize',y='Survived', data = fsize_sur)
def t_c(df):

    if df['FamilySize'] == 1:

        return 0

    elif df['FamilySize'] < 5:

        return 1

    else:

        return 2

my_all_data['FSGroup'] = my_all_data.apply(t_c, axis=1)

my_all_data.head()
from sklearn.ensemble import RandomForestClassifier



my_train_data = my_all_data.iloc[:891]

my_test_data = my_all_data.iloc[891:]

y = my_train_data.Survived



features = ['Age','Embarked','Pclass', 'Sex', 'FSGroup']

X = pd.get_dummies(my_train_data[features])

X_test = pd.get_dummies(my_test_data[features])



model = RandomForestClassifier(n_estimators=100,max_depth=5,

                              random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId':my_test_data.index+1, 

                      'Survived': predictions})

output.head(10)

output.tail(5)
# output.to_csv('RF_submission.csv', index=False)
from sklearn.model_selection import train_test_split

from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
model_names = ['Logistric Regression', 'K Nearest Neighbors (K=5)', 'Random Forest']

models = [LogisticRegression(), KNeighborsClassifier(n_neighbors=5), RandomForestClassifier(n_estimators=100,max_depth=5,random_state=1)]



for name,model in zip(model_names, models):

    m = model

    m.fit(X_train, y_train)

    y_pred = m.predict(X_test)

    score = metrics.accuracy_score(y_test, y_pred)

    print(name, 'accuracy score is: ', score)
import re

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""



# dataset['Title'] = dataset['Name'].apply(get_title)



my_all_data['Title'] = my_all_data.Name.apply(get_title)

my_all_data.head()
my_all_data.Title.value_counts()
plt.figure(figsize=(10,8))

sns.countplot(y=my_all_data.Title)
my_all_data['Title'] = my_all_data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



my_all_data['Title'] = my_all_data['Title'].replace('Mlle', 'Miss')

my_all_data['Title'] = my_all_data['Title'].replace('Ms', 'Miss')

my_all_data['Title'] = my_all_data['Title'].replace('Mme', 'Mrs')

sns.countplot(y=my_all_data.Title)
my_train_data = my_all_data.iloc[:891]

my_test_data = my_all_data.iloc[891:]

y = my_train_data.Survived



features = ['Age','Embarked','Pclass', 'Sex', 'FSGroup']

X = pd.get_dummies(my_train_data[features])

X_test = pd.get_dummies(my_test_data[features])



model = RandomForestClassifier(n_estimators=100,max_depth=5,

                              random_state=1)

model.fit(X, y)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId':my_test_data.index+1, 

                      'Survived': predictions})

output.head(10)

output.tail(5)
output.to_csv('RF_submission2.csv', index=False)