import pandas as pd

titanic = pd.read_csv('../input/titanic/train.csv', index_col=0)
titanic['Pclass'] = titanic.Pclass.astype('category')
print(f'We have the data of {len(titanic)} passengers. Only the first 5 rows are displayed here.')
titanic.head()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

X = titanic.drop(['Survived'], axis=1)
y = titanic.Survived

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2020)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
X = titanic.drop(['Survived', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
y = titanic.Survived

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2020)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn.score(X_test, y_test)
X = titanic.drop(['Survived', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1)
y = titanic.Survived

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2020)
knn = KNeighborsClassifier(7)
knn.fit(X_train, y_train)
accuracy = knn.score(X_test, y_test)
print(f'Accuracy = {round(accuracy*100,1)} %')
import numpy as np
%matplotlib notebook
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

fig, axes = plt.subplots(3, 3)
sns.countplot(x='Sex', hue='Survived', data=titanic, dodge=False, ax=axes[0, 0])
axes[0, 0].legend(loc='upper right')
sns.distplot(titanic.Age, label='Total', bins=np.linspace(0,80,17), kde=False, norm_hist=True, color='g', ax=axes[0, 1])
sns.distplot(titanic.Age[titanic.Survived == 0], bins=np.linspace(0,80,17), label='0', kde=False, norm_hist=True, ax=axes[0, 2])
sns.distplot(titanic.Age[titanic.Survived == 1], bins=np.linspace(0,80,17), label='1', kde=False, norm_hist=True, ax=axes[0, 2])
axes[0, 2].legend()

sns.countplot(x='Pclass', hue='Survived', data=titanic, dodge=False, ax=axes[1, 0])
axes[1, 0].legend(loc='upper left')
sns.distplot(titanic.Fare, label='Total', bins=np.linspace(0,200,9), kde=False, norm_hist=True, color='g', ax=axes[1, 1])
sns.distplot(titanic.Fare[titanic.Survived == 0], bins=np.linspace(0,200,9), label='0', kde=False, norm_hist=True, ax=axes[1, 2])
sns.distplot(titanic.Fare[titanic.Survived == 1], bins=np.linspace(0,200,9), label='1', kde=False, norm_hist=True, ax=axes[1, 2])
axes[1, 2].legend()

sns.countplot(x='SibSp', hue='Survived', data=titanic, dodge=False, ax=axes[2, 0])
axes[2, 0].legend(loc='upper right')
sns.countplot(x='Parch', hue='Survived', data=titanic, dodge=False, ax=axes[2, 1])
axes[2, 1].legend(loc='upper right')
sns.countplot(x='Embarked', hue='Survived', data=titanic, dodge=False, ax=axes[2, 2])
axes[2, 2].legend(loc='upper right')

plt.tight_layout()
fig.suptitle('The Sinking of the Titanic - Who Survived?')
fig.subplots_adjust(top=0.9)
plt.rcParams['figure.figsize'] = [9.5, 6]
plt.show()
X = titanic[['Sex', 'Pclass', 'Age','SibSp', 'Parch', 'Embarked']]
y = titanic.Survived
X.head()
X = pd.get_dummies(X, drop_first=True)
X.head()
import warnings
warnings.filterwarnings('ignore')

age_known = X.dropna(axis=0)
age_unknown = X[X.Age.isnull()].drop('Age', axis=1)

age_known['Age_group'] = np.nan

for row in range(len(age_known)):
    if age_known.Age.iloc[row] < 15: age_known.Age_group.iloc[row] = '<15'
    elif age_known.Age.iloc[row] >= 15 and age_known.Age.iloc[row] < 40: age_known.Age_group.iloc[row] = '15-39' 
    elif age_known.Age.iloc[row] >= 40 and age_known.Age.iloc[row] < 65: age_known.Age_group.iloc[row] = '40-64'
    else: age_known.Age_group.iloc[row] = '>=65'

X_age = age_known.drop(['Age', 'Age_group'], axis=1)
y_age = age_known.Age_group

knn = KNeighborsClassifier(8)
knn.fit(X_age, y_age)
age_unknown['Age_group'] = knn.predict(age_unknown)

age_unknown['Age'] = np.nan

for row in range(len(age_unknown)):
    if age_unknown.Age_group.iloc[row] == '<15': age_unknown.Age.iloc[row] = 10
    elif age_unknown.Age_group.iloc[row] == '15-39': age_unknown.Age.iloc[row] = 27    
    elif age_unknown.Age_group.iloc[row] == '40-64': age_unknown.Age.iloc[row] = 57
    else: age_unknown.Age.iloc[row] = 70
        
X_ready = age_known.append(age_unknown).drop('Age_group', axis=1).sort_index()
X_ready.head()
X_train, X_test, y_train, y_test = train_test_split(X_ready, y, test_size=0.1, random_state=2020)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

scaler = StandardScaler()
knn = KNeighborsClassifier(7)
pipeline = make_pipeline(scaler, knn)
pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
print(f'Accuracy = {round(accuracy*100,1)} %')
from sklearn.svm import SVC
svc = SVC()
pipeline2 = make_pipeline(scaler, svc)
pipeline2.fit(X_train, y_train)
accuracy = pipeline2.score(X_test, y_test)
print(f'Accuracy = {round(accuracy*100,1)} %')