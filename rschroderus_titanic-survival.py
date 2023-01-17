import pandas as pd
import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import os
print(os.listdir("../input"))
titanic_train_data = pd.read_csv('../input/train.csv', index_col='PassengerId')
titanic_train_data.info()
titanic_train_data['Survived'].value_counts()
titanic_train_data.head(50)
titanic_train_data['Ticket'].value_counts(normalize=True)
pd.crosstab(titanic_train_data['Survived'], titanic_train_data['Pclass'])
plt.figure(figsize=(15,5))
plt.title("Pclass vs. Survival")
sns.countplot(titanic_train_data['Pclass'], hue=titanic_train_data['Survived'])
pd.crosstab(titanic_train_data['Survived'], titanic_train_data['Sex'])
plt.figure(figsize=(15,5))
plt.title("Sex vs. Survival")
sns.countplot(titanic_train_data['Sex'], hue=titanic_train_data['Survived'])
pd.crosstab(titanic_train_data['Survived'], pd.cut(titanic_train_data['Age'], bins=[0,16,30,45,60,80]))
plt.figure(figsize=(15,5))
plt.title("Age vs. Survival")
sns.countplot(pd.cut(titanic_train_data['Age'], bins=[0,16,30,45,60,80]), hue=titanic_train_data['Survived'])
pd.crosstab(titanic_train_data['Survived'], titanic_train_data['SibSp'])
plt.figure(figsize=(15,5))
plt.title("Having siblings/spouse vs. Survival")
sns.countplot(titanic_train_data['SibSp'], hue=titanic_train_data['Survived'])
pd.crosstab(titanic_train_data['Survived'], pd.cut(titanic_train_data['SibSp'], bins=[-1,0,1,8]))
pd.crosstab(titanic_train_data['Survived'], titanic_train_data['Parch'])
pd.crosstab(titanic_train_data['Survived'], pd.cut(titanic_train_data['Parch'], bins=[-1,0,6]))
plt.figure(figsize=(15,5))
plt.title("Having parents/children vs. Survival")
sns.countplot(pd.cut(titanic_train_data['Parch'], bins=[-1,0,6]), hue=titanic_train_data['Survived'])
pd.crosstab(titanic_train_data['Survived'], pd.qcut(titanic_train_data['Fare'], 4))
plt.figure(figsize=(15,5))
plt.title("Fare vs. Survival")
sns.countplot(pd.qcut(titanic_train_data['Fare'], 4), hue=titanic_train_data['Survived'])
pd.crosstab(titanic_train_data['Survived'], titanic_train_data['Embarked'])
plt.figure(figsize=(15,5))
plt.title("Embarked vs. Survival")
sns.countplot(titanic_train_data['Embarked'], hue=titanic_train_data['Survived'])
titanic_train_data['Name'].head(50)
def titleMrs(name):
    return 1*("Mrs." in name)

def titleMiss(name):
    return 1*("Miss." in name)

def titleMr(name):
    return 1*("Mr." in name)

def titleMaster(name):
    return 1*("Master." in name)

def titleDon(name):
    return 1*("Don." in name)
titanic_train_data['TitleMrs.']=titanic_train_data['Name'].apply(titleMrs)
titanic_train_data['TitleMiss.']=titanic_train_data['Name'].apply(titleMiss)
titanic_train_data['TitleMr.']=titanic_train_data['Name'].apply(titleMr)
titanic_train_data['TitleMaster.']=titanic_train_data['Name'].apply(titleMaster)
titanic_train_data['TitleDon.']=titanic_train_data['Name'].apply(titleDon)
titanic_train_data.head(10)
pd.crosstab(titanic_train_data['Survived'], titanic_train_data['TitleMrs.'])
pd.crosstab(titanic_train_data['Survived'], titanic_train_data['TitleMr.'])
pd.crosstab(titanic_train_data['Survived'], titanic_train_data['TitleMiss.'])
pd.crosstab(titanic_train_data['Survived'], titanic_train_data['TitleMaster.'])
pd.crosstab(titanic_train_data['Survived'], titanic_train_data['TitleDon.'])
titanic_train_data.drop('TitleDon.', axis=1, inplace=True)
y=titanic_train_data.Survived
X=titanic_train_data.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
X.head(30)
X.loc[X.Sex == 'male', 'Sex'] = 0
X.loc[X.Sex == 'female', 'Sex'] = 1
X.head(20)
X_one_hot = pd.get_dummies(X)
X_one_hot.head(50)
from sklearn.feature_selection import mutual_info_classif
# We have to drop Age temporarily just for mutual info since it has NaN values.
X_one_hot.columns.drop('Age')
np.round(mutual_info_classif(X_one_hot.drop('Age', axis=1), y, discrete_features=True),2)
dict(zip(X_one_hot.columns.drop('Age'),
         np.round(mutual_info_classif(X_one_hot.drop('Age', axis=1), y, discrete_features=True),2)))
df = X_one_hot.copy()
df['Survived'] = titanic_train_data['Survived']
df.corr()['Survived'].sort_values()
X_one_hot.drop(['SibSp', 'Embarked_Q', 'TitleMaster.', 'Parch'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_one_hot, 
                                                    y,
                                                    train_size=0.7, 
                                                    test_size=0.3, 
                                                    random_state=2)
from sklearn.preprocessing import Imputer
my_imputer = Imputer()

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_train.columns = X_one_hot.columns

imputed_X_test = pd.DataFrame(my_imputer.transform(X_test))
imputed_X_test.columns = X_one_hot.columns
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=10, n_estimators=100, max_features=None, random_state=2)
rf.fit(imputed_X_train,y_train)
from sklearn.metrics import accuracy_score, confusion_matrix
pred = rf.predict(imputed_X_test)
accuracy_score(y_test, pred)
pred_proba = rf.predict_proba(imputed_X_test)[:,1]
df_results = pd.DataFrame()
df_results['Survived'] = y_test
df_results['PredProba'] = np.round(pred_proba,4)
df_results[df_results.Survived==0]['PredProba']
plt.figure(figsize=(15,5))
plt.title('Histogram of survival probabilities')
plt.ylabel('Number of passengers')
sns.distplot(df_results[df_results.Survived==0]['PredProba'], kde=False, bins=10, label='Deceased')
sns.distplot(df_results[df_results.Survived==1]['PredProba'], kde=False, bins=10, label='Survived')
plt.legend(loc=1)
confusion_matrix(y_test, pred)
X_comparison = X_test.copy()
X_comparison['Survived'] = y_test
X_comparison['Predicted'] = pred
X_comparison
Wrong_predicted = X_comparison[X_comparison.Survived != X_comparison.Predicted]
Wrong_predicted
from sklearn.model_selection import cross_val_score
imputed_X_one_hot = pd.DataFrame(my_imputer.fit_transform(X_one_hot))
imputed_X_one_hot.columns = X_one_hot.columns
scores = cross_val_score(rf, imputed_X_one_hot, y, cv=5, scoring='accuracy')
scores