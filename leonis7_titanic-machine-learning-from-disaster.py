# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting
import seaborn as sns # complex plotting

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load train data
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
display(train_data.head())

# Load test data
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
display(test_data.head())
# Explore a pattern
def bar_chart(feature):
    survived = train_data[train_data["Survived"] == 1][feature].value_counts()
    non_survived = train_data[train_data["Survived"] == 0][feature].value_counts()
    df = pd.DataFrame([survived, non_survived])
    df.index = ['Survived', 'None']
    df.plot(kind ='bar', stacked=True, figsize=(10, 5))
# Problem Analysis
# Explore a pattern
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)
print("% of men who survived:", rate_men)

bar_chart('Sex')
# Explore a pattern
young = train_data.loc[train_data.Age < 40]["Survived"]
rate_young = sum(young)/len(young)
print("% of young who survived:", rate_young)

old = train_data.loc[train_data.Age >= 40]["Survived"]
rate_old = sum(old)/len(old)
print("% of old who survived:", rate_old)
# Explore a pattern
pc1 = train_data.loc[train_data.Pclass == 1]["Survived"]
rate_pc1 = sum(pc1)/len(pc1)
print("% of Pclass 1 who survived:", rate_pc1)

pc2 = train_data.loc[train_data.Pclass == 2]["Survived"]
rate_pc2 = sum(pc2)/len(pc2)
print("% of Pclass 2 who survived:", rate_pc2)

pc3 = train_data.loc[train_data.Pclass == 3]["Survived"]
rate_pc3 = sum(pc3)/len(pc3)
print("% of Pclass 3 who survived:", rate_pc3)

bar_chart('Pclass')
# Explore a pattern
fare = train_data.loc[train_data.Fare > 76]["Survived"]
rate_fare = sum(fare)/len(fare)
print("% of fare who survived:", rate_fare)

bar_chart('SibSp')
bar_chart('Parch')
bar_chart('Embarked')
# Data Integration
# Inspect data to find inconsistency
train_data.info()

sns.heatmap(train_data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
# Inspect test data to find inconsistency
test_data.info()

sns.heatmap(test_data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
# Feature Engineering

# Get name prefix
def get_titles(data):
    return data.apply(lambda x: x.split(',')[1].split('.')[0].strip())

# update data (we no longer need full name)
train_data['Title'] = get_titles(train_data['Name'])
test_data['Title'] = get_titles(test_data['Name'])

# check
train_data.head()
# Preprocessing & Feature Engineering

# check unique name prefix from data for feature encoding
display(train_data['Title'].unique())
display(test_data['Title'].unique())

bar_chart('Title')
# Preprocessing & Feature Engineering
# Ordinal encoding: create name prefix dictionary based on value
# children gets most importance: Mr: 0, Miss: 1, Mrs: 2, Others: 0
prefix_dict = {
    'Mr': 0,
    'Mrs': 2, 
    'Miss': 1,
    'Master': 3,
    'Don': 3,
    'Rev': 3,
    'Dr': 3,
    'Mme': 3,
    'Ms': 3,
    'Major': 3,
    'Lady': 3,
    'Sir': 3,
    'Mlle': 3,
    'Col': 3, 
    'Capt': 3,
    'the Countess': 3,
    'Jonkheer': 3,
    'Dona': 3 
}

# replace data
train_data['Title'] = train_data['Title'].map(prefix_dict)
test_data['Title'] = test_data['Title'].map(prefix_dict)

train_data.head()
bar_chart('Title')
train_data.drop(['Name'], axis=1, inplace=True)
test_data.drop(['Name'], axis=1, inplace=True)
# Preprocessing & Feature Engineering
# Label Encoding
from sklearn.preprocessing import LabelEncoder
le_train_data = LabelEncoder()
le_test_data = LabelEncoder()
train_data["Sex"] = le_train_data.fit_transform(train_data["Sex"])
test_data["Sex"] = le_test_data.fit_transform(test_data["Sex"])

train_data.head()
# Age data
plt.figure(figsize=(12, 7))
sns.boxplot(x='Title',y='Age',data=train_data, palette='winter')
# Preprocessing
# Impute age columns based on median from passenger class
train_data['Age'].fillna(train_data.groupby('Title')['Age'].transform('median'), inplace=True)
test_data['Age'].fillna(train_data.groupby('Title')['Age'].transform('median'), inplace=True)
sns.set_style('darkgrid')
facet = sns.FacetGrid(train_data, hue='Survived', aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train_data['Age'].max()))
facet.add_legend()

plt.show()
facet = sns.FacetGrid(train_data, hue='Survived', aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train_data['Age'].max()))
facet.add_legend()
plt.xlim(0, 20)
facet = sns.FacetGrid(train_data, hue='Survived', aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train_data['Age'].max()))
facet.add_legend()
plt.xlim(20, 30)
facet = sns.FacetGrid(train_data, hue='Survived', aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train_data['Age'].max()))
facet.add_legend()
plt.xlim(30, 40)
facet = sns.FacetGrid(train_data, hue='Survived', aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train_data['Age'].max()))
facet.add_legend()
plt.xlim(40, 60)
# Binning
# child = 0, young = 1, adult = 2, mid-age = 3, older = 4
train_data['AgeGroup'] = pd.cut(train_data['Age'], bins=[0, 16, 26, 36, 62, 100], labels=False, precision=0)
test_data['AgeGroup'] = pd.cut(test_data['Age'], bins=[0, 16, 26, 36, 62, 100], labels=False, precision=0)
bar_chart('AgeGroup')
bar_chart('Embarked')
pc1 = train_data[train_data["Pclass"] == 1]["Embarked"].value_counts()
pc2 = train_data[train_data["Pclass"] == 2]["Embarked"].value_counts()
pc3 = train_data[train_data["Pclass"] == 3]["Embarked"].value_counts()
df = pd.DataFrame([pc1, pc2, pc3])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind ='bar', stacked=True, figsize=(10, 5))
# Preprocessing & Feature Engineering
# Ordinal encoding of Embarked data
train_data["Embarked"].fillna('S', inplace=True)
test_data["Embarked"].fillna('S', inplace = True)

embarked_map = {
    'S' : 0,
    'C' : 1,
    'Q' : 2
}

# replace data
train_data['Embarked'] = train_data['Embarked'].map(embarked_map)
test_data['Embarked'] = test_data['Embarked'].map(embarked_map)

train_data.head()
# Fare
# fill missing Fare data with mean value
test_data['Fare'].fillna(test_data.groupby('Pclass').transform('median')['Fare'], inplace=True)

sns.heatmap(test_data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
facet = sns.FacetGrid(train_data, hue='Survived', aspect=4)
facet.map(sns.kdeplot, 'Fare', shade=True)
facet.set(xlim=(0, train_data['Fare'].max()))
facet.add_legend()

plt.show()
facet = sns.FacetGrid(train_data, hue='Survived', aspect=4)
facet.map(sns.kdeplot, 'Fare', shade=True)
facet.set(xlim=(0, train_data['Fare'].max()))
facet.add_legend()
plt.xlim(0, 20)
facet = sns.FacetGrid(train_data, hue='Survived', aspect=4)
facet.map(sns.kdeplot, 'Fare', shade=True)
facet.set(xlim=(0, train_data['Fare'].max()))
facet.add_legend()
plt.xlim(0, 30)
facet = sns.FacetGrid(train_data, hue='Survived', aspect=4)
facet.map(sns.kdeplot, 'Fare', shade=True)
facet.set(xlim=(0, train_data['Fare'].max()))
facet.add_legend()
plt.xlim(30, 100)
facet = sns.FacetGrid(train_data, hue='Survived', aspect=4)
facet.map(sns.kdeplot, 'Fare', shade=True)
facet.set(xlim=(0, train_data['Fare'].max()))
facet.add_legend()
plt.xlim(100, 600)
# Binning
# child = 0, young = 1, adult = 2, mid-age = 3, older = 4
train_data['FareGroup'] = pd.cut(train_data['Fare'], bins=[-1, 17, 30, 100, 1000], labels=False, precision=0)
test_data['FareGroup'] = pd.cut(test_data['Fare'], bins=[-1, 17, 30, 100, 1000], labels=False, precision=0)
bar_chart('FareGroup')
train_data.drop(['Age', 'Fare'], axis=1, inplace=True)
test_data.drop(['Age', 'Fare'], axis=1, inplace=True)
# Preprocessing
# Impute age column in train dataset with LinearRegression using other columns
"""from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

# replace age
def ageRegression(data):
    age_data = data.drop(["Ticket", "Cabin"], axis=1)

    test_age_data = age_data[age_data['Age'].isna()].drop(['Age'], axis=1)

    train_age_data = age_data[~age_data['Age'].isna()]
    train_age = train_age_data["Age"]
    train_age_data_no_age = train_age_data.drop(['Age'], axis=1)

    # Run Model to find 
    model = LinearRegression()
    model.fit(train_age_data_no_age, train_age)
    
    # update test part
    test_age_data['Age'] = np.abs(np.ceil(model.predict(test_age_data)))
    # merge with train part and return
    out = pd.concat([train_age_data, test_age_data], axis=0)
    out["Ticket"] = data["Ticket"]
    out["Cabin"] = data["Cabin"]
    return out

train_data_new = ageRegression(train_data)
test_data_new = ageRegression(test_data)

display(train_data_new.head())
display(test_data_new.head())

sns.heatmap(train_data_new.isnull(), yticklabels=False, cbar=False, cmap='viridis')"""
#!pip install datawig
#import datawig
# Feature Engineering
# Use only first character of Cabin for simplicity
train_data['Cabin'] = train_data['Cabin'].str[:1]
test_data['Cabin'] = test_data['Cabin'].str[:1]

train_data.head()
pc1 = train_data[train_data["Pclass"] == 1]["Cabin"].value_counts()
pc2 = train_data[train_data["Pclass"] == 2]["Cabin"].value_counts()
pc3 = train_data[train_data["Pclass"] == 3]["Cabin"].value_counts()
df = pd.DataFrame([pc1, pc2, pc3])
df.index = ['1st class', '2nd class', '3rd class']
df.plot(kind ='bar', stacked=True, figsize=(10, 5))
# Feature Scaling
# Ordinal encoding of Cabin data
cabin_map = {
    'A' : 0.0,
    'B' : 0.4,
    'C' : 0.8,
    'D' : 1.2,
    'E' : 1.6,
    'F' : 2.0,
    'G' : 2.4,
    'T' : 2.8
}

# replace data
train_data['Cabin'] = train_data['Cabin'].map(cabin_map)
test_data['Cabin'] = test_data['Cabin'].map(cabin_map)

train_data.head()
# fill missing Cabin data with mean value
train_data['Cabin'].fillna(train_data.groupby('Pclass')['Cabin'].transform('median'), inplace=True)
test_data['Cabin'].fillna(test_data.groupby('Pclass')['Cabin'].transform('median'), inplace=True)

train_data.head()
# Preprocessing & Feature Engineering

# add additional feature from: Parch & SibSp
def process_family(data):
    data['Family'] = data['Parch'] + data['SibSp'] + 1 
    return data

train_data = process_family(train_data)
test_data = process_family(test_data)

train_data.head()
facet = sns.FacetGrid(train_data, hue='Survived', aspect=4)
facet.map(sns.kdeplot, 'Family', shade=True)
facet.set(xlim=(0, train_data['Family'].max()))
facet.add_legend()

plt.show()
# Ordinal encoding of Family data
family_map = { 1:0, 2:0.4, 3:0.8, 4:1.2, 5:1.6, 6:2, 7:2.4, 8:2.8, 9:3.2, 10:3.6, 11:4 }

# replace data
train_data['Family'] = train_data['Family'].map(family_map)
test_data['Family'] = test_data['Family'].map(family_map)

train_data.head()
train_data.drop(['Parch', 'SibSp'], axis=1, inplace=True)
test_data.drop(['Parch', 'SibSp'], axis=1, inplace=True)

train_data.head()
"""df_train, df_test = datawig.utils.random_split(train_data_new)

# Initialize a SimpleImputer model
imputer = datawig.SimpleImputer(
    input_columns=['Survived','Pclass','Name','Age','SibSp','Parch','Fare','Sex_male','Embarked_Q','Embarked_S','Family'], # column(s) containing information about the column we want to impute
    output_column= 'Cabin', # the column we'd like to impute values for
    output_path = 'imputer_model' # stores model data and metrics
    )

# Fit an imputer model on the train data
imputer.fit(train_df=df_train, num_epochs=10)

# Impute missing train Cabin values and return original dataframe with predictions
imputed_train = imputer.predict(train_data_new)

# Impute missing test Cabin values
df_train, df_test = datawig.utils.random_split(test_data_new)
imputer = datawig.SimpleImputer(
    input_columns=['Pclass','Name','Age','SibSp','Parch','Fare','Sex_male','Embarked_Q','Embarked_S','Family'], # column(s) containing information about the column we want to impute
    output_column= 'Cabin', # the column we'd like to impute values for
    output_path = 'imputer_model' # stores model data and metrics
    )
imputer.fit(train_df=df_train, num_epochs=10)
imputed_test = imputer.predict(test_data_new)"""
"""# Cabin imputation using deep learning (Datawig)
train_data_new['Cabin'].fillna(imputed_train['Cabin_imputed'], inplace=True)
test_data_new['Cabin'].fillna(imputed_test['Cabin_imputed'], inplace=True)"""
"""# Preprocessing

# Apply one hot encoding with k-1 columns
train_data_Cabin = pd.get_dummies(train_data_new['Cabin'].str[0], drop_first=True, prefix='Cabin')
test_data_Cabin = pd.get_dummies(test_data_new['Cabin'].str[0], drop_first=True, prefix='Cabin')

# Concat new columns to old datasets
train_data_new = pd.concat([train_data_new.drop(['Cabin'], axis=1), train_data_Cabin], axis=1)
test_data_new = pd.concat([test_data_new.drop(['Cabin'], axis=1), test_data_Cabin], axis=1)

display(train_data_new.head())

display(test_data_new.head())"""
"""# Additional column to keep consistency
test_data_new['Cabin_T'] = 0"""
# Preprocessing
# Tag passsengers having special ticket numbers
train_data['Special'] = train_data['Ticket'].apply(lambda x: 0 if x.split(' ')[0].isnumeric() else 1 )
test_data['Special'] = train_data['Ticket'].apply(lambda x: 0 if x.split(' ')[0].isnumeric() else 1 )

bar_chart('Special')
# Extract first two digits of ticket data (Replace LINE with 0)
train_data['Ticket'] = train_data['Ticket'].apply(lambda x: x.split(' ')[len(x.split(' ')) - 1]).apply(lambda x: 0 if x == "LINE" else x).apply(lambda x: str(x)[:2]).apply(lambda x: float(x)/100)
test_data['Ticket'] = test_data['Ticket'].apply(lambda x: x.split(' ')[len(x.split(' ')) - 1]).apply(lambda x: str(x)[:2]).apply(lambda x: float(x)/100)

#train_data[~train_data['Ticket'].apply(lambda x: str(x).isnumeric())]
train_data.head()
facet = sns.FacetGrid(train_data, hue='Survived', aspect=4)
facet.map(sns.kdeplot, 'Ticket', shade=True)
facet.set(xlim=(0, train_data['Ticket'].max()))
facet.add_legend()

plt.show()
# binning
train_data['TicketGroup'] = pd.cut(train_data['Ticket'], bins=[-1, 0.42, 0.84, 1], labels=False, precision=0)
test_data['TicketGroup'] = pd.cut(test_data['Ticket'], bins=[-1, 0.42, 0.84, 1], labels=False, precision=0)

bar_chart('TicketGroup')
passengerId = test_data['PassengerId']
train_data.drop(['Ticket', 'PassengerId'], axis=1, inplace=True)
test_data.drop(['Ticket', 'PassengerId'], axis=1, inplace=True)

train_data.head()
train_data.info()
# install packages uninstalled during datawig install
"""!pip install 'scikit-learn==0.22.2.post1'
!pip install 'typing==3.7.4.1'
!pip install 'pandas==1.0.3'
!pip install 'mxnet==1.6.0'"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
# LogisticRegression
clf = LogisticRegression(max_iter=1000)
scoring = 'accuracy'
score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

# LogisticRegression Score
print(round(np.mean(score)*100, 2))
# kNN
clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

# kNN Score
print(round(np.mean(score)*100, 2))
# DecisionTreeClassifier
clf = DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

# DecisionTreeClassifier Score
print(round(np.mean(score)*100, 2))
# RandomForestClassifier
clf = RandomForestClassifier(n_estimators=13)
scoring = 'accuracy'
score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

# RandomForestClassifier Score
print(round(np.mean(score)*100, 2))
# GradientBoostingClassifier
clf = GradientBoostingClassifier(learning_rate = 0.1, 
                    max_depth = 2,
                    min_samples_split = 10,
                    n_estimators = 200,
                    subsample = 0.6)
scoring = 'accuracy'
score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

# GradientBoostingClassifier Score
print(round(np.mean(score)*100, 2))
# GaussianNB
clf = GaussianNB()
scoring = 'accuracy'
score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

# GaussianNB Score
print(round(np.mean(score)*100, 2))
# SVC
clf = SVC()
scoring = 'accuracy'
score = cross_val_score(clf, X_train, y_train, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

# GaussianNB Score
print(round(np.mean(score)*100, 2))
clf = SVC()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
output = pd.DataFrame({'PassengerId': passengerId, 'Survived': predictions})
output.to_csv('submission_svm.csv', index=False)
print("Submission data successfully saved!")

print("Train accuracy: {}".format(round(model.score(X_train, y_train), 4)))

output.head()