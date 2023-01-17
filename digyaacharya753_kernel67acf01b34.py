!wget -O gender_submission.csv "https://drive.google.com/uc?export=download&id=10cAqc8q7hiVLQdQvwQMNa-Iq9eu-RPif"
!wget -O train.csv "https://drive.google.com/uc?export=download&id=1X-Ekwi5nYl4SYs3lD6pA2f62uLvFCz5y"
!wget -O test.csv "https://drive.google.com/uc?export=download&id=1olLJlYrYytKQ1xcJybTVpcvFfAK1mCQh"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
df_train = pd.read_csv("train.csv")
df_train.head()
df_train.columns
df_train.shape
df_test = pd.read_csv("test.csv")
df_test.head()
pId = df_test['PassengerId']
df_test.shape
df_gender_submit = pd.read_csv("gender_submission.csv")
df_gender_submit.head()
import pandas_profiling
pandas_profiling.ProfileReport(df_train)
pandas_profiling.ProfileReport(df_test)
df = pd.concat([df_train.loc[:, df_train.columns != 'Survived'], df_test], axis=0, sort=False, ignore_index=True)
survived = df_train['Survived']
df.info(); df.describe()
print(df_train['Survived'].value_counts())
print(df_train['Survived'].value_counts(normalize=True))
count = df_train['Survived'].value_counts().index.values
values = df_train['Survived'].value_counts().to_list()
plt.bar(count, values, alpha=0.9)
names = ['Not Survived', 'Survived']
plt.xticks(count, names)
plt.xlabel('Survival Rate')
plt.ylabel('Number of people')
df_train[df_train['Sex'] == "female"]
print((df_train['Sex'] == "female").value_counts())
print((df_train['Sex'] == "female").value_counts(normalize=True))
count = (df_train['Sex']=="female").value_counts().index.values
values = (df_train['Sex']=="female").value_counts().to_list()
plt.bar(count, values, alpha=0.9)
names = ['Male', 'Female']
plt.xticks(count, names)
plt.xlabel('Sex')
plt.ylabel('Number of people')
l_gender = ['female', 'male']
for gender in l_gender:
  print(df_train["Survived"][df_train["Sex"]==gender].value_counts())
  print(df_train["Survived"][df_train["Sex"]==gender].value_counts(normalize=True))
import seaborn as sns
sns.countplot(x="Sex", hue="Survived", data=df_train, alpha=0.9)
l_pclass=[1, 2, 3]
for pclass in l_pclass:
  print((df_train["Pclass"]==pclass).value_counts())
sns.countplot(x="Pclass", hue="Survived", data=df_train, alpha=0.9)
l_pclass=[1, 2, 3]
for pclass in l_pclass:
  print(df_train["Survived"][df_train["Pclass"]==pclass].value_counts())
l_pclass=[1, 2, 3]
for pclass in l_pclass:
  print(df_train["Survived"][df_train["Pclass"]==pclass].value_counts(normalize=True))
df_train.columns
fig, axs = plt.subplots(1,3, figsize=(20,7))
sns.countplot(x="SibSp", hue="Survived", data=df_train, ax=axs[0], alpha=0.8)
sns.countplot(x="Parch", hue="Survived", data=df_train, ax=axs[1], alpha=0.8)
sns.countplot(x="Embarked", hue="Survived", data=df_train, ax=axs[2], alpha=0.8)
plt.show()
sns.factorplot('Pclass','Survived',hue='Sex',data=df_train, alpha=0.8)
sns.factorplot('SibSp','Survived',hue='Sex',data=df_train, alpha=0.8)
sns.factorplot('Embarked','Survived',hue='Sex',data=df_train, alpha=0.8)
l_sibsp = [0, 1, 2, 3, 4, 5, 6, 7, 8]
for sibsp in l_sibsp:
  print(df_train["Survived"][df_train["SibSp"]==sibsp].value_counts(normalize=True))
l_parch = [0, 1, 2, 3, 4, 5, 6]
for parch in l_parch:
  print(df_train["Survived"][df_train["Parch"]==parch].value_counts(normalize=True))
l_embarked = ['S', 'C', 'Q']
for embarked in l_embarked:
  print(df_train["Survived"][df_train["Embarked"]==embarked].value_counts(normalize=True))
age_range = ['0-18', '19-34', '35-50', '51-69', '70-87', '87+' ]
child = df_train['Age'].between(0, 18, inclusive=True).sum()
millennial = df_train['Age'].between(19, 34, inclusive=True).sum()
generationX = df_train['Age'].between(35, 50, inclusive=True).sum()
boomer = df_train['Age'].between(51, 69, inclusive=True).sum()
silent = df_train['Age'].between(70, 87, inclusive=True).sum()
remaining = df_train['Age'].gt(87).sum()
age_groups = [child, millennial, generationX, boomer, silent, remaining]
plt.bar(age_range,  age_groups, alpha=0.9)
plt.xlabel('Age')
plt.ylabel('Number of people')
def create_age_labels(df):
  conditions = [
    (df['Age'].round(decimals=0).between(0, 18, inclusive=True)),
    (df['Age'].round(decimals=0).between(19, 34, inclusive=True)),
    (df['Age'].round(decimals=0).between(35, 50, inclusive=True)),
    (df['Age'].round(decimals=0).between(51, 69, inclusive=True)),
    (df['Age'].round(decimals=0).between(70, 87, inclusive=True)),
    (df['Age'].round(decimals=0).gt(87))] 
  df['age_labels'] = np.select(conditions, age_range, default='null')
  return df 

df_train = create_age_labels(df_train)
# for whole df including test data
df = create_age_labels(df)
ax = sns.countplot(x="age_labels", hue="Survived", data=df_train, alpha=0.8)
plt.xlabel('Age')
plt.ylabel('Number of people')
print(df_train["Survived"][df_train['Age'].between(0, 18, inclusive=True)].value_counts(normalize=True))
print(df_train["Survived"][df_train['Age'].between(19, 34, inclusive=True)].value_counts(normalize=True))
print(df_train["Survived"][df_train['Age'].between(35, 50, inclusive=True)].value_counts(normalize=True))
print(df_train["Survived"][df_train['Age'].between(51, 69, inclusive=True)].value_counts(normalize=True))
print(df_train["Survived"][df_train['Age'].between(70, 87, inclusive=True)].value_counts(normalize=True))
print(df_train["Survived"][df_train['Age'].gt(87)].value_counts(normalize=True))
ax = sns.countplot(x="age_labels", hue="Survived", data=df_train[df_train.Sex == "female"], alpha=0.8)
ax = sns.countplot(x="age_labels", hue="Survived", data=df_train[df_train.Sex == "male"], alpha=0.8)
df_train.columns
fig = plt.figure(figsize=(15,8),)
sns.kdeplot(df_train["Fare"][df_train['Survived'] == 1], color="lightcoral", shade=True)
sns.kdeplot(df_train["Fare"][df_train['Survived'] == 0], color="darkturquoise", shade=True)
plt.legend(['Survived', 'Not Survived']) 
plt.ylabel('Frequency of passengers survived')
plt.xlabel('Fare');
plt.xlim(-20,600)
plt.show()
sns.heatmap(df.isnull(), cbar = False).set_title("Missing values heatmap")
ax = df_train["Age"].hist(bins=70, color='blue', alpha=0.5)
ax.set(xlabel='Age', ylabel='Count')
plt.show()
df["Age"] = df["Age"].fillna(df["Age"].median(skipna=True))
df['Cabin'].fillna("N", inplace=True)
df['Cabin'] = [i[0] for i in df['Cabin']]
df['Cabin'].value_counts()
df['Embarked'].value_counts()
df['Embarked'] = df['Embarked'].fillna('S')
df[df["Fare"].isnull()]
df.head()
df.query('Pclass == 3 and Sex == "male" and Age > 60 and Parch == 0 and SibSp == 0 and Cabin == "N" and Embarked == "S"')
avg_fare = (df.loc[326, 'Fare'] + df.loc[851, 'Fare'])/2
avg_fare
df['Fare'] = df['Fare'].fillna(avg_fare)
df.isnull().sum()
mask = np.zeros_like(df_train.corr(), dtype=np.bool)

plt.subplots(figsize = (10,12))
sns.heatmap(df_train.corr(), 
            annot=True,
            mask = mask,
            cmap = 'RdBu',
            square=True)
plt.title("Correlations Among Features");
def create_relatives_series(df):
  df['relatives'] = df['SibSp'] + df['Parch']
  return df
df_train = create_relatives_series(df_train)
df = create_relatives_series(df)
data = [df, df_train]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr',\
                                            'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].map(titles)
    dataset['Title'] = dataset['Title'].fillna(0)
data = [df, df_train]

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()

df_2 = df.apply(le.fit_transform)
df_2.head()
df_2.columns
X_enc = df_2[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'age_labels', 'Title', 'relatives']].copy()
X_enc = pd.get_dummies(X_enc, columns=['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Cabin', 'age_labels', 'Title'])
X_enc.head()
print(X_enc.columns)
print(X_enc.shape)
X = X_enc.loc[:, ['Fare','relatives','Pclass_0', 'Pclass_1', 'Pclass_2', 'Sex_0', 'Sex_1', 'SibSp_0', 'SibSp_1', 'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_6', 'Parch_0', 'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6', 'Parch_7', 'Embarked_0', 'Embarked_1', 'Embarked_2', 'Cabin_0', 'Cabin_1','Cabin_2', 'Cabin_3', 'Cabin_4', 'Cabin_5', 'Cabin_6', 'Cabin_7','Cabin_8', 'age_labels_0', 'age_labels_1', 'age_labels_2','age_labels_3', 'age_labels_4', 'age_labels_5','Title_0', 'Title_1', 'Title_2', 'Title_3', 'Title_4' ]]
X_train = X[0: 891]
y_train= survived
X_test = X[891:]
print(X_train.shape)
print(len(y_train))
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
scores = cross_val_score(rf, X_train, y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(rf.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.head(15)
importances.plot.bar()
not_important = ['Fare', 'SibSp_4', 'Cabin_0','SibSp_3','SibSp_2','SibSp_5','SibSp_6','Cabin_0','age_labels_4', 'Cabin_5', 'Cabin_6', 'Parch_3', 'Parch_4', 'SibSp_6', 'Parch_5', 'Cabin_8', 'SibSp_5', 'Parch_6', 'Parch_7']
X_train  = X_train.drop(not_important, axis=1)
X_test  = X_test.drop(not_important, axis=1)
param_grid = { "criterion" : ["gini", "entropy"], "min_samples_leaf" : [1, 5, 10, 25, 50, 70], "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35], "n_estimators": [100, 400, 700, 1000, 1500]}
from sklearn.model_selection import GridSearchCV, cross_val_score
rf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=1, n_jobs=-1)
clf = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1)
clf.fit(X_train, y_train)
clf.best_estimator_
# Random Forest with best parameter values
random_forest = RandomForestClassifier(criterion = "gini", 
                                       min_samples_leaf = 1, 
                                       min_samples_split = 10,   
                                       n_estimators=400, 
                                       max_features='auto', 
                                       oob_score=True, 
                                       random_state=1, 
                                       n_jobs=-1)

random_forest.fit(X_train, y_train)

print(random_forest.score(X_train, y_train))

print("oob score:", round(random_forest.oob_score_, 4)*100, "%")
prediction = model2.predict(X_test)
df_final = pd.DataFrame()
df_final['PassengerId'] = pId
df_final['Survived'] = prediction
df_final['Survived'].value_counts()