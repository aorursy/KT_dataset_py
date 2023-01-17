# Pandas & Numpy
import pandas as pd
import numpy as np
import pandas_profiling

# Data Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Machine learning models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mpl.style.use(['ggplot']) 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test= pd.read_csv("/kaggle/input/titanic/test.csv")

print('Data loaded !')
print(train.shape)
train.head()
print(test.shape)
test.head()
train.profile_report()
test.profile_report()
train.info()
train.describe()
train["Survived"].value_counts().plot(kind='pie',
                                      figsize=(16,8),
                                      autopct='%1.1f',
                                      startangle=90,
                                      colors=['lightcoral', 'lightgreen'],
                                      labels=None,
                                      explode=[0.07, 0])
plt.title('Survival percentage')
plt.axis('equal')
plt.legend(labels=['Did not survive', 'Survived'], loc='upper left')
plt.show()
train["Sex"].value_counts().plot(kind='pie',
                                 figsize=(16,8),
                                 autopct='%1.1f',
                                 startangle=90,
                                 colors=['lightblue', 'pink'],
                                 labels=None,
                                 explode=[0.07, 0])
plt.title('Male/Female percentage')
plt.axis('equal')
plt.legend(labels=['Male', 'Female'], loc='upper left')
plt.show()
data_sex_survived = train[['Sex','Survived']]

d1 = data_sex_survived[data_sex_survived["Survived"] == 1].groupby("Sex").count()
d2 = data_sex_survived[data_sex_survived["Survived"] == 0].groupby("Sex").count().rename(columns={'Survived':'Did not Survive'})

sex_survived_count = d1.merge(d2, left_on='Sex',right_on='Sex')
sex_survived_count = sex_survived_count.div(sex_survived_count.sum(axis=1), axis=0)

sex_survived_count.plot(kind='bar',
                        figsize=(16,8),
                       color=['lightgreen','lightcoral'])

plt.title('Survival rate based on Sex',size=19)

plt.show()
train["Pclass"].value_counts().plot(kind='pie',
                                    figsize=(16,8),
                                    autopct='%1.1f',
                                    startangle=90,
                                    colors=['lightcoral','maroon','mistyrose'],
                                    labels=None)

plt.title('Ticket classes proportions within passengers')
plt.axis('equal')
plt.legend(labels=['Class 3', 'Class 1', 'Class 2'], loc='upper left', fontsize=14)
plt.show()
data_pclass_survived = train[['Pclass','Survived']]

d1 = data_pclass_survived[data_sex_survived["Survived"] == 1].groupby("Pclass").count()
d2 = data_pclass_survived[data_sex_survived["Survived"] == 0].groupby("Pclass").count().rename(columns={'Survived':'Did not Survive'})
pclass_survived_count = d1.merge(d2, left_on='Pclass',right_on='Pclass')

pclass_survived_count = pclass_survived_count.div(pclass_survived_count.sum(axis=1), axis=0)

pclass_survived_count.plot(kind='bar',
                           figsize=(16,8),
                           color=['lightgreen','lightcoral'])

plt.title('Survival rate based on classes',size=19)

plt.show()
train['SibSp'].value_counts().plot(kind='bar',
                                   figsize=(16,8),
                                   color='rosybrown')

plt.title('Number of passengers based on number of siblings / spouses',size=19)
plt.xlabel('Number of siblings')
plt.ylabel('Number of passengers')
plt.show()
data_SibSp_survived = train[['SibSp','Survived']]

d1 = data_SibSp_survived[data_sex_survived["Survived"] == 1].groupby("SibSp").count()
d2 = data_SibSp_survived[data_sex_survived["Survived"] == 0].groupby("SibSp").count().rename(columns={'Survived':'Did not Survive'})

SibSp_survived_count = d1.merge(d2, left_on='SibSp',right_on='SibSp')
SibSp_survived_count = SibSp_survived_count.div(SibSp_survived_count.sum(axis=1), axis=0)

SibSp_survived_count.plot(kind='bar',
                          figsize=(16,8),
                          color=['lightgreen','lightcoral'])

plt.title('Survival rate based on number of siblings / spouses',size=19)

plt.show()
train['Parch'].value_counts().plot(kind='bar',
                                   figsize=(16,8),
                                   color='rosybrown')

plt.title('Number of parents / children',size=19)

plt.show()
data_Parch_survived = train[['Parch','Survived']]

d1 = data_Parch_survived [data_sex_survived["Survived"] == 1].groupby("Parch").count()
d2 = data_Parch_survived [data_sex_survived["Survived"] == 0].groupby("Parch").count().rename(columns={'Survived':'Did not Survive'})

Parch_survived_count = d1.merge(d2, left_on='Parch',right_on='Parch')
Parch_survived_count = Parch_survived_count.div(Parch_survived_count.sum(axis=1), axis=0)

Parch_survived_count.plot(kind='bar',
                          figsize=(16,8),
                          color=['lightgreen','lightcoral'])

plt.title('Survival rate based on number of parents / children',size=19)

plt.show()
train["Embarked"].value_counts()
train["Embarked"].value_counts().plot(kind='pie',
                                    figsize=(16,8),
                                    autopct='%1.1f',
                                    startangle=90,
                                    colors=['lightcoral','maroon','mistyrose'],
                                    labels=None)

plt.title('Port of Embarkation proportions within passengers')
plt.axis('equal')
plt.legend(labels=['Southampton', 'Cherbourg', 'Queenstown'], loc='upper left', fontsize=14)
plt.show()
data_embarked_survived = train[['Embarked','Survived']]

d1 = data_embarked_survived[data_embarked_survived["Survived"] == 1].groupby("Embarked").count()
d2 = data_embarked_survived[data_embarked_survived["Survived"] == 0].groupby("Embarked").count().rename(columns={'Survived':'Did not Survive'})
data_embarked_survived_count = d1.merge(d2, left_on='Embarked',right_on='Embarked')

data_embarked_survived_count = data_embarked_survived_count.div(data_embarked_survived_count.sum(axis=1), axis=0)

data_embarked_survived_count.plot(kind='bar',
                           figsize=(16,8),
                           color=['lightgreen','lightcoral'])

plt.title('Survival rate based on port of embarkation',size=19)

plt.show()
train['Fare'].plot(kind='hist',
                  figsize=(16,8),
                  color='rosybrown')

plt.title('Passenger fare distribution',size=19)

plt.show()
data_fare_survived = train[['Fare','Survived']][train["Survived"]==1]
data_fare_survived_grouped = data_fare_survived.groupby('Fare').count()

data_fare_dsurvived = train[['Fare','Survived']][train["Survived"]==0]
data_fare_dsurvived_grouped = data_fare_dsurvived.groupby('Fare').count().rename(columns={'Survived':'Did not Survive'})

ax = data_fare_survived_grouped.plot(kind='area',
                                    figsize=(16,8),
                                    stacked=False,
                                    color='lightgreen')

data_fare_dsurvived_grouped.plot(kind='area',
                                figsize=(16,8),
                                stacked=False,
                                color='lightcoral',
                                ax=ax)

plt.title('Number of passengers - Surived/Did not Survive - by Fare')
plt.ylabel('Number of passengers')


plt.show()
pd.qcut(train['Fare'],4).value_counts()
data_fare_survived = train[['Fare', 'Survived']]
data_fare_survived['farecat'] = pd.qcut(data_fare_survived['Fare'],4, labels=['1st','2nd','3nd','4th'])
data_farecat_survived = data_fare_survived[['farecat','Survived']]

d1 = data_farecat_survived[data_farecat_survived["Survived"] == 1].groupby("farecat").count()
d2 = data_farecat_survived[data_farecat_survived["Survived"] == 0].groupby("farecat").count().rename(columns={'Survived':'Did not Survive'})
data_farecat_survived_count = d1.merge(d2, left_on='farecat',right_on='farecat')

data_farecat_survived_count = data_farecat_survived_count.div(data_farecat_survived_count.sum(axis=1), axis=0)

data_farecat_survived_count.plot(kind='bar',
                           figsize=(16,8),
                           color=['lightgreen','lightcoral'])

plt.title('Survival rate based on port of farecat',size=19)

plt.show()
train['Age'].plot(kind='hist',
                  figsize=(16,8),
                  color='rosybrown')

plt.title('Age distribution',size=20)

plt.show()
data_age_survived = train[['Age','Survived']][train["Survived"]==1]
data_age_survived_grouped = data_age_survived.groupby('Age').count()

data_age_dsurvived = train[['Age','Survived']][train["Survived"]==0]
data_age_dsurvived_grouped = data_age_dsurvived.groupby('Age').count().rename(columns={'Survived':'Did not Survive'})

ax = data_age_survived_grouped.plot(kind='area',
                                    figsize=(16,8),
                                    stacked=False,
                                    color='lightgreen')

data_age_dsurvived_grouped.plot(kind='area',
                                figsize=(16,8),
                                stacked=False,
                                color='lightcoral',
                                ax=ax)

plt.title('Number of passengers - Surived/Did not Survive - by Age', size=20)
plt.ylabel('Number of passengers')


plt.show()
female = train[train['Sex']=='female']
male = train[train['Sex']=='male']

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 6))


ax = sns.distplot(female[female['Survived']==1].Age.dropna(),
                  bins=20,
                  label = 'Survived',
                  ax = axes[0],
                  kde =False)

ax = sns.distplot(female[female['Survived']==0].Age.dropna(),
                  bins=20,
                  label = 'Did not survived',
                  ax = axes[0],
                  kde =False)
ax.legend()
ax.set_title('Female',size=20)

ax = sns.distplot(male[male['Survived']==1].Age.dropna(),
                  bins=20,
                  label = 'Survived',
                  ax = axes[1],
                  kde = False)

ax = sns.distplot(male[male['Survived']==0].Age.dropna(),
                  bins=20,
                  label = 'Did not survived',
                  ax = axes[1],
                  kde = False)
ax.legend()
_ = ax.set_title('Male',size=20)
data_relatives_survived = train[['Survived']]

data_relatives_survived['Relatives'] = train['Parch'] + train['SibSp']


g = sns.factorplot(x='Relatives',
                   y='Survived',
                   data=data_relatives_survived,
                   aspect = 2.5,
                   color='rosybrown')

_ = g.axes.flatten()[0].set_title('Survival rate based on relatives', size=20)
train_copy = train.copy()
train_copy['Cabin'].fillna('U00', inplace=True)

train_copy['Deck'] = train_copy['Cabin'].astype(str).str[0]
data_deck_survived = train_copy[['Deck','Survived']]

d1 = data_deck_survived[data_deck_survived["Survived"] == 1].groupby("Deck").count()
d2 = data_deck_survived[data_deck_survived["Survived"] == 0].groupby("Deck").count().rename(columns={'Survived':'Did not Survive'})
data_deck_survived_count = d1.merge(d2, left_on='Deck',right_on='Deck')

data_deck_survived_count = data_deck_survived_count.div(data_deck_survived_count.sum(axis=1), axis=0)

data_deck_survived_count.plot(kind='bar',
                           figsize=(16,8),
                           color=['lightgreen','lightcoral'])

plt.title('Survival rate based on port of deck',size=19)

plt.show()
for df in [train, test] :
    df.drop(['PassengerId','Ticket'],axis=1,inplace=True)
train.head()
for df in [train, test] :
    df['Title'] = df.Name.str.extract('([A-Za-z]+)\.')
train['Title'].value_counts()
for df in [train, test] :
    df['Title'].replace(['Dona','Lady','Mme','Countess','Ms'],'Mrs',inplace=True)
    df['Title'].replace(['Mlle'],'Miss',inplace=True)
    df['Title'].replace(['Sir','Capt','Don','Jonkheer','Col','Major','Rev','Dr','Major'],'Mr',inplace=True)
    df.drop(['Name'], axis=1,inplace=True)
    
train['Title'].value_counts()
train.head()
test.head()
for df in [train, test] :
    df['RelativesGroup'] = 0
    df['Relatives'] = df['Parch'] + df['SibSp']
    df.loc[(df['Relatives'] > 0) & (df['Relatives'] <= 3), 'RelativesGroup'] = 1
    df.loc[(df['Relatives'] >= 4), 'RelativesGroup'] = 2
    df.drop(['Relatives','Parch','SibSp'], axis=1,inplace=True)
    df['RelativesGroup'] = df['RelativesGroup'].astype('category')
train.isnull().sum()
test.isnull().sum()
for df in [train,test]:
    df['Age'] = df.groupby('Title')['Age'].apply(lambda x: x.fillna(x.mean()))
    
common = train['Embarked'].describe()['top']
train['Embarked'].fillna(common,inplace=True)
mean = test['Fare'].mean()
test['Fare'].fillna(mean,inplace=True)
for df in [train,test] :
    df['DeckUnkown'] = 1
    df.loc[(df['Cabin'].isnull()), 'DeckUnkown'] = 0
    df.drop(['Cabin'], axis=1,inplace=True)
train[['Age','Fare']].describe()
scaler = MinMaxScaler()
for df in [train,test]:
    df[['Age','Fare']] = scaler.fit_transform(df[['Age','Fare']])
train.head()
test.head()
for df in [train, test]:
    df['Sex'] = (df['Sex'] == "male").astype(int)
for df in [train, test]:
    df['Pclass'] = df['Pclass'].astype('category')
train = pd.get_dummies(train)
test = pd.get_dummies(test)
print(train.shape)
train.head()
print(test.shape)
test.head()
X = train.drop(['Survived'], axis = 1)
y = train['Survived']
rfr = RandomForestClassifier(random_state=42)

rfr.fit(X, y)

rfr_pred = rfr.predict(X)

rfr_acc = accuracy_score(y, rfr_pred)

print('Accuracy on the whole train set: ',round(rfr_acc*100,2,),'%')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state=42)
rfr = RandomForestClassifier(random_state=42)

rfr.fit(X_train, y_train)

rfr_pred_train = rfr.predict(X)
rfr_pred_test = rfr.predict(X_test)

rfr_acc_train = accuracy_score(y, rfr_pred_train)
rfr_acc_test = accuracy_score(y_test, rfr_pred_test)

print('Accuracy on the train set: ',round(rfr_acc_train*100,2,),'%')
print('Accuracy on the test set: \t',round(rfr_acc_test*100,2,),'%')
# param_grid = { "criterion" : ["gini", "entropy"],
#               "min_samples_leaf" : [1, 5, 10, 25, 50, 70],
#               "min_samples_split" : [2, 4, 10, 12, 16, 18, 25, 35],
#               "n_estimators": [100, 400, 700, 1000, 1500]}

# from sklearn.model_selection import GridSearchCV, cross_val_score

# rf = RandomForestClassifier(n_estimators=100, max_features='auto', oob_score=True, random_state=1, n_jobs=-1)

# clf = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=-1,verbose=10)
# clf.fit(X_train, y_train)

# clf.best_params_
rfr = RandomForestClassifier(criterion = "entropy", 
                              min_samples_leaf = 5,
                              min_samples_split = 12,
                              n_estimators=400,
                              random_state=42,
                              n_jobs=-1)
rfr.fit(X_train, y_train)

rfr_pred_train = rfr.predict(X_train)
rfr_pred_test = rfr.predict(X_test)

rfr_ac_train = accuracy_score(y_train, rfr_pred_train)
rfr_ac_test = accuracy_score(y_test, rfr_pred_test)

print('Accuracy on the train set: ',round(rfr_ac_train*100,2,),'%')
print('Accuracy on the test set: \t',round(rfr_ac_test*100,2,),'%')
rfr.fit(X, y)

rfr_pred = rfr.predict(test)

# test_df = pd.read_csv(file_path_test)

# submission = pd.DataFrame({"PassengerId": test_df["PassengerId"],
#                            "Survived": rfr_pred})

# submission.to_csv('submissionfinal', index=False)