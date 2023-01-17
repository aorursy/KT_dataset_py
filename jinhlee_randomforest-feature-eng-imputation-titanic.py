# Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

style.use('fivethirtyeight')

%matplotlib inline



# Load the dataset

train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")



# Print the dataset information

train_data.info()
### Select features that only make sense for survival prediction, SibSp and Parch are combined into Family

train_df = train_data[['Name','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived']]

family = (train_df.SibSp + train_df.Parch).rename('Family')

train_df = train_df.drop(['SibSp','Parch'],axis=1)

train_df = train_df.join(family)
### Define a fundtion for visualization of features against survival count

def visualize(key, ax):

    plt.sca(ax)

    if key=='Age': ## In case of Age feature, many levels are grouped into bins

        bins = [0,15,30,45,60,90]

        grouped = train_df.groupby([pd.cut(train_df.Age, bins),'Survived'])['Survived'].count()

    elif key=='Fare': ## in case of Fare feature, many levels are grouped into bins

        bins = np.append(np.arange(0,90,10),np.array([100,1000]))

        grouped = train_df.groupby([pd.cut(train_df.Fare, bins),'Survived'])['Survived'].count()

    else: grouped = train_df.groupby([key,'Survived'])['Survived'].count()

    

    barwidth, offset = 0.3, 0.5

    for ii in range(int(len(grouped)/2)):

        not_sur = plt.bar(ii+offset-barwidth, grouped.iloc[ii*2], width=barwidth, color='grey', alpha=1)

        sur = plt.bar(ii+offset, grouped.iloc[ii*2+1], width=barwidth, color='lightblue', alpha=1)

    

    ### Plot parameters

    xticks = np.arange(len(grouped)/2)+offset

    labels = grouped.index.get_level_values(level=0)[::2]

    rotation = 30 if (key=='Age')|(key=='Fare') else 0

    plt.xticks(xticks, labels, rotation=rotation)

    plt.xlabel(key)

    plt.ylabel('Count')

    plt.legend((not_sur[0],sur[0]),('Not Survived','Survived'),fontsize='small')

    plt.tight_layout()
### Display plots

fig, axes = plt.subplots(3,2, figsize=(12,12))



visualize('Pclass',axes[0,0])

visualize('Fare',axes[0,1])

visualize('Age',axes[1,0])

visualize('Sex',axes[1,1])

visualize('Family',axes[2,0])

visualize('Embarked',axes[2,1])
### train_df dataset "without" NaN values

train_short = train_df.copy().dropna(axis=0)

features_short = train_short[['Pclass','Fare','Age','Sex','Family','Embarked']]

target_short = train_short['Survived']
### For RandomForestClassifier, turn the column Sex into numeric values (0,1) and the column Embarked into 3 (C,Q,S) binary columns

from sklearn.preprocessing import LabelEncoder



#features_short.loc[:,'Sex'] = LabelEncoder().fit_transform(features_short['Sex'])

features_short['Sex_d'] = (features_short.Sex=='male').astype(int)

features_short = pd.concat([features_short, pd.get_dummies(features_short.Embarked, prefix='Emb')], axis=1)

features_short = features_short.drop(['Sex','Embarked'], axis=1)
### Prediction

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



### Prediction function (I'll call it several times)

def RFPred(features, target):

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.4, random_state=123)

    clf = RandomForestClassifier(n_estimators=80) #min_samples_leaf=2,min_samples_split=3, random_state=123)

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    print("features : {}".format(features.columns.tolist()))

    print("feature_importance : {}".format(clf.feature_importances_))

    print("score = {}".format(accuracy_score(pred, y_test)))

    return clf, X_train, X_test, y_train, y_test
clf, X_train, X_test, y_train, y_test = RFPred(features_short, target_short)
### Prediction by AdaBoost just for simple comparaison

from sklearn.ensemble import AdaBoostClassifier



abc = AdaBoostClassifier()

abc.fit(X_train, y_train)

pred = abc.predict(X_test)

print("features : {}".format(features_short.columns.tolist()))

print("feature importance : {}".format(abc.feature_importances_))

print("score = {}".format(accuracy_score(pred, y_test)))
### train_df dataset "with" NaN values

train_long = train_df.copy()

features_long = train_long[['Name','Pclass','Fare','Age','Sex','Family','Embarked']]

target_long = train_long['Survived']
### Display Age distribution by Sex and Title

fig, axes = plt.subplots(2,2,figsize=(10,5),sharey=True,sharex=True)



master = features_long[features_long.Name.str.

                          contains('Master\.')].Age.round(0).dropna()

axes[0,0].hist(master, bins=np.arange(0,90,1))

mr = features_long[features_long.Name.str.

                          contains(r'Mr\.|Dr\.|Rev\.|Major\.|Col\.|Capt\.|Don\.|Jonkheer',regex=True)].Age.round(0).dropna()

axes[0,1].hist(mr, bins=np.arange(0,90,1))

miss = features_long[features_long.Name.str.

                            contains(r'Miss\.|Mlle\.',regex=True)].Age.round(0).dropna()

axes[1,0].hist(miss, bins=np.arange(0,90,1), color='r')

mrs = features_long[features_long.Name.str.

                            contains(r'Mrs\.|Mme\.|Ms\.|Countess\.',regex=True)].Age.round(0).dropna()

axes[1,1].hist(mrs, bins=np.arange(0,90,1), color='r')



plt.suptitle('Age count for Master / Mr and similar / Miss / Mrs and similar (blue for men, red for women)')

fig.text(0.5, -0.01, 'Age', ha='center')

fig.text(-0.01, 0.5, 'Count', va='center', rotation='vertical')

plt.tight_layout()
### Impute NaN values in age according to their categories above - random selection from the existing distribution

age  = features_long.Age

name = features_long.Name

np.random.seed(123)



idx1 = features_long[(features_long.Age.isnull())&(features_long.Name.str.contains('Master\.'))].index

for ii in range(len(idx1)):

    features_long.set_value(idx1[ii], 'Age', master.iloc[np.random.randint(len(master))])



idx2 = features_long[(features_long.Age.isnull())&(features_long.Name.str.contains(r'Mr\.|Dr\.|Rev\.|Major\.|Col\.|Capt\.|Don\.|Jonkheer',regex=True))].index

for ii in range(len(idx2)):

    features_long.set_value(idx2[ii], 'Age', mr.iloc[np.random.randint(len(mr))])



idx3 = features_long[(features_long.Age.isnull())&(features_long.Name.str.contains(r'Miss\.|Mlle\.',regex=True))].index

for ii in range(len(idx3)):

    features_long.set_value(idx3[ii], 'Age', miss.iloc[np.random.randint(len(miss))])



idx4 = features_long[(features_long.Age.isnull())&(features_long.Name.str.contains(r'Mrs\.|Mme\.|Ms\.|Dr\.|Countess\.',regex=True))].index

for ii in range(len(idx4)):

    features_long.set_value(idx4[ii], 'Age', mrs.iloc[np.random.randint(len(mrs))])
### Age data original distribution

fig, axes = plt.subplots(1,2, figsize=(12,4), sharey=True)

age = train_df.Age.round(0)

grouped = age.groupby(age).count()

plt.sca(axes[0])

plt.bar(grouped.index,grouped,color='grey')

plt.xlabel('Age')

plt.ylabel('Count')

plt.title('Age count without imputation')



### Imputed Age dataset (new values added shown in orange)

age_imp = features_long.Age.round(0)

grouped_imp = age_imp.groupby(age_imp).count()

plt.sca(axes[1])

plt.bar(grouped_imp.index, grouped_imp, color='orange')

plt.bar(grouped.index, grouped, color='grey')

plt.xlabel('Age')

#plt.ylabel('Count')

plt.title('Age count with imputation')

plt.tight_layout()
### In the same way, impute Embarked column

emb = features_long.Embarked.copy()

num = emb.isnull().sum()

idx = np.random.randint(len(emb)-num, size=num)

emb[emb.isnull()] = emb.dropna().iloc[idx].tolist()

features_long = pd.concat([features_long.drop('Embarked', axis=1), emb], axis=1)
### Sex column into numeric values

#features_long.loc[:,'Sex'] = LabelEncoder().fit_transform(features_long['Sex'])

features_long['Sex_d'] = (features_long.Sex=='male').astype(int)

features_long.drop('Sex',axis=1, inplace=True)
### We saw earlier that Embarked doesn't help much so delete it with Name

features_long.drop(['Name','Embarked'], axis=1, inplace=True)
_ = RFPred(features_long, target_long)
"""features_long['S_Family'] = ((features_long.Family>0)&(features_long.Family<4)).astype(int)

#features_long['Solo'] = (features_long.Family==0).astype(int)

#features_long['B_Family'] = (features_long.Family>=4).astype(int)

features_long.drop(['Family'],axis=1,inplace=True)"""
"""features_long = pd.concat([features_long, pd.get_dummies(features_long.Pclass, prefix='Pclass')], axis=1)

features_long = features_long.drop(['Pclass','Pclass_3'], axis=1)"""
"""features_long['maleadult'] = ((features_long.Age>15)&(features_long.Sex_d==1)).astype(int)

#features_long['femalesenior'] = ((features_long.Age>30)&(features_long.Sex_d==0)).astype(int)

#features_long['malechild'] = ((features_long.Age<=15)&(features_long.Sex_d==1)).astype(int)"""
clf_opt, X_train, X_test, y_train, y_test = RFPred(features_long, target_long)
### Prediction optimization using GridSearchCV

from sklearn.model_selection import GridSearchCV



grid_param = {'n_estimators': [10,20,40,80,100]}

#             'min_samples_split': [2,4,6],

#             'min_samples_leaf': [1,2,3],

#             'criterion': ['gini'],

#             'random_state': [0]}

grid_search = GridSearchCV(clf, grid_param)

grid_search.fit(X_train, y_train)

pred = grid_search.predict(X_test)

clf_opt = grid_search.best_estimator_

print("best parameters : {}".format(grid_search.best_estimator_))

print("score = {}".format(accuracy_score(pred, y_test)))
### K-Fold cross validation to check variance of results

from sklearn.model_selection import KFold, cross_val_score



kfold = KFold(n_splits=5)

cross_val_score(clf_opt, X_train, y_train, cv=kfold, n_jobs=-1)
### Prediction by AdaBoost just for simple comparaison

from sklearn.ensemble import AdaBoostClassifier



abc = AdaBoostClassifier()

abc.fit(X_train, y_train)

pred = abc.predict(X_test)

print("features : {}".format(features_long.columns.tolist()))

print("feature importance : {}".format(abc.feature_importances_))

print("score = {}".format(accuracy_score(pred, y_test)))
### For submission - prediction without Age

nage = features_long.drop('Age',axis=1)

clf_opt_nage, X_train, X_test, y_train, y_test = RFPred(nage, target_long)
### Prediction optimization using GridSearchCV

grid_param = {'n_estimators': [10,20,40,80,100]}

#             'min_samples_split': [2,4,6],

#             'min_samples_leaf': [1,2,3],

#             'criterion': ['gini'],

#             'random_state': [0]}

grid_search = GridSearchCV(clf, grid_param)

grid_search.fit(X_train, y_train)

pred = grid_search.predict(X_test)

clf_opt_nage = grid_search.best_estimator_

print("best parameters : {}".format(grid_search.best_estimator_))

print("score = {}".format(accuracy_score(pred, y_test)))
### For submission - prediction without Fare

nfare = features_long.drop('Fare',axis=1)

clf_opt_nfare, X_train, X_test, y_train, y_test = RFPred(nfare, target_long)
### Prediction optimization using GridSearchCV

grid_param = {'n_estimators': [10,20,40,80,100]}

#             'min_samples_split': [2,4,6],

#             'min_samples_leaf': [1,2,3],

#             'criterion': ['gini'],

#             'random_state': [0]}

grid_search = GridSearchCV(clf, grid_param)

grid_search.fit(X_train, y_train)

pred = grid_search.predict(X_test)

clf_opt_nfare = grid_search.best_estimator_

print("best parameters : {}".format(grid_search.best_estimator_))

print("score = {}".format(accuracy_score(pred, y_test)))
test_df = test_data[['PassengerId','Name','Fare','Sex','Age','SibSp','Parch','Pclass']]

family = (test_df.SibSp + test_df.Parch).rename('Family')

test_df = test_df.join(family)

test_df['S_Family'] = ((test_df.Family>0)&(test_df.Family<4)).astype(int)

test_df = pd.concat([test_df, pd.get_dummies(test_df.Pclass, prefix='Pclass')], axis=1)

test_df['Sex_d'] = (test_df.Sex=='male').astype(int)

test_df['maleadult'] = ((test_df.Age>15)&(test_df.Sex_d==1)).astype(int)

test_df = test_df.drop(['Pclass_3','SibSp','Parch','Sex'], axis=1) #'Family','Pclass',
test_df_age = test_df[(test_df.Age.notnull())&(test_df.Fare.notnull())]

test_df_nage = test_df[test_df.Age.isnull()]

test_df_nfare = test_df[test_df.Fare.isnull()]
passengerid_age = test_df_age.PassengerId

X_test_age = test_df_age[['Pclass', 'Fare', 'Age', 'Family', 'Sex_d']]

pred_age = clf_opt.predict(X_test_age)



passengerid_nage = test_df_nage.PassengerId

X_test_nage = test_df_nage[['Pclass', 'Fare', 'Family', 'Sex_d']]

pred_nage = clf_opt_nage.predict(X_test_nage)



passengerid_nfare = test_df_nfare.PassengerId

X_test_nfare = test_df_nfare[['Pclass', 'Age', 'Family', 'Sex_d']]

pred_nfare = clf_opt_nfare.predict(X_test_nfare)
y_test_age = pd.concat([passengerid_age.reset_index().PassengerId, pd.Series(pred_age)], axis=1)

y_test_nage = pd.concat([passengerid_nage.reset_index().PassengerId, pd.Series(pred_nage)], axis=1)

y_test_nfare = pd.concat([passengerid_nfare.reset_index().PassengerId, pd.Series(pred_nfare)], axis=1)

y_test = pd.concat([y_test_age, y_test_nage, y_test_nfare]).sort_values('PassengerId').reset_index()

y_test.columns = ['index','PassengerId','Survived']

submission = y_test[['PassengerId','Survived']]
submission.to_csv('titanic_jpark_basic.csv', index=False)