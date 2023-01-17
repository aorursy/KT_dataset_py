import pandas as pd
import numpy as np
import re as re
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
pd.options.mode.chained_assignment = None
from sklearn.model_selection import KFold
data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')
data_train.head(5)
y_train = data_train.iloc[:, 1]
data_test.head(5)
data_train.isnull().sum()
data_test.isnull().sum()
all_data = pd.concat([data_train, data_test], axis=0)
all_data.isnull().sum()
all_data[pd.isnull(all_data['Embarked'])]
all_data[(all_data['Pclass'] == 1) & (all_data['Fare'] > 70) & (all_data['Fare'] < 90)]
all_data.iloc[:, 2] = all_data.iloc[:, 2].fillna(all_data.mode(0)['Embarked'][0])
all_data.isnull().sum()
all_data[pd.isnull(all_data['Fare'])]
third_class_pass = all_data[(all_data['Pclass'] == 3)]
all_data.iloc[:, 3] = all_data.iloc[:, 3].fillna(third_class_pass.mean(0)['Fare'])
all_data.isnull().sum()
all_data[pd.isnull(all_data['Age'])]
all_data.groupby(['Pclass'], as_index=False)['Age'].mean()
all_data.groupby(['Pclass', 'Sex'], as_index=False)['Age'].mean()
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

# create new column
all_data['Title'] = [get_title(i) for i in all_data['Name']]

pd.crosstab(all_data['Title'], all_data['Sex'])
all_data['Married'] = [1 if i in ['Mrs', 'Countess', 'Mme', 'Dona'] else 0 for i in all_data['Title']]

all_data.groupby(['Pclass', 'Sex', 'Married'], as_index=False)['Age'].mean()
all_data.groupby(['Pclass', 'Sex', 'Married'], as_index=False)['Age'].median()
for i in range(len(all_data)):
    # if age is null
    if pd.isnull(all_data.iloc[i, 0]):
        # if passenger male
        if all_data.iloc[i, 8] == 'male':
            # age estimate based on passenger class
            all_data.iloc[i, 0] = {1: 42, 2: 29.5, 3:25}[all_data.iloc[i, 7]]
        else:
            # if woman not married
            if all_data.iloc[i, 13] == 0:
                # age estimate based on passenger class
                all_data.iloc[i, 0] = {1: 30, 2: 20, 3:18}[all_data.iloc[i, 7]]
            else:
                all_data.iloc[i, 0] = {1: 45, 2: 30.5, 3:31}[all_data.iloc[i, 7]]
all_data.isnull().sum()
X_train = all_data[:891]
X_test = all_data[891:]
data_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()
data_train[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean()
fig,ax = plt.subplots(1)
plt.plot(data_train['PassengerId'], data_train['Fare'], 'r.')
plt.title('Ticket fares (training data)')
ax.set_ylabel('Ticket fare (£)')
ax.set_xlabel('Passenger Id')
plt.show()
data_train_no_zero_fares = data_train[data_train['Fare'] != 0]
data_train_no_zero_fares[["Pclass", "Fare"]].groupby(['Pclass'], as_index=False).min()
data_train_no_zero_fares[["Pclass", "Fare"]].groupby(['Pclass'], as_index=False).max()
low_fares = data_train[data_train['Fare'] < 100]
fig,ax = plt.subplots(1)
plt.plot(low_fares['PassengerId'], low_fares['Fare'], 'r.')
plt.title('Ticket fares (training data)')
ax.set_ylabel('Ticket fare (£)')
ax.set_xlabel('Passenger Id')
plt.show()
def createFaresRangeColumn(X):
    conditions = [
        (X['Fare'] > 300),
        (X['Fare'] > 100),
        (X['Fare'] > 30)
    ]
    choices = [0, 1, 2]
    X['FaresRange'] = np.select(conditions, choices, default=3)
    return X

X_train = createFaresRangeColumn(X_train)   
X_test = createFaresRangeColumn(X_test)
X_train[["FaresRange", "Survived"]].groupby(['FaresRange'], as_index=False).mean()
fig,ax = plt.subplots(1)
plt.plot(data_train['PassengerId'], data_train['Age'], 'b.')
plt.title('Age (training data)')
ax.set_ylabel('Age (years)')
ax.set_xlabel('Passenger Id')
plt.show()
def createAgeRangeColumn(X):
    conditions = [
        (X['Age'] > 35),
        (X['Age'] > 15),
    ]
    choices = [2, 1]
    X['AgeRange'] = np.select(conditions, choices, default=0)
    return X

X_train = createAgeRangeColumn(X_train)   
X_test = createAgeRangeColumn(X_test)
X_train[["AgeRange", "Survived"]].groupby(['AgeRange'], as_index=False).mean()
X_train[["AgeRange", "Pclass"]].groupby(['AgeRange'], as_index=False).mean()
X_train[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean()
X_train[['Embarked', 'Pclass']].groupby(['Embarked'], as_index=False).mean()
X_train['FamilySize'] = X_train['SibSp'] + X_train['Parch'] + 1
X_test['FamilySize'] = X_test['SibSp'] + X_test['Parch'] + 1
fig,ax = plt.subplots(1)
plt.plot(X_train['PassengerId'], X_train['FamilySize'], 'g.')
plt.title('Family size (training data)')
ax.set_ylabel('Family size (no. people)')
ax.set_xlabel('Passenger Id')
plt.show()
X_train['TraveledAlone'] = (X_train['FamilySize'] == 1).astype(int)
X_test['TraveledAlone'] = (X_test['FamilySize'] == 1).astype(int)
X_train[['TraveledAlone', 'Survived']].groupby(['TraveledAlone'], as_index=False).mean()
X_train[['TraveledAlone', 'Pclass']].groupby(['TraveledAlone'], as_index=False).mean()
X_train[['TraveledAlone', 'FaresRange']].groupby(['TraveledAlone'], as_index=False).mean()
X_train_features = X_train[['Sex', 'Pclass', 'Fare', 'Age', 'TraveledAlone']]
X_test_features = X_test[['Sex', 'Pclass', 'Fare', 'Age', 'TraveledAlone']]
X_train_features.head(5)
labelencoder = LabelEncoder()
X_train_features.loc[:, 'Sex'] = labelencoder.fit_transform(X_train_features.loc[:, 'Sex'])
X_test_features.loc[:, 'Sex'] = labelencoder.transform(X_test_features.loc[:, 'Sex'])
X_train_features.head(5)
scaler = StandardScaler()
X_train_features[['Fare', 'Age']] = scaler.fit_transform(X_train_features[['Fare', 'Age']])
X_test_features[['Fare', 'Age']] = scaler.fit_transform(X_test_features[['Fare', 'Age']])
X_train_features.head(5)
onehotencoder = OneHotEncoder(categorical_features = [1])
X_train_enc = onehotencoder.fit_transform(X_train_features).toarray()
X_test_enc = onehotencoder.transform(X_test_features).toarray()
print(X_train_enc[0, :])
def kfold_assessment(clf):
    k_fold = KFold(5)
    for k, (train, val) in enumerate(k_fold.split(X_train_enc, y_train)):
        clf.fit(X_train_enc[train], y_train[train])
        print("[fold {0}],  score: {1:.5f}".format(k, clf.score(X_train_enc[val], y_train[val])))

kn_clf = KNeighborsClassifier(n_neighbors=3)
kfold_assessment(kn_clf)
lr_clf = LogisticRegression(solver="lbfgs", C=10)
kfold_assessment(lr_clf)
gnb_clf = GaussianNB()
kfold_assessment(gnb_clf)
svm_clf = LinearSVC(C=1)
kfold_assessment(svm_clf)
rf_clf = RandomForestClassifier(max_depth=6, n_estimators=100, n_jobs=-1)
kfold_assessment(rf_clf)
ab_clf = AdaBoostClassifier()
kfold_assessment(ab_clf)
gb_clf = GradientBoostingClassifier()
kfold_assessment(gb_clf)
classifiers = [
    ('KNeighbors', KNeighborsClassifier(n_neighbors=3)), 
    ('LogisticRegression', LogisticRegression(solver="lbfgs", C=10)), 
    ('GaussianNB', GaussianNB()),
    ('SupportVectorMachine', SVC(probability=True)),
    ('Random Forest', RandomForestClassifier(max_depth=4, n_estimators=100, n_jobs=-1)), 
    ('AdaBoost', AdaBoostClassifier()), 
    ('GradientBoosting', GradientBoostingClassifier())
]
vc = VotingClassifier(estimators=classifiers, voting='soft')
vc = vc.fit(X_train_features, y_train)

preds = vc.predict(X_test_features)
submission = pd.DataFrame({
        "PassengerId": data_test["PassengerId"],
        "Survived": preds
    })
submission.to_csv('titanic.csv', index=False)