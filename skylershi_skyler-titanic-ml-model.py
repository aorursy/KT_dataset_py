import numpy as np

import pandas as pd

from pandas_profiling import ProfileReport

import seaborn as sns

import matplotlib.pyplot as plt

sns.set()





train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df  = pd.read_csv('/kaggle/input/titanic/test.csv')

combine_data = [train_df, test_df]
train_df.head()
test_df.head()
train_df.describe()
profile = ProfileReport(train_df, title='Titanic Data Profiling Report', html={'style':{'full_width':True}})
profile.to_widgets()
fig, axs = plt.subplots(3,3, figsize = (15,15))

sns.countplot(train_df.Survived, ax = axs[0][0])

sns.countplot(train_df.Pclass, ax = axs[0][1])

sns.countplot(train_df.Sex, ax = axs[0][2])

sns.distplot(train_df.Age, ax = axs[1][0])

sns.countplot(train_df.SibSp, ax = axs[1][1])

sns.countplot(train_df.Parch, ax = axs[1][2])

sns.distplot(train_df.Fare, ax = axs[2][0])

sns.countplot(train_df.Cabin, ax = axs[2][1], order=train_df.Cabin.value_counts().iloc[:10].index)

sns.countplot(train_df.Embarked, ax = axs[2][2])
# map non-null values to integer values for seaborn's regplot api.

# We throwaway the nulls for now for quick visualization. Decisions will be made later as to how to deal with the nulls.

reg_viz_df = train_df.copy()

reg_viz_df['Sex'] = reg_viz_df['Sex'][~ reg_viz_df.Sex.isna()].map( {'female': 1, 'male': 0} ).astype(int)

reg_viz_df['Embarked'] = reg_viz_df['Embarked'][~ reg_viz_df.Embarked.isna()].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
fig, axs = plt.subplots(3,3, figsize = (15,15))

sns.regplot('Pclass', 'Survived', data = reg_viz_df, ax = axs[0][0])

sns.regplot('Sex', 'Survived', data = reg_viz_df, ax = axs[0][1])

sns.regplot('Age', 'Survived', data = reg_viz_df, ax = axs[0][2])

sns.regplot('SibSp', 'Survived', data = reg_viz_df, ax = axs[1][0])

sns.regplot('Parch', 'Survived', data = reg_viz_df, ax = axs[1][1])

sns.regplot('Fare', 'Survived', data = reg_viz_df, ax = axs[1][2])

sns.regplot('Embarked', 'Survived', data = reg_viz_df, ax = axs[2][0])
sns.pairplot(reg_viz_df[reg_viz_df.columns.difference(["PassengerId", "Survived"])])
clean1 = train_df.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin'])
clean1[clean1['Sex'].isna()]
clean1['Sex'] = clean1['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
clean2 = clean1.copy()
clean2[clean2['Age'].isna()].head()
sns.distplot(train_df['Age'])
age_mean = clean2.Age.mean()

age_std  = clean2.Age.std()

age_nulls = clean2[clean2.Age.isna()].shape[0]

age_null_replacements = np.random.randint(age_mean - age_std, age_mean + age_std, size = age_nulls)





clean2.loc[clean2.Age.isna(),'Age'] = age_null_replacements
clean3 = clean2.copy()



clean3['FamilySize'] = clean3['SibSp'] + clean3['Parch'] + 1



clean3['IsAlone'] = 0

clean3.loc[clean3['FamilySize'] > 1, 'IsAlone'] = 1
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (18,6))

sns.regplot('FamilySize', 'Survived', data = clean3, ax = ax1)

sns.regplot('IsAlone', 'Survived', data = clean3, ax = ax2)
clean3 = clean3.drop(['SibSp', 'Parch', 'FamilySize'], axis = 1)
clean4 = clean3.copy()
sns.distplot(clean4['Fare'])
clean4['Fare'] = clean4['Fare'].fillna(clean4['Fare'].median())
clean5 = clean4.copy()
clean5['Embarked'] = clean5['Embarked'].fillna('S')

clean5['Embarked'] = clean5['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import GaussianNB



from sklearn.metrics import accuracy_score



from sklearn.model_selection import StratifiedShuffleSplit



classifiers = [

    LogisticRegression(),

    SVC(probability = True),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    RandomForestClassifier(),

    DecisionTreeClassifier(),

    KNeighborsClassifier(),

    GaussianNB(),

    Perceptron(),

    SGDClassifier()

]





# quick fix of data

train_baseline = train_df.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin'])

train_baseline['Sex'] = train_baseline['Sex'][~ train_baseline.Sex.isna()].map( {'female': 1, 'male': 0} ).astype(int)

train_baseline['Embarked'] = train_baseline['Embarked'][~ train_baseline.Embarked.isna()].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_baseline = train_baseline.fillna(train_baseline.Age.median())



X = train_baseline.iloc[:, 1:].values

y = train_baseline.iloc[:, 0].values



sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)



acc_dict = {}



for train_idx, test_idx in sss.split(X,y):

    X_train, X_test = X[train_idx], X[test_idx]

    y_train, y_test = y[train_idx], y[test_idx]

    

    for classifier in classifiers:

        name = classifier.__class__.__name__

        classifier.fit(X_train, y_train)

        predictions = classifier.predict(X_test)

        acc = accuracy_score(y_test, predictions)

        if name in acc_dict:

            acc_dict[name] += acc

        else:

            acc_dict[name] = acc



log_cols = ["Classifier", "Accuracy"]

log      = pd.DataFrame(columns=log_cols)





for clf in acc_dict:

    acc_dict[clf] = acc_dict[clf] / 10.0

    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)

    log = log.append(log_entry)



plt.xlabel('Accuracy')

plt.title('Classifier Accuracy')



sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
train_final
classifiers = [

    LogisticRegression(),

    SVC(probability = True),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    RandomForestClassifier(),

    DecisionTreeClassifier(),

    KNeighborsClassifier(),

    GaussianNB(),

    Perceptron(),

    SGDClassifier()

]



train_final = clean5.copy()



X = train_final.iloc[:, 1:].values

y = train_final.iloc[:, 0].values



sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)



acc_dict = {}



for train_idx, test_idx in sss.split(X,y):

    X_train, X_test = X[train_idx], X[test_idx]

    y_train, y_test = y[train_idx], y[test_idx]

    

    for classifier in classifiers:

        name = classifier.__class__.__name__

        classifier.fit(X_train, y_train)

        predictions = classifier.predict(X_test)

        acc = accuracy_score(y_test, predictions)

        if name in acc_dict:

            acc_dict[name] += acc

        else:

            acc_dict[name] = acc



log_cols = ["Classifier", "Accuracy"]

log      = pd.DataFrame(columns=log_cols)





for clf in acc_dict:

    acc_dict[clf] = acc_dict[clf] / 10.0

    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)

    log = log.append(log_entry)



plt.xlabel('Accuracy')

plt.title('Classifier Accuracy')



sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import roc_curve, auc
RF_estimator = RandomForestClassifier()





X = train_final.iloc[:, 1:].values

y = train_final.iloc[:, 0].values



sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=0)



acc = 0



conf_matrix = np.zeros((2,2))



for train_idx, test_idx in sss.split(X,y):

    X_train, X_test = X[train_idx], X[test_idx]

    y_train, y_test = y[train_idx], y[test_idx]

    

    RF_estimator.fit(X_train, y_train)

    predictions = RF_estimator.predict(X_test)

    

    acc += accuracy_score(y_test, predictions)

    conf_matrix += confusion_matrix(y_test, predictions)

    

print(acc / 10)

print(conf_matrix)

print(classification_report(y_test, predictions))
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit

from sklearn.ensemble import RandomForestClassifier

n_estimators = [140,145,150,155,160];

max_depth = range(1,10);

criterions = ['gini', 'entropy'];

cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)





parameters = {'n_estimators':n_estimators,

              'max_depth':max_depth,

              'criterion': criterions

              

        }

grid = GridSearchCV(estimator=RandomForestClassifier(max_features='auto'),

                                 param_grid=parameters,

                                 cv=cv,

                                 n_jobs = -1)

grid.fit(X,y)
print (grid.best_score_)

print (grid.best_params_)

print (grid.best_estimator_)
rf_grid = grid.best_estimator_

feature_importances = pd.DataFrame(rf_grid.feature_importances_,

                                   index = clean5.columns[1:],

                                    columns=['importance'])

feature_importances.sort_values(by='importance', ascending=False).head(10)
clean1 = test_df.drop(columns = ['Name', 'Ticket', 'Cabin'])

clean1['Sex'] = clean1['Sex'].map( {'female': 1, 'male': 0} ).astype(int)



clean2 = clean1.copy()

age_mean = clean2.Age.mean()

age_std  = clean2.Age.std()

age_nulls = clean2[clean2.Age.isna()].shape[0]

age_null_replacements = np.random.randint(age_mean - age_std, age_mean + age_std, size = age_nulls)

clean2.loc[clean2.Age.isna(),'Age'] = age_null_replacements



clean3 = clean2.copy()



clean3['FamilySize'] = clean3['SibSp'] + clean3['Parch'] + 1



clean3['IsAlone'] = 0

clean3.loc[clean3['FamilySize'] > 1, 'IsAlone'] = 1

clean3 = clean3.drop(['SibSp', 'Parch', 'FamilySize'], axis = 1)



clean4 = clean3.copy()

clean4['Fare'] = clean4['Fare'].fillna(clean4['Fare'].median())



clean5 = clean4.copy()

clean5['Embarked'] = clean5['Embarked'].fillna('S')

clean5['Embarked'] = clean5['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
X = clean5.iloc[:, 1:].values

predictions = rf_grid.predict(X)
submission = pd.DataFrame({

    "PassengerId": clean5['PassengerId'],

    "Survived": predictions

})



submission.PassengerId = submission.PassengerId.astype(int)

submission.Survived = submission.Survived.astype(int)

submission.to_csv("titanic1_submission.csv", index=False)