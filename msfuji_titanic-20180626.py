import numpy as np
import pandas as pd
import scipy.stats
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

# load  data
train = pd.read_csv('../input/train.csv')
train.head()
y_train = train['Survived']
X_train = train[['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
y_train.head()
X_train.head()
test = pd.read_csv('../input/test.csv')
test.head()
ID_test = test['PassengerId']
X_test = test[['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
ID_test.head()
X_test.head()
tab = pd.crosstab(y_train, X_train['Pclass'])
tab.T.plot.bar()
chi2, p, dof, expected = scipy.stats.chi2_contingency(tab)
print("p-value = %g" % p)
tab = pd.crosstab(y_train, X_train['Sex'])
tab.T.plot.bar()
chi2, p, dof, expected = scipy.stats.chi2_contingency(tab)
print("p-value = %g" % p)
## Age
train.boxplot(column='Age', by='Survived')
age0 = X_train[y_train==0]['Age']
age1 = X_train[y_train==1]['Age']
statistic, pvalue = scipy.stats.ttest_ind(age0, age1, nan_policy='omit')
print("Mean age of non-survivor: %f" % age0.mean())
print("Mean age of survivor: %f" % age1.mean())
print("p-value = %g" % pvalue)
tab = pd.crosstab(y_train, X_train['SibSp'])
tab.T.plot.bar()
chi2, p, dof, expected = scipy.stats.chi2_contingency(tab)
print("p-value = %g" % p)
tab = pd.crosstab(y_train, X_train['Parch'])
tab.T.plot.bar()
chi2, p, dof, expected = scipy.stats.chi2_contingency(tab)
print("p-value = %g" % p)
train.boxplot(column='Fare', by='Survived')
fare0 = X_train[y_train==0]['Fare']
fare1 = X_train[y_train==1]['Fare']
statistic, pvalue = scipy.stats.ttest_ind(fare0, fare1, nan_policy='omit')
print("Mean fare of non-survivor: %f" % fare0.mean())
print("Mean fare of survivor: %f" % fare1.mean())
print("p-value = %g" % pvalue)
tab = pd.crosstab(y_train, X_train['Embarked'])
tab.T.plot.bar()
chi2, p, dof, expected = scipy.stats.chi2_contingency(tab)
print("p-value = %g" % p)
tab = pd.crosstab(y_train, pd.isna(X_train['Cabin']).map({True: "NaN", False: "Other"}))
tab.T.plot.bar()
chi2, p, dof, expected = scipy.stats.chi2_contingency(tab)
print("p-value = %g" % p)
X_train = X_train.assign(Cabin_initial = X_train['Cabin'].str.slice(0,1))
X_test = X_test.assign(Cabin_initial = X_test['Cabin'].str.slice(0,1))
X_train.head()
X_test.head()
tab = pd.crosstab(y_train, X_train['Cabin'].str.slice(0,1))
tab.T.plot.bar()
chi2, p, dof, expected = scipy.stats.chi2_contingency(tab)
print("p-value = %g" % p)
X_train = X_train.assign(Title = X_train['Name'].str.extract(', (.+?)\.', expand=True).iloc[:, 0])
X_test = X_test.assign(Title = X_test['Name'].str.extract(', (.+?)\.', expand=True).iloc[:, 0])
X_train.head()
X_test.head()
tab = pd.crosstab(y_train, X_train['Title'])
tab.T.plot.bar()
chi2, p, dof, expected = scipy.stats.chi2_contingency(tab)
print("p-value = %g" % p)
X_train = X_train.drop(['Name', 'Ticket', 'Cabin'], axis=1)
X_test = X_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
def text2numeric(df):
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    df['Title'] = df['Title'].map({'Mr': 0, 'Miss': 2, 'Mrs': 3, 'Master': 1})
    df['Cabin_initial'] = df['Cabin_initial'].map({np.nan: 0, 'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7})
    return(df)

X_train = text2numeric(X_train)
X_test = text2numeric(X_test)
X_train.head()
X_test.head()
rfc =RandomForestClassifier(n_estimators=50)
svc = SVC(C=1)
xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)

pip = make_pipeline(
    preprocessing.Imputer(strategy="mean"),
    preprocessing.OneHotEncoder(categorical_features=[0]),
    VotingClassifier(estimators=[('rfc', rfc), ('svc', svc), ('xgb', xgb)])
    )

scores = cross_val_score(pip, X_train, y_train, scoring='accuracy', cv=5)
print(np.mean(scores))
pip.fit(X_train, y_train)
y_test = pip.predict(X_test)
sub = pd.DataFrame({"PassengerId": ID_test, "Survived": y_test})
sub.head()
sub.to_csv("submission.tsv", index=False)