%matplotlib inline
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
train = pd.read_csv("../input/train.csv", index_col="PassengerId")
train.head()
sns.pairplot(train.dropna(how='any'), hue='Survived')
train['Title'] = train.Name.apply(lambda x: x[x.find(',')+2:x.find('.')])
del train['Name'] # Remove the original name, which is no longer useful
train.loc[train.Cabin.str.len() > 5, 'Cabin'] # Checking whether a passenger can have multiple rooms in different cabins
# Add the cabin letter as a factor, NaN if no data is available
train['Cabin'] = train['Cabin'].fillna(value=' ')
train['Cabin'] = train.Cabin.map(lambda x: x[0]).replace(' ', np.nan)
del train['Ticket']
train.head()
train['Survived'].value_counts()
sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train)
plt.show()
train.groupby('Survived').mean()
train.head()
train.groupby('Sex').mean()
train.groupby('Embarked').mean()
train.groupby(['Embarked', 'Sex']).count()
train.groupby('Cabin').mean()
train.groupby(['Cabin', 'Sex']).count()
train.groupby('Title').mean()
pd.crosstab(train.Sex, train.Survived).plot(kind='bar')
table = pd.crosstab(train.Embarked, train.Survived)
table.plot(kind='bar', stacked=True)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Port of Embarkment Vs Survival')
plt.ylabel('Proportion of Passengers')
table_cabin = pd.crosstab(train.Cabin, train.Survived)
table_cabin.plot(kind='bar', stacked=True)

table_cabin.div(table_cabin.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
table_title = pd.crosstab(train.Title, train.Survived)
table_title.plot(kind='bar', stacked=True)
table_title.div(table_title.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
sns.distplot(train.Age.dropna(), rug=True)
train.head()
cat_vars=['Pclass', 'Sex', 'Cabin', 'Embarked', 'Title']
train_dummy = train.copy()
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(train[var], prefix=var)
    train_dummy=train_dummy.join(cat_list)
    
train_dummy.drop(columns=cat_vars, inplace=True)
train.Title.value_counts()
bad_title = ['Dr', 'Rev', 'Col', 'Mlle', 'Major', 'Mme', 'Ms', 'Capt', 'the Countess', 'Don', 'Lady', 'Sir', 'Jonkheer']
bad_title = ['Title_' + title for title in bad_title]
train_dummy.drop(columns=bad_title, inplace=True)
response = ['Survived']
predictor = [x for x in train_dummy.columns if x != 'Survived']
#imputing
from sklearn.preprocessing import Imputer
my_imputer = Imputer()

age_imputed = my_imputer.fit_transform(train_dummy.Age.values.reshape(-1,1))
train_dummy['Age'] = age_imputed
train_dummy.head()
from sklearn import datasets
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
rfe = RFE(logreg, n_features_to_select=15)
rfe = rfe.fit(train_dummy[predictor], train_dummy[response].as_matrix().ravel())
print(rfe.support_)
print(rfe.ranking_)
rfe_cols = [x for x,y in zip(predictor, rfe.support_) if y==True]
print(rfe_cols)
predictor
import statsmodels.api as sm
from scipy import stats
#d from a weird error
#from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)


logit_model=sm.Logit(train_dummy[response], train_dummy[predictor])
result=logit_model.fit()
result.summary()
X_train, X_test, y_train, y_test = train_test_split(train_dummy[predictor].as_matrix(), np.ravel(train_dummy[response]), test_size=0.3, random_state=0)
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, train_dummy[predictor].as_matrix(), train_dummy[response].as_matrix().ravel(), cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
test = pd.read_csv("../input/test.csv", index_col="PassengerId")
# Add the cabin letter as a factor, NaN if no data is available
test['Cabin'] = test['Cabin'].fillna(value=' ')
test['Cabin'] = test.Cabin.map(lambda x: x[0]).replace(' ', np.nan)
test['Title'] = test.Name.apply(lambda x: x[x.find(',')+2:x.find('.')])
del test['Name'] # Remove the original name, which is no longer useful
del test['Ticket']
cat_vars=['Pclass', 'Sex', 'Cabin', 'Embarked', 'Title']
test_dummy = test.copy()
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(test[var], prefix=var)
    test_dummy = test_dummy.join(cat_list)
    
test_dummy.drop(columns=cat_vars, inplace=True)
test.Title.value_counts()
bad_title = ['Col', 'Rev', 'Dr', 'Ms', 'Dona']
bad_title = ['Title_' + title for title in bad_title]
test_dummy.drop(columns=bad_title, inplace=True)
my_imputer = Imputer()

age_imputed = my_imputer.fit_transform(test_dummy.Age.values.reshape(-1,1))
test_dummy['Age'] = age_imputed

# 1 value needs fare imputing for the test set
fare_imputed = my_imputer.fit_transform(test_dummy.Fare.values.reshape(-1,1))
test_dummy['Fare'] = fare_imputed
train_dummy.drop(columns='Survived').columns
test_dummy.columns
test_dummy['Cabin_T'] = 0
test_dummy = test_dummy[train_dummy.drop(columns='Survived').columns.tolist()]

X_train = train_dummy[predictor].as_matrix()
X_test = test_dummy.as_matrix()
y_train = np.ravel(train_dummy[response])

from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
# Predictions
y_pred = logreg.predict(X_test)
#format for writing to csv
submission = pd.DataFrame(data={'PassengerId': test_dummy.index.values, 'Survived': y_pred})
submission.head()
submission.to_csv("solution_logit.csv", index=False)
