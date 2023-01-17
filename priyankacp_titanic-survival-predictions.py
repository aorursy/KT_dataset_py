import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import naive_bayes

from sklearn.svm import SVC



import os

print(os.listdir("../input"))
### Importing the input datasets



titanic_train = pd.read_csv("../input/train.csv",index_col='PassengerId')

titanic_test = pd.read_csv("../input/test.csv",index_col='PassengerId')

titanic_gender = pd.read_csv("../input/gender_submission.csv")
titanic_train.sample(n=5)
titanic_test.head()
# Example output dataset

titanic_gender.head()
titanic_train.info()



# Age, Cabin and Embarked have null values.

# Cabin has a lot of null values. After EDA, we can decide whether to drop this column.
titanic_test.info()



# Age, Cabin, Embarked and Fare have null values
# Same as above but represented in a heatmap



ax1 = plt.figure(figsize=(16,5))



ax1.add_subplot(121)

sns.heatmap(titanic_train.isnull(), yticklabels=False, cmap='viridis')

plt.title('Train data values')





ax1.add_subplot(122)

sns.heatmap(titanic_test.isnull(), yticklabels=False, cmap='viridis')

plt.title('Test data values')



plt.show()
ax2 = plt.figure(figsize=(16,5))



ax2.add_subplot(121)

sns.countplot(x='Survived', data=titanic_train, palette="magma", hue='Sex')



ax2.add_subplot(122)

sns.countplot(x='Survived', data=titanic_train, palette='magma', hue='Pclass')



plt.show()
sns.distplot(titanic_train['Age'].dropna(),bins=30, kde=False, color='darkviolet')

plt.show()
plt.figure(figsize=(16,5))

sns.countplot(x='Survived', hue='SibSp', data=titanic_train, palette='magma')

plt.show()
plt.figure(figsize=(16,5))

sns.countplot(x='Survived', hue='Embarked', data=titanic_train, palette='magma')

plt.show()
plt.figure(figsize=(12,8))

titanic_train.plot(x='Fare',y='Survived', kind='scatter')

plt.show()
titanic_train['Sex'] = titanic_train['Sex'].apply(lambda i: 0 if i == 'male' else 1)

titanic_test['Sex'] = titanic_test['Sex'].apply(lambda i: 0 if i == 'male' else 1)
titanic_train.head()
dummy_Embarked = pd.get_dummies(titanic_train.Embarked, prefix='Embarked')

dummy_Embarked.drop(dummy_Embarked.columns[0], axis=1, inplace=True)

titanic_train = pd.concat([titanic_train, dummy_Embarked], axis=1)

titanic_train.drop(['Embarked'],axis=1, inplace=True)
titanic_train.head()
dummy_Embarked = pd.get_dummies(titanic_test.Embarked, prefix='Embarked')

dummy_Embarked.drop(dummy_Embarked.columns[0], axis=1, inplace=True)

titanic_test = pd.concat([titanic_test, dummy_Embarked], axis=1)

titanic_test.drop(['Embarked'],axis=1, inplace=True)
titanic_test.head()
plt.figure(figsize=(12, 8))

sns.boxplot(x='Pclass',y='Age',data=titanic_train,palette='magma')
def age_null(cols):

    age = cols[0]

    pclass = cols[1]

    

    if pd.isnull(age):

        if pclass == 3:

            return titanic_train.Age[titanic_train['Pclass'] == 3].dropna().mean()

        elif pclass == 2:

            return titanic_train.Age[titanic_train['Pclass'] == 2].dropna().mean()

        else:

            return titanic_train.Age[titanic_train['Pclass'] == 1].dropna().mean()

    else:

        return age
titanic_train['Age'] = titanic_train[['Age','Pclass']].apply(age_null, axis=1)

titanic_test['Age'] = titanic_test[['Age','Pclass']].apply(age_null, axis=1)
# Same as above but represented in a heatmap



ax1 = plt.figure(figsize=(16,5))



ax1.add_subplot(121)

sns.heatmap(titanic_train.isnull(), yticklabels=False, cmap='viridis')

plt.title('Train data values')





ax1.add_subplot(122)

sns.heatmap(titanic_test.isnull(), yticklabels=False, cmap='viridis')

plt.title('Test data values')



plt.show()
feature_cols = ['Pclass','Sex','Age','SibSp','Embarked_Q','Embarked_S']
X_train, X_test, y_train, y_test = train_test_split(titanic_train[feature_cols], titanic_train.Survived)
logreg = LogisticRegression(solver='liblinear')

logreg.fit(X_train, y_train)

y_logreg = logreg.predict(X_test)
print (metrics.f1_score(y_test.values, y_logreg))

print (metrics.classification_report(y_test.values, y_logreg))
dectree = DecisionTreeClassifier(criterion='entropy', max_depth=3)

dectree.fit(X_train, y_train)

y_dectree = dectree.predict(X_test)
print (metrics.f1_score(y_test.values, y_dectree))

print (metrics.classification_report(y_test.values, y_dectree))
randfor = RandomForestClassifier(criterion="entropy", max_depth=5, n_estimators=100)

randfor.fit(X_train, y_train)

y_randfor = randfor.predict(X_test)
print (metrics.f1_score(y_test.values, y_randfor))

print (metrics.classification_report(y_test.values, y_randfor))
nbayes = naive_bayes.BernoulliNB()

nbayes.fit(X_train, y_train)

y_nbayes = nbayes.predict(X_test)
print (metrics.f1_score(y_test.values, y_nbayes))

print (metrics.classification_report(y_test.values, y_nbayes))
svecto = SVC(kernel='linear')

svecto.fit(X_train, y_train)

y_svecto = svecto.predict(X_test)
print (metrics.f1_score(y_test.values, y_svecto))

print (metrics.classification_report(y_test.values, y_svecto))
def addplot(y, pred):

    sns.heatmap(metrics.confusion_matrix(y,pred), annot=True, cmap='viridis', fmt="d",

            xticklabels=['Not Survived','Survived'],

            yticklabels=['Not Survived','Survived'])
fig1 = plt.figure(figsize=(14,15))



fig1.add_subplot(3,2,1)

addplot(y_test, y_logreg)

plt.title('Logistic Regression: F1 score - %0.2f'% metrics.f1_score(y_test.values, y_logreg))



fig1.add_subplot(3,2,2)

addplot(y_test, y_dectree)

plt.title('Decision Tree: F1 score - %0.2f'% metrics.f1_score(y_test.values, y_dectree))



fig1.add_subplot(3,2,3)

addplot(y_test, y_randfor)

plt.title('Random Forest: F1 score - %0.2f'% metrics.f1_score(y_test.values, y_randfor))



fig1.add_subplot(3,2,4)

addplot(y_test, y_nbayes)

plt.title('Naive Bayes: F1 score - %0.2f'% metrics.f1_score(y_test.values, y_nbayes))



fig1.add_subplot(3,2,5)

addplot(y_test, y_svecto)

plt.title('SVM: F1 score - %0.2f'% metrics.f1_score(y_test.values, y_svecto))



plt.show()
fpr_lg, tpr_lg, _ = metrics.roc_curve(y_test, y_logreg)

fpr_dt, tpr_dt, _ = metrics.roc_curve(y_test, y_dectree)

fpr_rf, tpr_rf, _ = metrics.roc_curve(y_test, y_randfor)

fpr_nb, tpr_nb, _ = metrics.roc_curve(y_test, y_nbayes)

fpr_sv, tpr_sv, _ = metrics.roc_curve(y_test, y_svecto)



auc_lg = metrics.auc(fpr_lg, tpr_lg)

auc_dt = metrics.auc(fpr_dt, tpr_dt)

auc_rf = metrics.auc(fpr_rf, tpr_rf)

auc_nb = metrics.auc(fpr_nb, tpr_nb)

auc_sv = metrics.auc(fpr_sv, tpr_sv)
plt.title('Receiver Operating Characteristic')



plt.plot(fpr_lg, tpr_lg, 'b', label='LG AUC = %0.3f'% auc_lg)

plt.plot(fpr_dt, tpr_dt, 'g', label='DT AUC = %0.3f'% auc_dt)

plt.plot(fpr_rf, tpr_rf, 'y', label='RF AUC = %0.3f'% auc_rf)

plt.plot(fpr_nb, tpr_nb, 'c', label='NB AUC = %0.3f'% auc_nb)

plt.plot(fpr_nb, tpr_nb, 'm', label='SV AUC = %0.3f'% auc_sv)



plt.legend(loc='lower right')



plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')



plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])



plt.show()
logreg = LogisticRegression(solver='liblinear')

logreg.fit(titanic_train[feature_cols], titanic_train.Survived)

y_pred = logreg.predict(titanic_test[feature_cols])



#randfor = RandomForestClassifier(criterion="entropy", max_depth=5, n_estimators=100)

#randfor.fit(titanic_train[feature_cols], titanic_train.Survived)

#y_pred = randfor.predict(titanic_test[feature_cols])
predictions = pd.concat([pd.DataFrame(titanic_test.index, columns=['PassengerId']), pd.DataFrame(y_pred, columns=['Survived'])],

           axis=1)
predictions.sample(n=5)
predictions.to_csv('predictions.csv', index=False)