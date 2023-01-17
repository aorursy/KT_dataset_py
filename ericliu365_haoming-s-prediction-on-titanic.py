import pandas as pd

import numpy as np

from sklearn.impute import SimpleImputer

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

from sklearn.feature_selection import mutual_info_classif

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import svm
train = pd.read_csv('../input/titanic/train.csv')

train.head()
test = pd.read_csv('../input/titanic/test.csv')

test.head()
gender = pd.read_csv('../input/titanic/gender_submission.csv')

gender.head()
train.isnull().sum()/train.isnull().count()
test.isnull().sum()/test.isnull().count()
imputer_num = SimpleImputer(missing_values=np.nan, strategy='mean')

train['Age'] = imputer_num.fit_transform(np.array(train['Age']).reshape(891,1))

test['Age'] = imputer_num.fit_transform(np.array(test['Age']).reshape(418,1))

test['Fare'] = imputer_num.fit_transform(np.array(test['Fare']).reshape(418,1))
imputer_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

train['Embarked'] = imputer_cat.fit_transform(np.array(train['Embarked']).reshape(891,1))

test['Embarked'] = imputer_cat.fit_transform(np.array(test['Embarked']).reshape(418,1))

train = train.drop(columns = ['Cabin'])

test = test.drop(columns = ['Cabin'])
train['Sex'] = train['Sex'].replace({'male': 0, 'female': 1})

test['Sex'] = test['Sex'].replace({'male': 0, 'female': 1})

train['Embarked'] = train['Embarked'].replace({'C': 0, 'Q': 1, 'S': 2})

test['Embarked'] = test['Embarked'].replace({'C': 0, 'Q': 1, 'S': 2})
np.any(train.isnull())
np.any(test.isnull())
sns.countplot(x = 'Survived', hue = 'Sex', data = train, palette='hls')

plt.show()
bins = pd.cut(train["Fare"], bins = np.arange(0,700,100))

fare_dist = train.groupby(bins)['Fare'].count()

Sur_dist = train.groupby(bins)['Survived'].sum()

Prob = Sur_dist/fare_dist

prob_df = pd.DataFrame(Prob, columns=["Survival probability"]).reset_index()

prob_df['Survival probability'] = prob_df['Survival probability'].fillna(0)

sns.barplot(x='Fare', y='Survival probability', data=prob_df)

plt.show()
sns.catplot(x = "Pclass", hue="Survived", data = train, kind="count")

plt.show()
selector = SelectKBest(score_func=f_classif, k=2)

selector_fit = selector.fit(np.array(train[['Age', 'Fare']]).reshape(891,2), np.array(train['Survived']).reshape(891,1))

print(selector_fit.scores_) #select Fare as one of the input column in the model
selectorcat = SelectKBest(score_func=mutual_info_classif, k = 3)

selectorcat_fit = selectorcat.fit(np.array(train[['Pclass', 'Sex', 'SibSp', 'Parch','Embarked']]).reshape(891,5), np.array(train['Survived']).reshape(891,1))

print(selectorcat_fit.scores_) #select sex and Pclass as two of the input column in the model
X_train = train[['Fare', 'Sex', 'Pclass']]

y_train = train['Survived']

X_test = test[['Fare', 'Sex', 'Pclass']]



model1 = LogisticRegression().fit(X_train,y_train)

y_pred = model1.predict(X_test)



gender['Survived'] =  y_pred

gender.head()

gender.to_csv('/kaggle/working/submission_logistic_regression.csv') #Accuracy = 76.56%
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

n_est = [20, 30, 50, 75, 100, 150, 200, 250, 300]

lr_score = {}

for lr in lr_list:

    temp = []

    for ne in n_est:

        gbm = GradientBoostingClassifier(n_estimators=ne, learning_rate=lr, max_features=2, max_depth=2, random_state=0)

        gbm.fit(X_train, y_train)

        temp.append(gbm.score(X_train, y_train))

    lr_score[lr] = temp



lr_n_tb = pd.DataFrame.from_dict(lr_score, orient='index', columns=n_est)

lr_n_tb.index.name = 'learning rate'
graph = lr_n_tb.T.plot.line()

graph.set_xlabel("Number of trees")

graph.set_ylabel("Accuracy")

graph.legend(title = "learning rate")# lr = 0.75 n_trees = 100 is optimum. This is a trade off between overfit and actual accuracy
gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=0.75, max_features=2, max_depth=2, random_state=0)

model2 = gbm.fit(X_train, y_train)

y_pred = model2.predict(X_test)

gender['Survived'] =  y_pred

gender.to_csv('/kaggle/working/submission_GBM.csv') #Accuracy = 77.99% It is the highest accuracy score among these three models
#To find optimal parameters for higher accuracy score

param_c = [50, 100, 120, 150]

svm_score = []

for par_c in param_c:

    svc = svm.SVC(kernel='linear', C=par_c)

    svc.fit(X_train, y_train)

    svm_score.append(svc.score(X_train, y_train))

print(svm_score) #100 is the best C number in SVC



svc = svm.SVC(kernel='linear', C=100)

model3 = svc.fit(X_train, y_train)

y_pred = model3.predict(X_test)

gender['Survived'] = y_pred

gender.to_csv('/kaggle/working/submission_SVM.csv') #Accuracy = 76.56%