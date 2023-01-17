from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Data manipulation, linear algebra

import numpy as np 

import pandas as pd 



# Data visualisation

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Machine Learning

# Models

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

# Model selection

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV



# For reproducibility

import random



SEED = 0

np.random.seed(SEED)

random.seed(SEED)
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



full = train.append( test , ignore_index = True )
train.info()
test.info()
train.head()
full_data = [train, test]



for dataset in full_data:

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

    dataset['Age'] = dataset['Age'].fillna(train['Fare'].median())
def drop_unwanted_variables(variable_list, df):

    df = df.drop(variable_list, axis=1)

    return df

    

unwanted_vars_train = ['Embarked', 'Cabin', 'Ticket', 'Name', 'PassengerId']

unwanted_vars_test = ['Embarked', 'Cabin', 'Ticket', 'Name']



train = drop_unwanted_variables(unwanted_vars_train, train)

test = drop_unwanted_variables(unwanted_vars_test, test)

train.info()
test.info()
train.head()
train.describe()
corr = train.corr()

_, ax = plt.subplots(figsize=(13,10)) 

_ = sns.heatmap(corr, ax=ax,

                xticklabels=corr.columns.values,

                yticklabels=corr.columns.values,

                annot = True, 

                linewidths=0.1, 

                cmap = plt.cm.PuOr)
plot_vars = [u'Survived', u'Pclass', u'Sex', u'Age', u'SibSp', u'Parch', u'Fare']

sns.set()

sns.pairplot(train, vars = plot_vars, hue='Survived')
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean()
train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean()
ax = sns.violinplot(x='Survived', y ='Age', data=train)
sns.violinplot(x='Survived', y ='Fare', data=train)
X_train = train.drop("Survived", axis=1)

Y_train = train["Survived"]
lr=LogisticRegression()

lr.fit(X_train, Y_train)

acc_log = round(lr.score(X_train, Y_train) * 100, 2)

acc_log
coeff_df = pd.DataFrame(train.columns.delete(0))

coeff_df.columns=['Feature']

coeff_df['Coefficient']=pd.Series(lr.coef_[0])

coeff_df['Coef_abs']=pd.Series(abs(lr.coef_[0]))



coeff_df.sort_values(by='Coef_abs', ascending=False)
rf=RandomForestClassifier(n_estimators=100)

rf.fit(X_train, Y_train)

acc_rf = rf.score(X_train, Y_train)

print("Random Forest accuracy score : {:.4f}".format(acc_rf))
ntrain = train.shape[0]

NFOLDS = 5 

kf = KFold(n_splits= NFOLDS, shuffle=True, random_state=SEED)

kfold_score = cross_val_score(rf, X_train, Y_train, cv=kf)

kfold_score_mean = np.mean(kfold_score)



print("Random Forest accuracy score per fold : ", kfold_score, "\n")

print("Average accuracy score : {:.4f}".format(kfold_score_mean))
parameters = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],

             'max_features' : ["auto", None, "sqrt", "log2"],

             'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

clf = GridSearchCV(rf, parameters)

clf.fit(X_train, Y_train)
rf_Model = clf.best_estimator_

print ("Chosen model  score: {:.4f}".format(clf.best_score_), 

       "\nChosen model parameters: ",clf.best_params_)

print(rf_Model)
X_test = test.drop(["PassengerId"], axis=1)

labels = test["PassengerId"]

Y_test = rf_Model.predict(X_test)



submission = pd.DataFrame({

        "PassengerId": labels,

        "Survived": Y_test

    })



submission.to_csv("Titanic_submit.csv", index = False)