# importing required packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb

plt.rcParams['figure.figsize'] = (9.0,9.0)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_dataset = pd.read_csv('/kaggle/input/titanic/train.csv')

test_dataset =  pd.read_csv('/kaggle/input/titanic/test.csv')
# Shape of the datasets

print(train_dataset.shape, test_dataset.shape)
test_dataset.head()
train_dataset.head()
train_dataset.SibSp.unique()
train_dataset.describe()
train_dataset.isnull().sum()
sb.heatmap(train_dataset.isnull())
# writing a function for imputing the data

def imput(cols):

    Age = cols[0]

    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:

            return 37

        elif Pclass == 2:

            return 29

        else:

            return 24

    else:

        return Age
train_dataset['Age'] = train_dataset[['Age','Pclass']].apply(imput, axis=1)
train_dataset['Embarked'].unique()
# a function for missing value imputation

def impu(Embarked):

    if pd.isnull(Embarked):

        return 'C'

    else:

        return Embarked
train_dataset['Embarked'] = train_dataset['Embarked'].apply(impu)
sb.heatmap(train_dataset.isnull())
train_dataset.info()
test_dataset.isnull().sum()
test_dataset.info()
test_dataset['Age'] = test_dataset[['Age','Pclass']].apply(imput, axis=1)
sb.heatmap(test_dataset.isnull())
train_dataset.groupby(train_dataset.Age//10*10).size().plot.bar(cmap='Set3', width=0.9)

plt.title('Age Group Size', fontsize = 20)

plt.show()
sb.countplot(x='Survived',data = train_dataset, palette = 'Dark2')

plt.title('Survival Count', fontsize = 20)

plt.show()
sb.countplot(x='Survived',hue = train_dataset['Sex'], data = train_dataset, palette = 'Reds')

plt.title('Survived vs sex', fontsize = 20)

plt.show()
sb.countplot(x='Survived', hue='Pclass',data=train_dataset)

plt.title('Survived vs Pclass', fontsize = 20)

plt.show()
sb.boxplot(x='Survived', y='Age',data=train_dataset, palette = 'winter')

plt.title('Survived vs Age', fontsize = 20)

plt.show()

train_dataset = train_dataset.set_index('PassengerId')

train_dataset = train_dataset.drop(columns=['Name', 'Ticket', 'Cabin'])

train_dataset = pd.get_dummies(train_dataset, columns=[ 'Pclass','Sex','Embarked'])
train_dataset.head()
test_dataset = test_dataset.set_index('PassengerId')

test_dataset = test_dataset.drop(columns=['Name', 'Ticket', 'Cabin'])

test_dataset = pd.get_dummies(test_dataset, columns=['Pclass','Sex', 'Embarked'])

test_dataset.head()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(train_dataset.drop(['Survived'],axis=1),

                                                    train_dataset['Survived'], test_size=0.1,  random_state= 101)
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from xgboost.sklearn import XGBClassifier

model = XGBClassifier()

model.fit(x_train, y_train)
# parameters you want to test, finding out which gives the best accuracy.

params={

 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,

 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],

 "min_child_weight" : [ 1, 3, 5, 7 ],

 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],

 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]

    }
#finding the best parameter

random_search=RandomizedSearchCV(model,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)
random_search.fit(x_train,y_train)
# this gives out the best estimator values

# you can literally copy this and paste it

random_search.best_estimator_
# best parameters that we asked for. 

random_search.best_params_
# slecting paramters avoiding overfitting or underfitting, being careful with learning_rate and max_depth etc

xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.7, gamma=0.2,

              learning_rate=0.1, max_delta_step=0, max_depth=6,

              min_child_weight=5, missing=None, n_estimators=100, n_jobs=-1,

              nthread=None, objective='binary:logistic', random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

              silent=None, subsample=1, verbosity=1)

xgb.fit(x_train,y_train)

pred = xgb.predict(x_test)
print('Score:', xgb.score(x_test,y_test))
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test,pred))
model.predict(test_dataset)
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score
def plot_roc_curve(fpr,tpr):

    plt.plot(fpr, tpr, color = 'lime', label = 'ROC')

    plt.plot([0,1], [0, 1], color = 'darkblue', linestyle='--')

    plt.xlabel('False positive rate')

    plt.ylabel('true positive rate')

    plt.title('receiver operating charcteristic (ROC Curve')

    plt.legend()

    plt.show()
probs = model.predict_proba(x_test)
probs = probs[:,1]

probs
auc = roc_auc_score(y_test, probs)

print("auc:%.2f" %auc)
fpr, tpr, thresholds = roc_curve(y_test, probs)

plot_roc_curve(fpr,tpr)
submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])

submission_df['PassengerId'] = test_dataset.index

submission_df['Survived'] = xgb.predict(test_dataset)

submission_df.to_csv('submissions.csv', header=True, index=False)

submission_df.head(10)