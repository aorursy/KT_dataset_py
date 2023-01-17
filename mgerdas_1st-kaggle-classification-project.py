# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import re

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.metrics import confusion_matrix



from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import roc_curve, accuracy_score, make_scorer, roc_auc_score, confusion_matrix, classification_report, cohen_kappa_score



from sklearn import ensemble, tree



sns.set(style="ticks", context="talk")

#rcParams['figure.figsize'] = (8, 5)
testing = pd.read_csv('/kaggle/input/titanic/test.csv')

train = pd.read_csv('/kaggle/input/titanic/train.csv')



target = 'Survived'



test = testing.copy()



#checking data type and null values, overview of dataset

test.info()

print('-'*70)

train.info()

print('-'*70)

train.tail(10)
train.describe()
test.describe()
train.describe(include=['O'])
test.describe(include=['O'])
print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False).round(2), 

'\n\n',

train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False).round(2))
#feature engineering

def title(i):

    i['Title'] = i.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))

    i['Title'] = i['Title'].replace({'Mlle':'Miss', 'Mme':'Mrs', 'Ms':'Miss'})

    i['Title'] = i['Title'].replace(['Don', 'Dona', 'Rev', 'Dr', 'Major', 'Lady', 'Sir', 'Col', 'Capt', 'Countess', 'Jonkheer'],'Special')

    

title(train)

title(test)

#---------------------------------------------------------------------------------------------------------------------------------------



train['Age'] = train.groupby(['Title'])['Age'].apply(lambda x: x.fillna(x.median()))

test['Age'] = train.groupby(['Title'])['Age'].apply(lambda x: x.fillna(x.median()))



#---------------------------------------------------------------------------------------------------------------------------------------



def fillna_median_fare(i, u):

    i[u] = i.groupby(['Pclass', 'Sex'])[u].apply(lambda x: x.fillna(x.median()))

    

fillna_median_fare(test,'Fare')

#---------------------------------------------------------------------------------------------------------------------------------------



train['Embarked'] = train['Embarked'].fillna("S")

#---------------------------------------------------------------------------------------------------------------------------------------



def family_size(x):

    x['Family'] =  x["Parch"] + x["SibSp"]



family_size(train)

family_size(test)
g = sns.FacetGrid(train, col='Survived', height = 4)

g.map(plt.hist, 'Age', bins=20);
train['agebucket'] = pd.cut(train['Age'], 5)

test['agebucket'] = pd.cut(test['Age'], 5)



train[['agebucket', 'Survived']].groupby(['agebucket']).mean().sort_values(by='agebucket', ascending=True).round(2)
for dataset in [train, test]:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

    

train.head()
print(train[['Family', 'Survived']].groupby(['Family'], as_index=False).mean().sort_values(by='Survived', ascending=False).round(2),

'\n\n',

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False).round(2))
train['isalone'] = [1 if x == 1 else 0 for x in train['Family']]

test['isalone'] = [1 if x == 1 else 0 for x in test['Family']]



train[['isalone', 'Survived']].groupby(['isalone']).mean().sort_values(by='Survived', ascending=False).round(2)
# Preparing features for analysis

dummy_features = ['Sex','Title', 'isalone']

drop_features = ['Embarked', 'PassengerId', 'Ticket', 'Name', 'Cabin','Parch','SibSp', 'agebucket']

    

train = pd.concat([train, pd.get_dummies(train[dummy_features])], axis = 1, sort = False)

train.drop(columns = train[dummy_features], inplace = True)

train.drop(columns = train[drop_features], inplace = True)



test = pd.concat([test, pd.get_dummies(test[dummy_features])], axis = 1, sort = False)

test.drop(columns = test[dummy_features], inplace = True)

test.drop(columns = test[drop_features], inplace = True)



train.tail()
# cor = train.corr()

# pyplot.figure(figsize=(20,8))

# sns.heatmap(cor, annot = True);



# # cor_target = abs(cor[target])



# # #Selecting highly correlated features

# # relevant_features = cor_target[cor_target>=0.0]

# # print(relevant_features)
#last check for NaN values in dataset and check if column amount is the same in both datasets

train.info()

print('-'*70)

test.info()
# Separating target column from other features

y = train[target]

x = train.drop(columns = target)



# Train and Test dataset split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 42)
# random forest model hyper-tuned



RF = ensemble.RandomForestClassifier()

RF_params = {

          'n_estimators':[n for n in range(60,140,10)],

          'max_depth':[n for n in range(3, 6)],

          #'min_samples_leaf': [n for n in range(2, 6, 2)],

          'max_features' : ['sqrt', 'log2', None],

          'random_state' : [42]

            }



RF_model = GridSearchCV(RF, param_grid = RF_params, cv = 5, n_jobs = -1).fit(x_train, y_train)

print("Best Hyper Parameters:",RF_model.best_params_)



# Area under the curve probability score

RF_probs = RF_model.predict_proba(x_test)

RF_probs = RF_probs[:, 1]

RF_auc = roc_auc_score(y_test, RF_probs)

print('AUC: %.3f' % RF_auc)



RF_predictions = RF_model.predict(x_test)

RF_accuracy = accuracy_score(y_test, RF_predictions)

print("RF accuracy: %.3f" % RF_accuracy)



# AUC plot

RF_fpr, RF_tpr, RF_thresholds = roc_curve(y_test, RF_probs)

plt.plot([0, 1], [0, 1], linestyle='--')

plt.plot(RF_fpr, RF_tpr, color = 'tab:green')

plt.show()
#gradient boosting tree model hyper-tuned



GBT = ensemble.GradientBoostingClassifier()

GBT_params = {

          'n_estimators':[n for n in range(180, 240, 20)],

          'max_depth':[n for n in range(3, 6)],

          #'min_samples_leaf': [n for n in range(2, 6, 2)],

          'learning_rate': [0.1, 0.25, 0.5],

          'random_state' : [42]

             }



GBT_model = GridSearchCV(GBT, param_grid = GBT_params, cv = 5, n_jobs = -1)

GBT_model.fit(x_train, y_train)

print("Best Hyper Parameters:",GBT_model.best_params_)



# Area under the curve probability score

GBT_probs = GBT_model.predict_proba(x_test)

GBT_probs = GBT_probs[:, 1]

GBT_auc = roc_auc_score(y_test, GBT_probs)

print('AUC: %.3f' % GBT_auc)



GBT_predictions = GBT_model.predict(x_test)

GBT_accuracy = accuracy_score(y_test, GBT_predictions)

print("GBT accuracy: %.3f" % GBT_accuracy)



# AUC plot

GBT_fpr, GBT_tpr, GBT_thresholds = roc_curve(y_test, GBT_probs)

plt.plot([0, 1], [0, 1], linestyle='--')

plt.plot(GBT_fpr, GBT_tpr, color = 'tab:orange')

plt.show()
pd.Series(RF_model.best_estimator_.feature_importances_, index=x.columns).nlargest(15).plot(kind='barh')

plt.figure(figsize=[6,5])

plt.show()
print("GBT cohen_kappa_score: %.3f" % cohen_kappa_score(y_test, GBT_predictions))

print("RF cohen_kappa_score: %.3f" % cohen_kappa_score(y_test, RF_predictions))



#The kappa score (see docstring) is a number between -1 and 1. Scores above .8 are generally considered good agreement; zero or lower means no agreement (practically random labels).
# The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.

# The recall is intuitively the ability of the classifier to find all the positive samples.

# The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0.

print("GBT", classification_report(y_test, GBT_predictions))

print("-"*100)

print("RF", classification_report(y_test, RF_predictions))
# Area under the curve probability score

b = '\033[1m'

ub = '\033[0m'



plt.figure(figsize=[6,4])

print(b + 'AUC scores :')

print(ub + 'Random Forest - ' + b +  '%.3f' % RF_auc)

print(ub + 'Gradient Boosting Classifier - ' + b +  '%.3f' % GBT_auc)

# print(ub + 'XGBoost Classifier - ' + b +  '%.3f' % XGB_auc)



# AUC plot

plt.plot([0, 1], [0, 1], linestyle='--', color = 'grey')

plt.plot(RF_fpr, RF_tpr, color = 'tab:green')

plt.plot(GBT_fpr, GBT_tpr, color = 'tab:orange')

# plt.plot(XGB_fpr, XGB_tpr, color = 'tab:blue')



plt.legend(('',

            'Random Forest', 

            'Gradient Boost Classifier',

           ))

plt.show()
def conf_matrix(x, y):

    prediction = x.predict(x_test)

    CM_abs = confusion_matrix(y_test, prediction)

#     CM_rel = confusion_matrix(y_test, prediction, normalize = 'true')



    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,3))

    sns.heatmap(CM_abs, annot = True, fmt='d', ax = ax1)

    ax1.set(xlabel='Predicted', ylabel='Actual')

    ax1.set_title(y)

    

#     sns.heatmap(CM_rel, annot = True, fmt='.2f', ax = ax2);

    ax2.set(xlabel='Predicted', ylabel='Actual')

    ax2.set_title(y)



    pred = accuracy_score(y_test, prediction)

    print(y + " accuracy: %.3f" % pred)

    

    

conf_matrix(RF_model, 'Random Forest')  

conf_matrix(GBT_model, 'Gradient Boosting')  

# conf_matrix(XGB_model, 'XGB')
predict_RF = RF_model.predict(test)

predict_GBT = GBT_model.predict(test)



submit_RF = pd.DataFrame({'PassengerId':testing['PassengerId'],'Survived':predict_RF})

submit_GBT = pd.DataFrame({'PassengerId':testing['PassengerId'],'Survived':predict_GBT})





#creating submission file

filename_RF = 'Titanic Prediction RF.csv'

submit_RF.to_csv(filename_RF,index=False)

print('Saved file: ' + filename_RF)



filename_GBT = 'Titanic Prediction GBT.csv'

submit_GBT.to_csv(filename_GBT,index=False)

print('Saved file: ' + filename_GBT)