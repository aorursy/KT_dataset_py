# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression as LogR



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# First load training and test data and remove features believed to be of low improtance

train = pd.read_csv("../input/titanic/train.csv")

# remove unimportant features. 

train_name = list(train['Name'])

train = train.drop(columns = ['Name', 'Ticket', 'Cabin'])



test = pd.read_csv("../input/titanic/test.csv")

test_name = list(test['Name'])

test = test.drop(columns = ['Name', 'Ticket', 'Cabin'])
train.isnull().sum()
test.isnull().sum()
test.corr().Fare
# replace 0 fare values with NaN, since it does not make sense

train['Fare'] = train['Fare'].map(lambda x: np.nan if x ==0 else x)

test['Fare'] = test['Fare'].map(lambda x: np.nan if x ==0 else x)
# impute missing Age and Fare data using sklearn's SimpleImputer

from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='median')



train_data = train[['Age','Fare']]

imp.fit(train_data)

train_data = pd.DataFrame(imp.transform(train_data))

train[['Age','Fare']] = train_data



test_data = test[['Age','Fare']]

imp.fit(train_data)

test_data = pd.DataFrame(imp.transform(test_data))

test[['Age','Fare']] = test_data

# Change Pclass dtype to str

train['Pclass'] = train['Pclass'].astype(str)

test['Pclass'] = test['Pclass'].astype(str)



train.corr().Survived.sort_values(ascending = False)
# categorize Age data into child (0-14), Adult (15-60) and seniors (61+)

train_child = [0]*len(train)

train_adult = [0]*len(train)

train_senior = [0]*len(train)

for i in range(len(train)):

    if train['Age'][i] <= 14:

        train_child[i] = 1

    elif train['Age'][i] <= 60:

        train_adult[i] = 1

    else:

        train_senior[i] = 1

train['isChild'] = train_child

train['isAdult'] = train_adult

train['isSenior'] = train_senior

train = train.drop(columns = 'Age')



test_child = [0]*len(test)

test_adult = [0]*len(test)

test_senior = [0]*len(test)

for i in range(len(test)):

    if test['Age'][i] <= 14:

        test_child[i] = 1

    elif test['Age'][i] <= 60:

        test_adult[i] = 1

    else:

        test_senior[i] = 1

test['isChild'] = test_child

test['isAdult'] = test_adult

test['isSenior'] = test_senior

test = test.drop(columns = 'Age')
train.corr().Survived
# categorize Fare data into class1 (0-10), class2(10-20), class3(20-30), class4(30+)

train_fare = ['class1']*len(train)

for i in range(len(train)):

    if 10 <= train['Fare'][i] < 20:

        train_fare[i] = 'class2'

    elif 20 <= train['Fare'][i] < 30:

        train_fare[i] = 'class3'

    elif 30 <= train['Fare'][i]:

        train_fare[i] = 'class4'

train['Fare'] = train_fare



test_fare = ['class1']*len(test)

for i in range(len(test)):

    if 10 <= test['Fare'][i] < 20:

        test_fare[i] = 'class2'

    elif 20 <= test['Fare'][i] < 30:

        test_fare[i] = 'class3'

    elif 30 <= test['Fare'][i]:

        test_fare[i] = 'class4'

test['Fare'] = test_fare

    
# Assign age group based on title (some children were imputed as adults but can easily be seen to be children based on their titles)

for i in range(len(train)):

    if 'Master.' in train_name[i]:

        train.set_value(i,'isChild',1)

        train.set_value(i,'isAdult',0)

        train.set_value(i,'isSenior',0)

for i in range(len(test)):

    if 'Master.' in test_name[i]:

        test.set_value(i,'isChild',1)

        test.set_value(i,'isAdult',0)

        test.set_value(i,'isSenior',0)
# Add family size as a feature to the training and test data

train_famil = [1]*len(train)

for i in range(len(train)):

    train_famil[i] = train_famil[i] + train['Parch'][i] + train['SibSp'][i]

train['family size'] = train_famil



test_famil = [1]*len(test)

for i in range(len(test)):

    test_famil[i] = test_famil[i] + test['Parch'][i] + test['SibSp'][i]

test['family size'] = test_famil
"""

Logistic Regression using Pclass, Sex, Fare and Age, Pclass, family size, Embarked port as features

"""



Y_train = train['Survived']

X_train = train[['isChild','isAdult','isSenior','Fare','Sex','Pclass', 'family size','Embarked']]

X_test = test[['isChild','isAdult','isSenior','Fare','Sex','Pclass', 'family size','Embarked']]



# Convert Embarked to One-hot-class encoding via the pd.get_dummies method 

X_train = pd.get_dummies(X_train)

X_test = pd.get_dummies(X_test)



# fit model

Log_R_model = LogR(solver = 'lbfgs').fit(X_train,Y_train)



Y_test = Log_R_model.predict(X_test)

passId = np.array(test['PassengerId'])



prediction = pd.DataFrame(data = np.transpose([passId, Y_test]), columns = ['PassengerId','Survived'])

prediction = prediction.astype('int32')



# write prediction to .csv file

# This line is commented out since this .csv file is written locally

# prediction.to_csv('prediction_LR.csv', index = False)



"""

tenfold CV using GridsearchCV

"""

from sklearn.model_selection import GridSearchCV

# penalty space

penalty = ['l1','l2']

# regularization hyperparameter space

Cspace = np.linspace(0.01,1)

hyperparameters = dict(C=Cspace, penalty=penalty)



# create new logistic regression model

LRmodel = LogR() 

clf = GridSearchCV(LRmodel, hyperparameters, cv=10, verbose=False)



best_LR = clf.fit(X_train, Y_train)



Y_test = best_LR.predict(X_test)

# predict probability of survival being 0/1



LR_proba = best_LR.predict_proba(X_test)

passId = np.array(test['PassengerId'])



LR_prediction = pd.DataFrame(data = np.transpose([passId, Y_test]), columns = ['PassengerId','Survived'])

LR_prediction = LR_prediction.astype('int32')
best_LR.score(X_train,Y_train)
best_LR.best_estimator_
from sklearn.ensemble import GradientBoostingClassifier

param_grid = {'max_features':['sqrt',4,5,6,7,8], 'max_depth': [i for i in range(1,7)]}



# create new logistic regression model

GB_model = GradientBoostingClassifier( random_state = 0) 

GridSearch = GridSearchCV(GB_model, param_grid, cv=10, verbose=False)



best_GB = GridSearch.fit(X_train, Y_train)



# predict GB probabilities 

GB_proba = best_GB.predict_proba(X_test)

Y_test = best_GB.predict(X_test)

passId = np.array(test['PassengerId'])



GB_prediction = pd.DataFrame(data = np.transpose([passId, Y_test]), columns = ['PassengerId','Survived'])

GB_prediction = GB_prediction.astype('int32')
best_GB.score(X_train,Y_train)
# find predictions where LR and GB models differ:



diff = []



for i in range(len(X_test)):

    if GB_prediction.Survived[i] != LR_prediction.Survived[i]:

        diff.append(i)

    

disagree = len(diff)/len(X_test)

    



LR_disagree_prob = []

for i in diff:

    LR_disagree_prob.append(LR_proba[i])



print('number of disagreeing instances:')

print(len(diff))



print('percentage of disagreeing instances:')

print(round(len(diff)/len(X_test)*100,2))
print('probabilities associated with differing predictions using LR model:')

LR_disagree_prob
# make table of final prediction of blended model

final_prediction = GB_prediction.copy()

for i in range(len(X_test)):

    if i in diff:

        if LR_disagree_prob[diff.index(i)][1] > 0.75:

            final_prediction.Survived[i] = 1

        elif LR_disagree_prob[diff.index(i)][1] < 0.25:

            final_prediction.Survived[i] = 0

        else:

            continue

# write prediction to .csv file

final_prediction.to_csv('prediction_Blended.csv', index = False)