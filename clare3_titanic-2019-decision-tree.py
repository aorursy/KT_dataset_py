# IMPORTING THE GOOD STUFF

import pandas as pd

import numpy as np

import random as rnd





import seaborn as sns

import matplotlib.pyplot as plt

import sklearn.model_selection

from sklearn.linear_model import LogisticRegression

import sklearn.metrics

import sklearn.tree as sktree
train_dataframe=pd.read_csv("../input/train.csv")

test_dataframe=pd.read_csv("../input/test.csv")

gender_submission = pd.read_csv("../input/gender_submission.csv")

print("Got them!")



# cleansing as before. Define Cabin? as cabin known (or did they have one? presumably they did)

train_dataframe['Cabin?'] = np.where(pd.isnull(train_dataframe['Cabin']), 0,1)

predictors = ['Pclass','Sex', 'Age', 'SibSp','Parch', 'Fare', 'Cabin?', 'Embarked']



# missing values for age - impute from medians for sex and class

def impute_age(cols):

    age = cols[0]

    sex = cols[1]

    pclass = cols[2]

    if pd.isnull(age):

        if sex == 'female':

            if pclass == 1:

                return 35

            elif pclass == 2:

                return 28

            elif pclass == 3:

                return 21.5

            else:

                print('error! pclass should be 1, 2, or 3 but it is '+pclass+'!')

                return np.nan

        elif sex == 'male':

            if pclass == 1:

                return 40

            elif pclass == 2:

                return 30

            elif pclass == 3:

                return 25

            else:

                print('error! pclass should be 1, 2, or 3 but it is '+pclass+'!')

                return np.nan

        else: print('error! sex should be female or male but it is '+sex+'!')

    else:

        return age



    

train_dataframe['Age']=train_dataframe[['Age','Sex','Pclass']].apply(impute_age,axis=1)
# FOR SOME MODELS, WILL NEED TO HAVE ONLY NUMERICAL FEATURES

# NOT THE CASE FOR TREES SO DON'T NEED TO RUN THIS CODE

# sex and embarked are categorical

male = pd.get_dummies(train_dataframe['Sex'], drop_first=True)

port = pd.get_dummies(train_dataframe['Embarked'], drop_first=True)

train = pd.concat([train_dataframe, male, port],axis=1)

train[pd.isnull(train['Embarked'])==True]

# just change those to 0.33 don't know

train.loc[61,'Q']=0.33

train.loc[61,'S']=0.33

train.loc[829,'Q']=0.33

train.loc[829,'S']=0.33

train[pd.isnull(train['Embarked'])==True]

predictors_num = ['Pclass', 'male', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin?', 'Q','S']

#train = train_dataframe
# Train test split the training data

X_tr, X_test, y_tr, y_test = sklearn.model_selection.train_test_split(train[predictors_num],train['Survived'], random_state = 1)
tree_model = sktree.DecisionTreeClassifier(max_depth = 3, random_state = 28)

tree_model.fit(X_tr, y_tr)

predictions = tree_model.predict(X_test)

sklearn.metrics.confusion_matrix(y_test, predictions)
tree_model = sktree.DecisionTreeClassifier(max_depth = 30, random_state = 28)

tree_model.fit(X_tr, y_tr)

predictions = tree_model.predict(X_test)

sklearn.metrics.confusion_matrix(y_test, predictions)

# it got worse with 30 layers - presumably overfitting

# I tried out a few and actually think 3 is optimal
tree_model = sktree.DecisionTreeClassifier(max_depth = 3, random_state = 28)

tree_model.fit(X_tr, y_tr)

predictions = tree_model.predict(X_test)

tree_model.feature_importances_

# interestingly, this only uses a few of the features

# sex is still the most important, parch and port are also irrelevant
predictors_num
# Now repeat for the whole dataset

tree_model = sktree.DecisionTreeClassifier(max_depth = 3, random_state = 28)

tree_model.fit(train[predictors_num], train['Survived'])





feature_dataframe = pd.DataFrame(tree_model.feature_importances_, index = predictors_num)

feature_dataframe.to_csv("tree feature importance.csv", index=False)
#see whether this works! Should be able to see it pretty clearly since it's only got 3 layers

tree = sktree.export_graphviz(tree_model, 'tree.dot')
# now I need to apply the same cleansing to the test data as I did to the training data

test_dataframe['Cabin?'] = np.where(pd.isnull(test_dataframe['Cabin']), 0,1)



# for age, I found the medians unchanged by using all the data

test_dataframe['Age']=test_dataframe[['Age','Sex','Pclass']].apply(impute_age,axis=1)



sex = pd.get_dummies(test_dataframe['Sex'])

port = pd.get_dummies(test_dataframe['Embarked'])

sex.drop('female', axis = 1, inplace=True)

port.drop('C', axis = 1, inplace = True)

test = pd.concat([test_dataframe, sex, port],axis=1)

# and there are no missing values for port this time



# but there was one missing fare

# I gave him the median fare for his class

test['Fare']=np.where(pd.isnull(test['Fare'])==True, 7.8958, test['Fare'])
final_predictions = tree_model.predict(test[predictors_num])

final = pd.DataFrame(final_predictions, columns = ['Survived'])
output = pd.concat([test['PassengerId'],final], axis=1)

output
output.to_csv('csv_to_submit.csv', index=False)