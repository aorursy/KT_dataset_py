import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns # Seaborn visualization library

sns.set(style="darkgrid")



import os

print(os.listdir("../input"))



%matplotlib inline



import gc

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
# Look at first 10 records of the train dataset

train.head(n=10).T
# Check out the shape of the train and test sets

print('Train:', train.shape)

print('Test:', test.shape)
# Check the target variable destribution

train['target'].value_counts()
# Imports for Modeling



#from sklearn.preprocessing import Imputer, MinMaxScaler

from sklearn.model_selection import KFold, train_test_split

from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb
# Target variable from the Training Set

Target = train['target']



# Input dataset for Train and Test 

train_inp = train.drop(columns = ['target', 'ID_code'])

test_inp = test.drop(columns = ['ID_code'])



# List of feature names

features = list(train_inp.columns)
# Split the Train Dataset into training and validation sets for model building. 

# The training set now has 140K records and validation set has 60K records



X_train, X_test, Y_train, Y_test = train_test_split(train_inp, Target, 

                                                    test_size= 0.3, random_state = 2019)
# check the split of train and validation

print('Train:',X_train.shape)

print('Test:',X_test.shape)
# Create an object of Logistic Regression with parameters C and class_weight

logist = LogisticRegression(C=0.001, class_weight='balanced')



# Fit the training data on this object

logist.fit(X_train, Y_train)
# Predict the Target for validation dataset 

logist_pred = logist.predict_proba(X_test)[:,1]
logist_pred
def performance(Y_test, logist_pred):

    logist_pred_var = [0 if i < 0.5 else 1 for i in logist_pred]

    print('Confusion Matrix:')

    print(confusion_matrix(Y_test, logist_pred_var)) 

      

    #print(classification_report(Y_test, logist_pred)) 



    fpr, tpr, thresholds = roc_curve(Y_test, logist_pred, pos_label=1)

    print('AUC:')

    print(auc(fpr, tpr))
performance(Y_test, logist_pred)
# Submission dataframe

logist_pred_test = logist.predict_proba(test_inp)[:,1]



submit = test[['ID_code']]

submit['target'] = logist_pred_test



submit.head()
# Create the Submission File using logistic regression model

submit.to_csv('log_reg_baseline.csv', index = False)
# Create Decision Tree Classifier object with few parameters

tree_clf = DecisionTreeClassifier(class_weight='balanced', random_state = 2019, 

                                  max_features = 0.7, min_samples_leaf = 80)



# Fit the object on training data

tree_clf.fit(X_train, Y_train)
# Predict for validation set and check the performance

tree_preds = tree_clf.predict_proba(X_test)[:, 1]

performance(Y_test, tree_preds)
# Submission dataframe

tree_pred_test = tree_clf.predict_proba(test_inp)[:, 1]



submitTree = test[['ID_code']]

submitTree['target'] = tree_pred_test



# Create the Submission File using logistic regression model

submitTree.to_csv('Decision_Tree.csv', index = False)
# Extract feature importances

feature_importance_values = tree_clf.feature_importances_

feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

feature_importances.sort_values(by='importance', ascending=False).head(n=10)
plt.figure(figsize=(20,8))

sns.boxplot(data=train[['var_81', 'var_139', 'var_12', 'var_26', 'var_146', 'var_110',

                        'var_109', 'var_53', 'var_6', 'var_166']])
# Create random Forest Object using the mentioned parameters

random_forest = RandomForestClassifier(n_estimators=100, random_state=2019, verbose=1,

                                      class_weight='balanced', max_features = 0.5, 

                                       min_samples_leaf = 100)



# Fit the object on training set 

random_forest.fit(X_train, Y_train)
# Predict the validation set target and check the performance

forest_preds = random_forest.predict_proba(X_test)[:, 1]

performance(Y_test, forest_preds)
# Submission dataframe

forest_pred_test = random_forest.predict_proba(test_inp)[:, 1]



submitForest = test[['ID_code']]

submitForest['target'] = forest_pred_test



# Create the Submission File using logistic regression model

submitForest.to_csv('Random_Forest.csv', index = False)
# Extract feature importances

feature_importance_values = random_forest.feature_importances_

feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

feature_importances.sort_values(by='importance', ascending=False).head(n=10)
#custom function to build the LightGBM model.

def run_lgb(X_train, Y_train, X_test, Y_test, test_inp):

    params = {

        "objective" : "binary",

        "metric" : "auc",

        "num_leaves" : 1000,

        "learning_rate" : 0.01,

        "bagging_fraction" : 0.8,

        "feature_fraction" : 0.8,

        "bagging_freq" : 5,

        "reg_alpha" : 1.728910519108444,

        "reg_lambda" : 4.9847051755586085,

        "random_state" : 42,

        "bagging_seed" : 2019,

        "verbosity" : -1,

        "max_depth": 18,

        "min_child_samples":100

       # ,"boosting":"rf"

    }

    

    lgtrain = lgb.Dataset(X_train, label=Y_train)

    lgval = lgb.Dataset(X_test, label=Y_test)

    evals_result = {}

    model = lgb.train(params, lgtrain, 2500, valid_sets=[lgval], 

                      early_stopping_rounds=50, verbose_eval=50, evals_result=evals_result)

    

    pred_test_y = model.predict(test_inp, num_iteration=model.best_iteration)

    return pred_test_y, model, evals_result



# Training the model #

pred_test, model, evals_result = run_lgb(X_train, Y_train, X_test, Y_test, test_inp)
# Extract feature importances

feature_importance_values = model.feature_importance()

feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})

feature_importances.sort_values(by='importance', ascending=False).head(n=10)
# Submission dataframe

pred_test[pred_test>1] = 1

pred_test[pred_test<0] = 0



submitLGB = test[['ID_code']]

submitLGB["target"] = pred_test



# Create the Submission File using Light GBM

submitLGB.to_csv('LightGBM.csv', index = False)



submitLGB.head()