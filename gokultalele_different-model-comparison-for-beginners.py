# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Imports for Modeling



#from sklearn.preprocessing import Imputer, MinMaxScaler

from sklearn.model_selection import KFold, train_test_split

from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix, classification_report

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb

from mlxtend.classifier import StackingClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.


train = pd.read_csv('../input/train.csv')

train0 = train[ train['target']==0 ].copy()

train1 = train[ train['target']==1 ].copy()

train.sample(5)
# CALCULATE MEANS AND STANDARD DEVIATIONS

s = [0]*200

m = [0]*200

for i in range(200):

    s[i] = np.std(train['var_'+str(i)])

    m[i] = np.mean(train['var_'+str(i)])

    

# CALCULATE PROB(TARGET=1 | X)

def getp(i,x):

    c = 3 #smoothing factor

    a = len( train1[ (train1['var_'+str(i)]>x-s[i]/c)&(train1['var_'+str(i)]<x+s[i]/c) ] ) 

    b = len( train0[ (train0['var_'+str(i)]>x-s[i]/c)&(train0['var_'+str(i)]<x+s[i]/c) ] )

    if a+b<500: return 0.1 #smoothing factor

    # RETURN PROBABILITY

    return a / (a+b)

    # ALTERNATIVELY RETURN ODDS

    # return a / b

    

# SMOOTH A DISCRETE FUNCTION

def smooth(x,st=1):

    for j in range(st):

        x2 = np.ones(len(x)) * 0.1

        for i in range(len(x)-2):

            x2[i+1] = 0.25*x[i]+0.5*x[i+1]+0.25*x[i+2]

        x = x2.copy()

    return x
import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



Picture = True #draw plots

rmin=-5; rmax=5; res=501

pr = 0.1 * np.ones((200,res))

pr2 = pr.copy()

xr = np.zeros((200,res))

xr2 = xr.copy()

ct2 = 0

for j in range(50):

    if Picture: plt.figure(figsize=(15,8))

    for v in range(4):

        ct = 0

        # CALCULATE PROBABILITY FUNCTION FOR VAR

        for i in np.linspace(rmin,rmax,res):

            pr[v+4*j,ct] = getp(v+4*j,m[v+4*j]+i*s[v+4*j])

            xr[v+4*j,ct] = m[v+4*j]+i*s[v+4*j]

            xr2[v+4*j,ct] = i

            ct += 1

        # SMOOTH FUNCTION FOR PRETTIER DISPLAY

        # BUT USE UNSMOOTHED FUNCTION FOR PREDICTION

        pr2[v+4*j,:] = smooth(pr[v+4*j,:],50)

        if Picture:

            # DISPLAY PROBABILITY FUNCTION

            plt.subplot(2, 4, ct2%4+5)

            plt.plot(xr[v+4*j,:],pr2[v+4*j,:],'-')

            plt.title('P( t=1 | var_'+str(v+4*j)+' )')

            xx = plt.xlim()

            # DISPLAY TARGET DENSITIES

            plt.subplot(2, 4, ct2%4+1)            

            sns.distplot(train0['var_'+str(v+4*j)], label = 't=0')

            sns.distplot(train1['var_'+str(v+4*j)], label = 't=1')

            plt.title('var_'+str(v+4*j))

            plt.legend()

            plt.xlim(xx)

            plt.xlabel('')

        if (ct2%8==0): print('Showing vars',ct2,'to',ct2+7,'...')

        ct2 += 1

    if Picture: plt.show()
def getp2(i,x):

    z = (x-m[i])/s[i]

    ss = (rmax-rmin)/(res-1)

    idx = min( (res+1)//2 + (z-ss/2)//ss, res-1)

    idx = max(idx,0)

    return pr[i,int(idx)]
from sklearn.metrics import roc_auc_score

print('Calculating 200000 predictions and displaying a few examples...')

pred = [0]*200000; ct = 0

for r in train.index:

    p = 0.1

    for i in range(200):

        p *= 10*getp2(i,train.iloc[r,2+i])

    if ct%25000==0: print('train',r,'has target =',train.iloc[r,1],'and prediction =',p)

    pred[ct]=p; ct += 1

print('###############')

print('Validation AUC =',roc_auc_score(train['target'], pred))
test = pd.read_csv('../input/test.csv')

print('Calculating 200000 predictions and displaying a few examples...')

pred = [0]*200000; ct = 0

for r in test.index:

    p = 0.1

    for i in range(200):

        p *= 10*getp2(i,test.iloc[r,1+i])

    if ct%25000==0: print('test',r,'has prediction =',p)

    pred[ct]=p

    ct += 1

sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = pred

sub.to_csv('submission.csv',index=False)

print('###############')

print('Finished. Wrote predictions to submission.csv')
sub.loc[ sub['target']>1 , 'target'] = 1

b = plt.hist(sub['target'], bins=200)
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

                                                    test_size= 0.3, random_state = 2078)

# check the split of train and validation

print('Train:',X_train.shape)

print('Test:',X_test.shape)
def performance(Y_test, logist_pred):

    logist_pred_var = [0 if i < 0.5 else 1 for i in logist_pred]

    print('Confusion Matrix:')

    print(confusion_matrix(Y_test, logist_pred_var)) 

      

    #print(classification_report(Y_test, logist_pred)) 



    fpr, tpr, thresholds = roc_curve(Y_test, logist_pred, pos_label=1)

    print('AUC:')

    print(auc(fpr, tpr))
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



# Create the Submission File using Decision tree model

sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = tree_pred_test

sub.to_csv('Decision_Tree.csv',index=False)

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



# Create the Submission File using Random_Forest model

sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = forest_pred_test

sub.to_csv('Random_Forest.csv',index=False)
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



# Create the Submission File using Light GBM

sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = pred_test

sub.to_csv('LightGBM.csv', index = False)



#sig_clf3 = CalibratedClassifierCV(clf3, method="sigmoid")

submitLGB.head()