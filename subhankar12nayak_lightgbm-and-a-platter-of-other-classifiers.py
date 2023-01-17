import numpy as np

import pandas as pd

from scipy import stats



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



from sklearn.utils import shuffle



from sklearn.decomposition import PCA



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from imblearn.under_sampling import NearMiss

from imblearn.over_sampling import SMOTE



from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import SGDClassifier

from sklearn import svm

from sklearn.ensemble import VotingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import GradientBoostingClassifier

from lightgbm import LGBMClassifier

from lightgbm import record_evaluation



from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score



from sklearn.pipeline import Pipeline

import joblib

import pickle

import json



import random



import warnings

warnings.filterwarnings('ignore')
ccfr= pd.read_csv('../input/creditcardfraud/creditcard.csv')

ccfr.head()
x= ccfr['Class']

plt.figure(figsize=(5,7))

sns.countplot(ccfr['Class'])

plt.xlabel('Class')

plt.ylabel('Count')

plt.show()
fraud_df= ccfr[ccfr['Class']==1]

nonfraud_df= ccfr[ccfr['Class']==0]

fraudcount= fraud_df.count()[0]

nonfraudcount= nonfraud_df.count()[0]

print ("Frauds=",fraudcount)

print ("Non Frauds=",nonfraudcount)

print (ccfr.shape)
ccfr.describe()
ccfr.isna().sum()
plt.figure(figsize=(22,22))

sns.heatmap(ccfr.corr(),cmap="coolwarm", annot=True)

plt.show()
for c in ccfr.columns[0:30]:

    print ("******************** COLUMN ",c," ***********************")

    col= ccfr[c]

    col=np.array(col)

    col_mean= np.mean(col)

    col_median= np.median(col)

    col_std= np.std(col)

    col_var= np.var(col)

    col_range= col.max()-col.min()

    fig=sns.FacetGrid(ccfr,hue="Class",height=5,aspect=2,palette=["blue", "green"])

    fig.map(sns.distplot,c)

    fig.add_legend(labels=['Non Fraud','Fraud'])

    plt.axvline(col_mean,color='red',label='mean')

    plt.axvline(col_median,color='yellow',label='median')

    plt.legend()

    plt.show()
ccfr.columns
for feature in ccfr.drop('Class', axis= 1).columns:

    sns.boxplot(x='Class', y= feature, data= ccfr)

    plt.show()
X = ccfr.drop('Class', axis= 1)

y = ccfr['Class']
scaler = StandardScaler()

scaled_features = scaler.fit_transform(X.values)

X_scaled = pd.DataFrame(scaled_features, columns= X.columns)
X_scaled.head()
X_scaled.describe()
ccfr_scaled = pd.concat([X_scaled, y], axis= 1)
X = ccfr_scaled.drop('Class', axis= 1)

y = ccfr_scaled['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.13, random_state= 48)
X_train.shape
y_train.value_counts()
y_test.value_counts()
train = pd.concat([X_train, y_train], axis =1)
under_sampler = NearMiss(sampling_strategy= {0:100000, 1:410}) 

#under sampling the majority to 80000 records keeping minority as it is

X_train, y_train = under_sampler.fit_sample(X_train, y_train)
over_sampler = SMOTE(sampling_strategy= {0:100000, 1:10000}, random_state= 48)

#over sampling minority class to 20000 records

X_train, y_train = over_sampler.fit_sample(X_train, y_train)
train_sampled = pd.concat([X_train, y_train], axis= 1)

#it is good practice to shuffle the training set

train_sampled = train_sampled.sample(frac=1).reset_index(drop= True)

X_train = train_sampled.drop('Class', axis= 1)

y_train = train_sampled['Class']
X_train.shape
y_train.value_counts()
X_test.shape
y_test.value_counts()
sns.countplot(y_train)

plt.show()
knn_model = KNeighborsClassifier(n_neighbors= 4, n_jobs= -1)

knn_model.fit(X_train, y_train)

pred = knn_model.predict(X_test)

print ("ROC AUC Score=",roc_auc_score(y_test, pred))

print ("Classification Report:")

print (classification_report(y_test, pred))
cm = pd.DataFrame(confusion_matrix(y_test, pred))

plt.figure()

plt.title("Confusion Matrix for KNN")

sns.heatmap(cm, cmap= "Blues", annot= True, fmt= "d")

plt.xlabel("Predicted Classes")

plt.ylabel("True Classes")

plt.show()
lr_model= LogisticRegression(solver= 'liblinear')

lr_model.fit(X_train,y_train)

pred = lr_model.predict(X_test)

print ("ROC AUC Score=",roc_auc_score(y_test, pred))

print ("Classification Report:")

print (classification_report(y_test, pred))
cm = pd.DataFrame(confusion_matrix(y_test, pred))

plt.figure()

plt.title("Confusion Matrix for LR")

sns.heatmap(cm, cmap= "Blues", annot= True, fmt= "d")

plt.xlabel("Predicted Classes")

plt.ylabel("True Classes")

plt.show()
svc_model = svm.SVC(kernel='rbf', gamma= 0.03, C= 1.0)

svc_model.fit(X_train, y_train)

pred= svc_model.predict(X_test)
print ("ROC AUC Score=",roc_auc_score(y_test, pred))

print ("Classification Report:")

print (classification_report(y_test, pred))
cm = pd.DataFrame(confusion_matrix(y_test, pred))

plt.figure()

plt.title("Confusion Matrix for SVM")

sns.heatmap(cm, cmap= "Blues", annot= True, fmt= "d")

plt.xlabel("Predicted Classes")

plt.ylabel("True Classes")

plt.show()
voting_clf_hard = VotingClassifier(estimators = [('lr', lr_model), ('knn', knn_model)], voting = 'hard')

voting_clf_hard.fit(X_train, y_train)

pred= voting_clf_hard.predict(X_test)

print ("ROC AUC Score=",roc_auc_score(y_test, pred))

print ("Classification Report:")

print (classification_report(y_test, pred))
cm = pd.DataFrame(confusion_matrix(y_test, pred))

plt.figure()

plt.title("Confusion Matrix for LR + KNN Voting Classifier (Hard Voting)")

sns.heatmap(cm, cmap= "Blues", annot= True, fmt= "d")

plt.xlabel("Predicted Classes")

plt.ylabel("True Classes")

plt.show()
voting_clf_soft = VotingClassifier(estimators = [('lr', lr_model), ('knn', knn_model)], voting = 'soft',

                                  weights= [0.7, 0.3])

voting_clf_soft.fit(X_train, y_train)

pred= voting_clf_soft.predict(X_test)

print ("ROC AUC Score=",roc_auc_score(y_test, pred))

print ("Classification Report:")

print (classification_report(y_test, pred))
cm = pd.DataFrame(confusion_matrix(y_test, pred))

plt.figure()

plt.title("Confusion Matrix for LR + KNN Voting Classifier (Soft Voting)")

sns.heatmap(cm, cmap= "Blues", annot= True, fmt= "d")

plt.xlabel("Predicted Classes")

plt.ylabel("True Classes")

plt.show()
dctree_model = DecisionTreeClassifier(random_state=0)

dctree_model.fit(X_train, y_train)

pred= dctree_model.predict(X_test)

print ("ROC AUC Score=",roc_auc_score(y_test, pred))

print ("Classification Report:")

print (classification_report(y_test, pred))
cm = pd.DataFrame(confusion_matrix(y_test, pred))

plt.figure()

plt.title("Confusion Matrix for Decision Tree")

sns.heatmap(cm, cmap= "Blues", annot= True, fmt= "d")

plt.xlabel("Predicted Classes")

plt.ylabel("True Classes")

plt.show()
sgd_model = SGDClassifier(class_weight = 'balanced', learning_rate = 'adaptive', n_jobs = -1, eta0 = 0.001, 

                          max_iter = 100000)

sgd_model.fit(X_train, y_train)

pred = sgd_model.predict(X_test)

print ("ROC AUC Score=",roc_auc_score(y_test, pred))

print ("Classification Report:")

print (classification_report(y_test, pred))
cm = pd.DataFrame(confusion_matrix(y_test, pred))

plt.figure()

plt.title("Confusion Matrix for SGD")

sns.heatmap(cm, cmap= "Blues", annot= True, fmt= "d")

plt.xlabel("Predicted Classes")

plt.ylabel("True Classes")

plt.show()
bagging_clf = BaggingClassifier(DecisionTreeClassifier(random_state = 0), n_estimators = 1000, bootstrap = True,

                               max_samples = 0.85, n_jobs = -1, oob_score = True)

#bootsrap True signifies sampling (max_samples) from the data without replacement for each estimator

#oob_score True enables training on set of samples chosen and test on the out-of-bag samples(samples not chosen for training)

bagging_clf.fit(X_train, y_train)

pred = bagging_clf.predict(X_test)

print ("ROC AUC Score=",roc_auc_score(y_test, pred))

print ("Classification Report:")

print (classification_report(y_test, pred))
cm = pd.DataFrame(confusion_matrix(y_test, pred))

plt.figure()

plt.title("Confusion Matrix for Bagging Classifier")

sns.heatmap(cm, cmap= "Blues", annot= True, fmt= "d")

plt.xlabel("Predicted Classes")

plt.ylabel("True Classes")

plt.show()
lgb_model = LGBMClassifier(boosting_type = 'gbdt', num_leaves = 32, max_depth = 7, learning_rate = 0.007, 

                           n_estimators = 3500, objective = 'binary', min_split_gain = 0.1, min_child_weight = 0.01,

                           class_weight= {0:0.2, 1:1},

                           min_child_samples = 20, subsample=0.6, colsample_bytree = 0.8, reg_alpha = 0.3, reg_lambda = 0.7,

                           n_jobs = -1, verbose = -1)

history = {}

eval_history = record_evaluation(history)

lgb_model.fit(X_train, y_train,

             eval_set = [(X_train, y_train), (X_test, y_test)],

             eval_metric = 'auc', verbose = 500,

             callbacks = [eval_history])

pred = lgb_model.predict(X_test)

print ("ROC AUC Score=",roc_auc_score(y_test, pred))

print ("Classification Report:")

print (classification_report(y_test, pred))
train_aucs = history['training']['auc']

test_aucs = history['valid_1']['auc']

plt.figure(figsize = (8,6))

plt.ylim([0,1.01])

plt.plot(train_aucs, color= 'r', label= 'training')

plt.plot(test_aucs, color= 'g', label= 'testing')

plt.xlabel("No. of Estimators")

plt.ylabel('AUC Scores')

plt.legend(loc= 'best')

plt.title("LightGBM Performance with n_estimators chosen")

plt.show()
train_logloss = history['training']['binary_logloss']

test_logloss = history['valid_1']['binary_logloss']

plt.figure(figsize = (8,6))

plt.ylim([0,1.01])

plt.plot(train_logloss, color= 'r', label= 'training')

plt.plot(test_logloss, color= 'g', label= 'testing')

plt.xlabel("No. of Estimators")

plt.ylabel('Binary Log Loss')

plt.legend(loc= 'best')

plt.title("LightGBM Loss Minimization")

plt.show()
cm = pd.DataFrame(confusion_matrix(y_test, pred))

plt.figure()

plt.title("Confusion Matrix for LightGBM")

sns.heatmap(cm, cmap= "Blues", annot= True, fmt= "d")

plt.xlabel("Predicted Classes")

plt.ylabel("True Classes")

plt.show()
lgb_model = LGBMClassifier(boosting_type = 'gbdt', num_leaves = 30, max_depth = 7, learning_rate = 0.007, 

                           n_estimators = 4500, objective = 'binary', min_split_gain = 0.1, min_child_weight = 0.01,

                           class_weight= {0:0.2, 1:1},

                           min_child_samples = 20, subsample=0.6, colsample_bytree = 0.8, reg_alpha = 0.3, reg_lambda = 0.7,

                           n_jobs = -1, verbose = -1)



cv = KFold(n_splits = 10, random_state = 48, shuffle = True)



TP = 0 #TruePositives

TN = 0 #TrueNegatives

FP = 0 #FalsePositives

FN = 0 #FalseNegatives

roc_auc_scores = []



x1 = X #taking the original scaled data post train test split

y1 = y



for train_ind, test_ind in cv.split(x1):

    xtrain, xtest, ytrain, ytest= x1.loc[list(train_ind)], x1.loc[list(test_ind)], y1.loc[list(train_ind)], y1.loc[list(test_ind)]

    

    under_sampler_cv = NearMiss(sampling_strategy= {0:100000, 1:410}) 

    xtrain, ytrain = under_sampler_cv.fit_sample(xtrain, ytrain)

    

    over_sampler_cv = SMOTE(sampling_strategy= {0:100000, 1:10000}, random_state= 48)

    xtrain, ytrain = over_sampler_cv.fit_sample(xtrain, ytrain)

    

    train_set = pd.concat([xtrain, ytrain], axis= 1)

    train_set = train_set.sample(frac=1).reset_index(drop= True)

    xtrain = train_set.drop('Class', axis= 1)

    ytrain = train_set['Class']

    

    lgb_model.fit(xtrain, ytrain)

    prd = lgb_model.predict(xtest)

    true = np.array(ytest)

    l = len(prd)

    for i in range (l):

        if true[i]==1 and prd[i]==1:

            TP+=1

        if true[i]==1 and prd[i]==0:

            FN+=1

        if true[i]==0 and prd[i]==1:

            FP+=1

        if true[i]==0 and prd[i]==0:

            TN+=1

    roc_auc_scores.append(roc_auc_score(true, prd))
cm = pd.DataFrame([[TN, FP], [FN, TP]], index= [0,1], columns= [0,1])

plt.figure()

plt.title("Confusion Matrix for LightGBM with 10 Fold CV")

sns.heatmap(cm, cmap= "Blues", annot= True, fmt= "d")

plt.xlabel("Predicted Classes")

plt.ylabel("True Classes")

plt.show()
print ("% of Frauds predicted correctly=",(426/(66+426))*100)

print ("Average ROC AUC Score=",np.average(roc_auc_scores))
params = {'num_leaves' : [30, 40],

          'max_depth' : [7, 9],  

          'min_child_weight' : [0.01, 1],

          'subsample' : [0.6, 0.7],

          'colsample_bytree' : [0.8, 0.9],

          'reg_alpha' : [0.1, 0.3],

          'reg_lambda' : [0.1, 0.7]}

clf = GridSearchCV(lgb_model, params, scoring= 'recall', n_jobs= -1, cv= 2)

clf.fit(X_train, y_train)
clf.best_params_
lgb_model = LGBMClassifier(boosting_type = 'gbdt', num_leaves = 30, max_depth = 9, learning_rate = 0.007, 

                           n_estimators = 3500, objective = 'binary', min_split_gain = 0.1, min_child_weight = 0.01,

                           class_weight= {0:0.1, 1:1},

                           min_child_samples = 20, subsample=0.6, colsample_bytree = 0.9, reg_alpha = 0.1, reg_lambda = 0.7,

                           n_jobs = -1, verbose = -1)



cv = KFold(n_splits = 10, random_state = 48, shuffle = True)



TP = 0 #TruePositives

TN = 0 #TrueNegatives

FP = 0 #FalsePositives

FN = 0 #FalseNegatives

roc_auc_scores = []



x1 = X #taking the original scaled data post train test split

y1 = y



for train_ind, test_ind in cv.split(x1):

    xtrain, xtest, ytrain, ytest= x1.loc[list(train_ind)], x1.loc[list(test_ind)], y1.loc[list(train_ind)], y1.loc[list(test_ind)]

    

    under_sampler_cv = NearMiss(sampling_strategy= {0:100000, 1:410}) 

    xtrain, ytrain = under_sampler_cv.fit_sample(xtrain, ytrain)

    

    over_sampler_cv = SMOTE(sampling_strategy= {0:100000, 1:10000}, random_state= 48)

    xtrain, ytrain = over_sampler_cv.fit_sample(xtrain, ytrain)

    

    train_set = pd.concat([xtrain, ytrain], axis= 1)

    train_set = train_set.sample(frac=1).reset_index(drop= True)

    xtrain = train_set.drop('Class', axis= 1)

    ytrain = train_set['Class']

    

    lgb_model.fit(xtrain, ytrain)

    prd = lgb_model.predict(xtest)

    true = np.array(ytest)

    l = len(prd)

    for i in range (l):

        if true[i]==1 and prd[i]==1:

            TP+=1

        if true[i]==1 and prd[i]==0:

            FN+=1

        if true[i]==0 and prd[i]==1:

            FP+=1

        if true[i]==0 and prd[i]==0:

            TN+=1

    roc_auc_scores.append(roc_auc_score(true, prd))
cm = pd.DataFrame([[TN, FP], [FN, TP]], index= [0,1], columns= [0,1])

plt.figure()

plt.title("Confusion Matrix for LightGBM with 10 Fold CV")

sns.heatmap(cm, cmap= "Blues", annot= True, fmt= "d")

plt.xlabel("Predicted Classes")

plt.ylabel("True Classes")

plt.show()
print ("% of Frauds predicted correctly=",(430/(62+430))*100)

print ("Average ROC AUC Score=",np.average(roc_auc_scores))
import lightgbm

lightgbm.plot_importance(lgb_model, figsize= (12, 10))
pred = lgb_model.predict(X_test)

print ("ROC AUC Score=",roc_auc_score(y_test, pred))

print ("Classification Report:")

print (classification_report(y_test, pred))
cm = pd.DataFrame(confusion_matrix(y_test, pred))

plt.figure()

plt.title("Confusion Matrix for LightGBM")

sns.heatmap(cm, cmap= "Blues", annot= True, fmt= "d")

plt.xlabel("Predicted Classes")

plt.ylabel("True Classes")

plt.show()
from sklearn.metrics import roc_curve

fpr , tpr, threshold = roc_curve(y_test, pred)

roc_auc = roc_auc_score(y_test, pred)



plt.title('Receiver Operating Characteristics')

plt.plot(fpr, tpr, 'b', label= "AUC = %0.2f" % roc_auc)

plt.legend(loc = 'best')

plt.plot([0,1], [0,1], 'r--')

plt.xlim([0,1])

plt.ylim([0,1])

plt.xlabel("False Postive Rate")

plt.ylabel("True Positive Rate")

plt.show()