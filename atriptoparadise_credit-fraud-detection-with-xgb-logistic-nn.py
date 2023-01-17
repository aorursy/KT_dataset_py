# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Imported Libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA, TruncatedSVD

import matplotlib.patches as mpatches

import time



# Classifier Libraries

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import collections





# Other Libraries

from imblearn.datasets import fetch_datasets

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

from imblearn.over_sampling import SMOTE

from imblearn.under_sampling import NearMiss

from imblearn.metrics import classification_report_imbalanced

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report

from collections import Counter

from sklearn.model_selection import KFold, StratifiedKFold

import warnings

warnings.filterwarnings("ignore")





df = pd.read_csv('../input/creditcard.csv')

df.head()
def feature_summary(df_fa):

    print('DataFrame shape')

    print('rows:',df_fa.shape[0])

    print('cols:',df_fa.shape[1])

    col_list=['Null','Unique_Count','Data_type','Max/Min','Mean','Std','Skewness','Sample_values']

    df=pd.DataFrame(index=df_fa.columns,columns=col_list)

    df['Null']=list([len(df_fa[col][df_fa[col].isnull()]) for i,col in enumerate(df_fa.columns)])

    #df['%_Null']=list([len(df_fa[col][df_fa[col].isnull()])/df_fa.shape[0]*100 for i,col in enumerate(df_fa.columns)])

    df['Unique_Count']=list([len(df_fa[col].unique()) for i,col in enumerate(df_fa.columns)])

    df['Data_type']=list([df_fa[col].dtype for i,col in enumerate(df_fa.columns)])

    for i,col in enumerate(df_fa.columns):

        if 'float' in str(df_fa[col].dtype) or 'int' in str(df_fa[col].dtype):

            df.at[col,'Max/Min']=str(round(df_fa[col].max(),2))+'/'+str(round(df_fa[col].min(),2))

            df.at[col,'Mean']=df_fa[col].mean()

            df.at[col,'Std']=df_fa[col].std()

            df.at[col,'Skewness']=df_fa[col].skew()

        df.at[col,'Sample_values']=list(df_fa[col].unique())

           

    return(df.fillna('-'))
feature_summary(df)
# Since most of our data has already been scaled we should scale the columns that are left to scale (Amount and Time)

from sklearn.preprocessing import StandardScaler, RobustScaler



# RobustScaler is less prone to outliers.



std_scaler = StandardScaler()

rob_scaler = RobustScaler()



df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))

df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))



df.drop(['Time','Amount'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import StratifiedKFold



print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')

print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')



X = df.drop('Class', axis=1)

y = df['Class']



sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)



for train_index, test_index in sss.split(X, y):

    print("Train:", train_index, "Test:", test_index)

    X_train, X_test = X.iloc[train_index], X.iloc[test_index]

    y_train, y_test = y.iloc[train_index], y.iloc[test_index]



# We already have X_train and y_train for undersample data thats why I am using original to distinguish and to not overwrite these variables.

# original_Xtrain, original_Xtest, original_ytrain, original_ytest = train_test_split(X, y, test_size=0.2, random_state=42)



# Check the Distribution of the labels
# Turn into an array

original_Xtrain = X_train.values

original_Xtest = X_test.values

original_ytrain = y_train.values

original_ytest = y_test.values



# See if both the train and test label distribution are similarly distributed

train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)

test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)

print('-' * 100)



print('Label Distributions: \n')

print(train_counts_label/ len(original_ytrain))

print(test_counts_label/ len(original_ytest))
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline



print('Length of X (train): {} | Length of y (train): {}'.format(len(original_Xtrain), len(original_ytrain)))

print('Length of X (test): {} | Length of y (test): {}'.format(len(original_Xtest), len(original_ytest)))



# List to append the score and then find the average

accuracy_lst = []

precision_lst = []

recall_lst = []

f1_lst = []

auc_lst = []



# Classifier with optimal parameters

# log_reg_sm = grid_log_reg.best_estimator_

log_reg_sm = LogisticRegression()



log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}



rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline



print('Length of X (train): {} | Length of y (train): {}'.format(len(original_Xtrain), len(original_ytrain)))

print('Length of X (test): {} | Length of y (test): {}'.format(len(original_Xtest), len(original_ytest)))



# List to append the score and then find the average

accuracy_lst = []

precision_lst = []

recall_lst = []

f1_lst = []

auc_lst = []



# Classifier with optimal parameters

# log_reg_sm = grid_log_reg.best_estimator_

log_reg_sm = LogisticRegression()



log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}



rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)
%%time

# Implementing SMOTE Technique 

# Cross Validating the right way



for train, test in sss.split(original_Xtrain, original_ytrain):

    pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_log_reg) 

    # SMOTE happens during Cross Validation not before..

    model = pipeline.fit(original_Xtrain[train], original_ytrain[train])

    best_est = rand_log_reg.best_estimator_

    prediction = best_est.predict(original_Xtrain[test])

    

    accuracy_lst.append(pipeline.score(original_Xtrain[test], original_ytrain[test]))

    precision_lst.append(precision_score(original_ytrain[test], prediction))

    recall_lst.append(recall_score(original_ytrain[test], prediction))

    f1_lst.append(f1_score(original_ytrain[test], prediction))

    auc_lst.append(roc_auc_score(original_ytrain[test], prediction))

    

print('---' * 35)

print('')

print("accuracy: {}".format(np.mean(accuracy_lst)))

print("precision: {}".format(np.mean(precision_lst)))

print("recall: {}".format(np.mean(recall_lst)))

print("f1: {}".format(np.mean(f1_lst)))

print('AUC: {}'.format(np.mean(auc_lst)))

print('---' * 35)
labels = ['No Fraud', 'Fraud']

smote_prediction = best_est.predict(original_Xtest)

print(classification_report(original_ytest, smote_prediction, target_names=labels))
import xgboost as xgb

from xgboost import XGBClassifier
# %%time

# GridSearchCV

# from sklearn.model_selection import GridSearchCV



# xgb_params = {"eta": [0.02,0.05,0.1], 

#              'min_child_weight': [50,100,200,300,360],

#              'subsample':[0.5,0.7,0.8,0.9],

#              "colsample_bytree":[0.3,0.5,0.7,0.9],

#              "max_depth":[8,10,12,15],

#              'objective':['binary:logistic',"reg:linear"]

#             }



#xgb_sm = XGBClassifier()

#grid_xgb = GridSearchCV(xgb_sm, xgb_params)

#grid_xgb.fit(X_train, y_train)

#get the best estimator

#best_est_grid = grid_xgb.best_estimator_
%%time

# RandomizedSearchCV

# fit model no training data



xgb_params = {"eta": [0.01,0.02,0.03,0.05,0.8,0.1], 

              'min_child_weight': [50,100,200,300,360,500],

              'subsample':[0.5,0.7,0.8,0.9,0.95,0.99],

              "colsample_bytree":[0.3,0.5,0.7,0.9,0.99],

              "max_depth":[6,8,10,12]

             }



xgb_sm = XGBClassifier()

rand_xgb = RandomizedSearchCV(xgb_sm, xgb_params, n_iter=6)

rand_xgb.fit(X_train,y_train)

best_est_ran = rand_xgb.best_estimator_
best_xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

       colsample_bytree=0.5, eta=0.8, gamma=0, learning_rate=0.1,

       max_delta_step=0, max_depth=12, min_child_weight=50, missing=None,

       n_estimators=100, n_jobs=1, nthread=None,

       objective='binary:logistic', random_state=0, reg_alpha=0,

       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,

       subsample=0.7)
%%time

# Implementing SMOTE Technique 

# Cross Validating the right way



for train, test in sss.split(original_Xtrain, original_ytrain):

    pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), best_xgb) 

    # SMOTE happens during Cross Validation not before..

    model = pipeline.fit(original_Xtrain[train], original_ytrain[train])

    prediction = best_xgb.predict(original_Xtrain[test])

    

    accuracy_lst.append(pipeline.score(original_Xtrain[test], original_ytrain[test]))

    precision_lst.append(precision_score(original_ytrain[test], prediction))

    recall_lst.append(recall_score(original_ytrain[test], prediction))

    f1_lst.append(f1_score(original_ytrain[test], prediction))

    auc_lst.append(roc_auc_score(original_ytrain[test], prediction))

    

print('---' * 35)

print('')

print("accuracy: {}".format(np.mean(accuracy_lst)))

print("precision: {}".format(np.mean(precision_lst)))

print("recall: {}".format(np.mean(recall_lst)))

print("f1: {}".format(np.mean(f1_lst)))

print('AUC: {}'.format(np.mean(auc_lst)))

print('---' * 35)
labels = ['No Fraud', 'Fraud']

smote_prediction = best_xgb.predict(original_Xtest)

print(classification_report(original_ytest, smote_prediction, target_names=labels))
import keras

from keras import backend as K

from keras.models import Sequential

from keras.layers import Activation

from keras.layers.core import Dense

from keras.optimizers import Adam

from keras.metrics import categorical_crossentropy



n_inputs = X_train.shape[1]



oversample_model = Sequential([

    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),

    Dense(32, activation='relu'),

    Dense(2, activation='softmax')])



# SMOTE Technique (OverSampling) After splitting and Cross Validating

sm = SMOTE(ratio='minority', random_state=42)

# Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)





# This will be the data were we are going to 

Xsm_train, ysm_train = sm.fit_sample(original_Xtrain, original_ytrain)
oversample_model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
oversample_model.fit(Xsm_train, ysm_train, validation_split=0.2, batch_size=300, epochs=20, shuffle=True, verbose=2)
oversample_predictions = oversample_model.predict(original_Xtest, batch_size=200, verbose=0)
oversample_fraud_predictions = oversample_model.predict_classes(original_Xtest, batch_size=200, verbose=0)
from sklearn.metrics import confusion_matrix

import itertools
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title, fontsize=14)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
oversample_smote = confusion_matrix(original_ytest, oversample_fraud_predictions)

actual_cm = confusion_matrix(original_ytest, original_ytest)



labels = ['No Fraud', 'Fraud']



fig = plt.figure(figsize=(16,8))



fig.add_subplot(221)

plot_confusion_matrix(oversample_smote, labels, title="OverSample (SMOTE) \n Confusion Matrix", cmap=plt.cm.Oranges)



fig.add_subplot(222)

plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)