# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import re



import seaborn as sns

from matplotlib import pyplot as plt



from sklearn.preprocessing import StandardScaler

from sklearn import metrics

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score

from sklearn.model_selection import train_test_split



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.dummy import DummyClassifier



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from xgboost import XGBClassifier



# from imblearn.over_sampling import SMOTE

# from imblearn.under_sampling import RandomUnderSampler
# set plot rc parameters



# jtplot.style(grid=False)

plt.rcParams['figure.facecolor'] = 'white'

plt.rcParams['axes.facecolor'] = '#232323'

#plt.rcParams['axes.edgecolor'] = '#FFFFFF'

plt.rcParams['figure.figsize'] = 10, 7

plt.rcParams['legend.loc'] = 'best'

plt.rcParams['legend.framealpha'] = 0.2

plt.rcParams['text.color'] = '#666666'

plt.rcParams['axes.labelcolor'] = '#666666'

plt.rcParams['axes.labelsize'] = 14

plt.rcParams['axes.titlesize'] = 16

plt.rcParams['xtick.color'] = '#666666'

plt.rcParams['xtick.labelsize'] = 14

plt.rcParams['ytick.color'] = '#666666'

plt.rcParams['ytick.labelsize'] = 14



# plt.rcParams['font.size'] = 16



sns.color_palette('dark')

%matplotlib inline
# load data and print first five data points



dftrain = pd.read_csv('../input/titanic/train.csv')

dftest = pd.read_csv('../input/titanic/test.csv')

dftrain.head()
# get dimentions of data table



dftrain.shape
# meta data



dftrain.info()
# meta data



dftest.info()
# let's look at unique values for each attribute



for col in dftrain.columns:

    print('Unique values in {}: {}'.format(col, len(dftrain[col].value_counts())))
# let's look at unique values for each attribute



for col in dftest.columns:

    print('Unique values in {}: {}'.format(col, len(dftest[col].value_counts())))
dftrain.describe()
dftest.describe()
# Distribution of categorical features



fig, axs = plt.subplots(2,2, figsize=(15,12))

axs = axs.flatten()

cols = ['Survived', 'Sex', 'Pclass', 'Embarked']

for idx, ax in enumerate(axs):

    sns.countplot(data=dftrain, x=cols[idx], ax=ax)
# Survival w.r.p.t gender and passenger class



fig, axs = plt.subplots(1,2, figsize=(15,8))

axs = axs.flatten()

sns.countplot(data=dftrain, x='Sex', hue='Survived', ax=axs[0])

sns.countplot(data=dftrain, x='Pclass', hue='Survived', ax=axs[1])



plt.show()
# Distribution of age and fare w.r.p.t survival 



fig, axs = plt.subplots(1,2, figsize= (15, 8))

axs = axs.flatten()

sns.violinplot(data=dftrain, x='Sex', y='Age', hue='Survived', ax=axs[0], split=True)

sns.violinplot(data=dftrain, x='Sex', y='Fare', hue='Survived', ax=axs[1], split=True)



plt.show()
# Distribution of age and fare w.r.p.t survival 



fig, axs = plt.subplots(1,2, figsize= (15, 8))

axs = axs.flatten()

sns.violinplot(data=dftrain, x='Pclass', y='Age', hue='Sex', ax=axs[0], split=True)

sns.violinplot(data=dftrain, x='Pclass', y='Fare', hue='Sex', ax=axs[1], split=True)



plt.show()
# impute age

dftrain['Age'] = dftrain['Age'].fillna(dftrain['Age'].median())

dftest['Age'] = dftest['Age'].fillna(dftrain['Age'].median())



# impute embarked

dftrain['Embarked'] = dftrain['Embarked'].fillna('S')



# impute fare

dftest['Fare'] = dftest['Fare'].fillna(dftrain['Fare'].median())
dftrain.info(), dftest.info()
# one hot encode train data

dftrain = pd.concat([dftrain,

                    pd.get_dummies(dftrain['Sex']),

                    pd.get_dummies(dftrain['Embarked'])], axis=1)

# one hot encode test data

dftest = pd.concat([dftest,

                    pd.get_dummies(dftest['Sex']),

                    pd.get_dummies(dftest['Embarked'])], axis=1)
dftrain.head()
dftrain.shape, dftest.shape
# passenger had cabin or not

dftrain['Cabin_bool'] = dftrain['Cabin'].isna().astype(int)

dftest['Cabin_bool'] = dftest['Cabin'].isna().astype(int)
# drop non numeric columns

dropcols = ['PassengerId', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']

trainid = dftrain['PassengerId']

testid = dftest['PassengerId']

dftrain.drop(dropcols, inplace=True, axis=1)

dftest.drop(dropcols, inplace=True, axis=1)

dftrain.head()
dftest.head()
# x and y for training

X = dftrain.drop('Survived', axis=1)

Y = dftrain['Survived']
cols = X.columns
# scaler instances

train_scaler = StandardScaler()

test_scaler = StandardScaler()

# scale data

X = train_scaler.fit_transform(X)

dftest = test_scaler.fit_transform(dftest)

# add columns

X = pd.DataFrame(X, columns=cols)

dftest = pd.DataFrame(dftest, columns=cols)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.25, random_state=12)
Xcorr = X.corr()



# plot correlatation matrix

plt.figure(figsize=(16, 16))

g = sns.heatmap(Xcorr,

            cbar = True,

            square = True,

            linewidth=0.3,

            fmt='.2f')

plt.title('Feature correlation',

          color='#666666',

          fontdict={'fontsize': 22})

plt.show()
# PCA for visualization

pca = PCA(n_components = 2)

pca.fit(Xtrain)

X_pca = pca.transform(Xtrain)



# plot PCA components

fig, axs = plt.subplots(figsize=[10,9])

sns.scatterplot(x=X_pca[:,0],

                y=X_pca[:,1],

                hue=Ytrain,

                palette=[sns.xkcd_rgb['red pink'],

                         sns.xkcd_rgb['greenish cyan']],

                edgecolor=None,

                ax=axs)

plt.xlabel('PC1')

plt.ylabel('PC2')

plt.title('Scatter plot based on survival')

plt.show()
# Function to print model performance summary statistics



def performance_summary(model, Xtrain, Xtest, Ytrain, Ytest):

    

    Ytrain_pred = model.predict(Xtrain)

    Ytest_pred = model.predict(Xtest)



    # model performance

    # accuracy score

    print('Training Accuracy:\n', accuracy_score(Ytrain, Ytrain_pred))

    print('\n')

    print('Test Accuracy:\n', accuracy_score(Ytest, Ytest_pred))

    print('\n')

    # classification report

    print('Classification Report training:\n', classification_report(Ytrain,Ytrain_pred))

    print('\n')

    print('Classification Report test:\n', classification_report(Ytest,Ytest_pred))

    

    return
# Function to plot Confusion matrix



def plot_confusion_matrix(model, Xtrain, Xtest, Ytrain, Ytest):

    

    Ytrain_pred = model.predict(Xtrain)

    Ytest_pred = model.predict(Xtest)



    # confusion matrix

    fig, axs = plt.subplots(1,2,

                            figsize=[15,5])

    axs = axs.flatten()

    

    axs[0].set_title('Training data')

    # axs[0].set_xlabel('Predicted label')

    # axs[0].set_ylabel('True label')

    axs[1].set_title('Test data')

    # axs[1].set_xlabel('Predicted label')

    # axs[1].set_ylabel('True label')

    

    fig.text(0.27, 0.04, 'Predicted label', ha='center')

    fig.text(0.70, 0.04, 'Predicted label', ha='center')

    fig.text(0.04, 0.5, 'True label', va='center', rotation='vertical')

    fig.text(0.5, 0.5, 'True label', va='center', rotation='vertical')

    

    sns.heatmap(confusion_matrix(Ytrain,Ytrain_pred),

                    annot=True,

                    xticklabels=['dpd < 30', 'dpd > 30'],

                    yticklabels=['dpd < 30', 'dpd > 30'],

                    fmt="d",

                    ax=axs[0])

    

    sns.heatmap(confusion_matrix(Ytest,Ytest_pred),

                    annot=True,

                    xticklabels=['dpd < 30', 'dpd > 30'],

                    yticklabels=['dpd < 30', 'dpd > 30'],

                    fmt="d",

                    ax=axs[1])

    plt.show()

    

    return
# Function to plot ROC



def plot_roc(model, Xtrain, Xtest, Ytrain, Ytest):

    # ROC curve and area under ROC curve



    # get FPR and TPR for training and test data

    Ytrain_pred_proba = model.predict_proba(Xtrain)

    fpr_train, tpr_train, thresholds_train = roc_curve(Ytrain, Ytrain_pred_proba[:,1])

    # tpr fpr are swapped 

    roc_auc_train = auc(fpr_train, tpr_train)

    Ytest_pred_proba = model.predict_proba(Xtest)

    fpr_test, tpr_test, thresholds_test = roc_curve(Ytest, Ytest_pred_proba[:,1])

    # tpr fpr are swapped

    roc_auc_test = auc(fpr_test, tpr_test)



    # print area under roc curve

    print ('AUC_ROC train:\t', roc_auc_train)

    print ('AUC_ROC test:\t', roc_auc_test)



    # plot auc roc

    fig, axs = plt.subplots(1,2,

                            figsize=[15,5],

                            sharex=False,

                            sharey=False)

    

    # training data

    axs[0].set_title('Receiver Operating Characteristic trainning')

    axs[0].plot(fpr_train,

                tpr_train,

                sns.xkcd_rgb['greenish cyan'],

                label='AUC = %0.2f'% roc_auc_train)

    axs[0].legend(loc='lower right')

    

    axs[0].plot([0,1],[0,1],

                ls='--',

                c=sns.xkcd_rgb['red pink'])

    

    axs[0].set_xlim([-0.01,1.01])

    axs[0].set_ylim([-0.01,1.01])

    axs[0].set_ylabel('True Positive Rate')

    axs[0].set_xlabel('False Positive Rate')

    

    # test data

    axs[1].set_title('Receiver Operating Characteristic testing')

    axs[1].plot(fpr_test,

                tpr_test,

                sns.xkcd_rgb['greenish cyan'],

                label='AUC = %0.2f'% roc_auc_test)

    axs[1].legend(loc='lower right')

    

    axs[1].plot([0,1],[0,1],

                ls='--',

                c=sns.xkcd_rgb['red pink'])

    

    axs[1].set_xlim([0.0,1.0])

    axs[1].set_ylim([0.0,1.0])

    axs[1].set_ylabel('True Positive Rate')

    axs[1].set_xlabel('False Positive Rate')



    plt.show()

    

    return
# hyper parameter tuning

lr_clf = GridSearchCV(LogisticRegression(penalty='l1', solver='saga', l1_ratio=1),

                      cv=5,

                      param_grid={'C': [0.001,0.01,0.1,1,10,100,1000]},

                      scoring='accuracy')



# fit to data

lr_clf.fit(Xtrain, Ytrain)
lr_clf.best_params_, lr_clf.best_score_
# Train model Logistic Regression

logreg_model = LogisticRegression(penalty='l1',solver='saga', C=0.1)

logreg_model.fit(Xtrain, Ytrain)



# performance summary

performance_summary(logreg_model, Xtrain, Xtest, Ytrain, Ytest)



# confusion matrix

plot_confusion_matrix(logreg_model, Xtrain, Xtest, Ytrain, Ytest)



# ROC plot

plot_roc(logreg_model, Xtrain, Xtest, Ytrain, Ytest)
svm_params = {'kernel': ('linear', 'rbf'),

             'C': [0.001,0.01,0.1,1,10,100,1000]}



svm_clf = GridSearchCV(estimator=SVC(),

                      cv=5,

                      param_grid=svm_params,

                      scoring='recall',

                      n_jobs=-1)



svm_clf.fit(Xtrain,Ytrain)
svm_clf.best_params_, svm_clf.best_score_
svm_model = SVC(kernel='rbf',

                C=1000,

                probability=True)

svm_model.fit(Xtrain, Ytrain)



# performance summary

performance_summary(svm_model, Xtrain, Xtest, Ytrain, Ytest)



# confusion matrix

plot_confusion_matrix(svm_model, Xtrain, Xtest, Ytrain, Ytest)



# ROC plot

plot_roc(svm_model, Xtrain, Xtest, Ytrain, Ytest)
# training model

xgb_model = XGBClassifier()

xgb_model.fit(Xtrain, Ytrain)



# performance summary

performance_summary(xgb_model, Xtrain, Xtest, Ytrain, Ytest)



# confusion matrix

plot_confusion_matrix(xgb_model, Xtrain, Xtest, Ytrain, Ytest)



# ROC plot

plot_roc(xgb_model, Xtrain, Xtest, Ytrain, Ytest)
# Grid search xgboost



gbm_params = {'subsample':[0.6,0.7,0.8],

             'colsample_bytree':[0.6,0.7,0.8],

             'learning_rate': [0.1,1,10],

             'n_estimators': [500,1000,1500],

             'reg_alpha': [1,3,5,9],

             'max_depth': [2,3,4,5],

             'gamma': [0.01,0.1,1,10],

             'min_child_weight': [0.01,0.5,1,3,5]}



clf_params = {'silent':False,

              'objective':'binary:logistic'}



gbm_clf = RandomizedSearchCV(XGBClassifier(**clf_params),

                             param_distributions=gbm_params,

                             scoring = 'accuracy',

                             cv = 5,

                             n_jobs = -1,

                             n_iter=100)



gbm_clf.fit(Xtrain, Ytrain)
gbm_clf.best_score_, gbm_clf.best_params_
# model params

param_xgb = {'n_estimators': 1000,

                'colsample_bytree': 0.8,

                'objective': 'binary:logistic',

                'max_depth': 5,

                'min_child_weight': 0.01,

                'learning_rate': 0.1,

                'subsample': 0.7,

                'gamma': 1,

                'reg_alpha': 1}



# training model

xgb_model = XGBClassifier(**param_xgb)

xgb_model.fit(Xtrain, Ytrain)



# performance summary

performance_summary(xgb_model, Xtrain, Xtest, Ytrain, Ytest)



# confusion matrix

plot_confusion_matrix(xgb_model, Xtrain, Xtest, Ytrain, Ytest)



# ROC plot

plot_roc(xgb_model, Xtrain, Xtest, Ytrain, Ytest)
# training model

rf_model = RandomForestClassifier()

rf_model.fit(Xtrain, Ytrain)



# performance summary

performance_summary(rf_model, Xtrain, Xtest, Ytrain, Ytest)



# confusion matrix

plot_confusion_matrix(rf_model, Xtrain, Xtest, Ytrain, Ytest)



# ROC plot

plot_roc(rf_model, Xtrain, Xtest, Ytrain, Ytest)
rf_param = {'bootstrap': [True, False],

            'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],

            'max_features': ['auto', 'sqrt'],

            'min_samples_leaf': [1, 2, 4],

            'min_samples_split': [2, 5, 10],

            'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]}



rf_clf = RandomizedSearchCV(estimator=RandomForestClassifier(),

                           cv=5,

                           param_distributions=rf_param,

                           scoring='recall',

                           n_jobs=-1,

                           n_iter=100)



rf_clf.fit(Xtrain,Ytrain)
rf_clf.best_params_, rf_clf.best_score_
# model params

param_rf = {'n_estimators': 1400,

           'bootstrap':False,

           'max_depth':40,

           'max_features':'sqrt',

           'min_samples_leaf':2,

           'min_samples_split':2}



# training model

rf_model = RandomForestClassifier(**param_rf)

rf_model.fit(Xtrain, Ytrain)



# performance summary

performance_summary(rf_model, Xtrain, Xtest, Ytrain, Ytest)



# confusion matrix

plot_confusion_matrix(rf_model, Xtrain, Xtest, Ytrain, Ytest)



# ROC plot

plot_roc(rf_model, Xtrain, Xtest, Ytrain, Ytest)
# Train model Logistic Regression

model = LogisticRegression(penalty='l1',solver='saga', C=0.1)

model.fit(X, Y)



# prediction on test data

Ypred = model.predict(dftest)
my_submission = pd.DataFrame([testid, Ypred], columns=['PassengerId', 'Survived'])

my_submission.to_csv('submission.csv',index=False)