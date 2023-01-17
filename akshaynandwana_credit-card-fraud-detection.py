# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold,StratifiedKFold, cross_validate, train_test_split, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve, auc
from sklearn.metrics import plot_confusion_matrix
from imblearn.over_sampling import SMOTE

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.wrappers.scikit_learn import KerasClassifier
import re
import pandas as pd
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Input, Dropout
from keras import Sequential

import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
print(df.shape)
df.head(3)
print(df.columns)
df.describe()
df.Time = (df.Time - df.Time.mean())/df.Time.std()
df.Amount = (df.Amount - df.Amount.mean())/df.Amount.std()

#features and targets
X = df.iloc[:,:-1]
y = df.iloc[:,-1:]
# distribution plots for our principal components
# shows each PC is approximately normal and centered

f, axs = plt.subplots(7, 4,figsize=(20,20))
axs= axs.flatten().tolist()
for pc in df.columns[1:-2]:
    ax = axs.pop(0)
    sns.distplot(df[pc],ax=ax)
    ax.set_title(pc)
    
    ax.tick_params(
    which='both',
    bottom='off',
    left='off',
    right='off',
    top='off'
    )
    
    ax.grid(linewidth=0.25)
    ax.set_xlabel("")
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
plt.show()
fraud_n = np.sum(df.Class==1)
legit_n = df.shape[0] - fraud_n
print("The total number of fraudulent transactions in our data set is: " + str(fraud_n))
print("The total number of legitimate transactions in our data set is: " + str(df.shape[0] - np.sum(df.Class==1)))
print("")
print("The percentage of frauds in our entire data set is: " + str(np.round(fraud_n/df.shape[0],5)*100)+"%")
#simple logistic classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=77)
model = LogisticRegression()
model.fit(X_train,y_train)
model.score(X_test,y_test)
X_train.shape
print("number of correctly classifed frauds in test data: " + str(\
    np.sum((model.predict(X_test)==1)*np.array((y_test==1)).reshape(1,-1))))
print("total number of frauds in test data: " + str(np.sum(y_test,axis=0)[0]))
print("")
print(classification_report(y_test, model.predict(X_test)))

cm=plot_confusion_matrix(model,X_test,y_test,normalize='true',cmap=plt.cm.viridis)
cm.ax_.set_title("Confusion Matrix. 0: Legitimate, 1: Fraud")
plt.show()
fpr, tpr, threshold = roc_curve(y_test, model.predict_proba(X_test)[:,1])
auc_score = auc(fpr,tpr)
plt.title('ROC Curve for naive classifer')
plt.plot(fpr, tpr, 'b',label="ROC")
plt.legend()
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.0011, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3,stratify=y, random_state=77)

print("# of frauds in our test set is: " + str(np.sum(y_test)[0]))
print("# of legitimate cases in our test set is: " + str(y_test.shape[0] - np.sum(y_test)[0]))
print("the percentage of frauds in our test set is: " + str(round(100*np.sum(y_test)[0]/len(y_test),3))+"%")

print("")

print("# of frauds in our training set: " + str((np.sum(y_train)[0])))
print("# of legitimate in our training set: " + str(y_train.shape[0] - (np.sum(y_train)[0])))
print("the percentage of frauds in our training sample is: " + str(round(100*np.sum(y_train)[0]/len(y_train),3))+"%")
def undersample(X_train,y_train,rs=None):
    """returns *BALANCED* tuple of independent variables
    and their associated targets from the training set"""
    #bootstraps index of 328 legitimate cases
    legit_index = y_train[y_train.Class == 0].sample(328,replace=True,random_state=rs).index 
    #all 328 fraud indices from training set 
    fraud_index = y_train[y_train['Class']==1].index 
    
    X_legit = X_train.loc[legit_index]
    y_legit = y_train.loc[legit_index]
    X_fraud = X_train.loc[fraud_index]
    y_fraud = y_train.loc[fraud_index]
    return (pd.concat([X_legit,X_fraud]),pd.concat([y_legit,y_fraud]))
#Let's see if our function does what it's supposed to
U = undersample(X_train,y_train)
sns.countplot('Class',data=U[1])
plt.title("Distribution of labels within each undersample")
plt.show()

print("the ratio of frauds within each undersample is: " + str(np.sum(U[1])[0]/U[1].shape[0]))
U = undersample(X_train,y_train,rs=77) #defined undersample from training set
#LDA:
LDA_params = {'solver': ['svd','lsqr','eigen'], 'shrinkage': np.arange(0.1,1.01,step=0.01)}
g_LDA = GridSearchCV(LinearDiscriminantAnalysis(), LDA_params).fit(U[0],U[1])
LDA = g_LDA.best_estimator_

#QDA:
QDA_params = {'reg_param':np.arange(0.1,10.1,step=0.1)}
g_QDA = GridSearchCV(QuadraticDiscriminantAnalysis(), QDA_params).fit(U[0],U[1])
QDA = g_QDA.best_estimator_

#KNN:
KNN_params = {"n_neighbors": np.arange(1,10,step=1), 'weights': ['uniform','distance'] ,'p': [1,2]}
g_KNN = GridSearchCV(KNeighborsClassifier(), KNN_params).fit(U[0],U[1])
KNN = g_KNN.best_estimator_
#Logistic Regression:
LR_params = {'penalty': ['l1','l2','elasticnet'], 'C': np.arange(0.01,1,step=0.03),'dual': [False,True],
             'solver': ['newton-cg', 'lbfgs', 'saga']}
g_LR = GridSearchCV(LogisticRegression(), LR_params).fit(U[0],U[1])
LR = g_LR.best_estimator_
#Decision Tree:
DT_params = {'criterion': ['gini','entropy'], 'max_depth': [None,5,10,25,50], 'min_samples_split': [2,3,4,5],
            'min_samples_leaf': [1,2,10,20,25]}
g_DT = GridSearchCV(DecisionTreeClassifier(), DT_params).fit(U[0],U[1])
DT = g_DT.best_estimator_

#Random Forest: 
RF_params = {'n_estimators':[100], 'max_depth': [None,5,10], 'min_samples_split': [2,3],
            'min_samples_leaf': [1,2,3],'max_features': ['sqrt','log2'],'random_state':[77]}
g_RF = GridSearchCV(RandomForestClassifier(), RF_params).fit(U[0],U[1])
RF = g_RF.best_estimator_
#SVM:
SVM_params = {'C': np.arange(0.01,2,step=0.05), 'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
             'gamma': ['scale','auto'],'probability':[True]}
g_SVM = GridSearchCV(SVC(), SVM_params).fit(U[0],U[1])
SVM = g_SVM.best_estimator_
#all our previous classifiers
clfs = {'Naive Logistic Reg (no undersampling)': model, 'LDA':LDA, 'QDA': QDA, 'KNN':KNN, 
        'Logistic Regression': LR, 'Decision Tree': DT, 'Random Forest': RF, 'SVM':SVM}
#details of our classifiers after grid search
for key,clf in clfs.items():
    print('params for ' +str(key) +": " + str(clf))
    print("")
f, axs = plt.subplots(4, 2, figsize=(10,18))
axs= axs.flatten().tolist()
for key,clf in clfs.items():
    ax = axs.pop(0)
    cm=plot_confusion_matrix(clf,X_test,y_test,normalize='true',ax=ax,cmap='viridis')
    ax.set_title(key)
plt.show()
def plot_rocs(clfs,y_bound=0,x_bound=1,omit=[],title="",size=(8,8)):
    """takes in a dict of classifiers and y axis lower bound.
    omit is a list of the classifiers key to omit.
    returns plot of ROC and associated AUC scores"""
    plt.figure(figsize=size)
    for key,clf in clfs.items():
        if key in omit:
            continue
        else:
            fpr, tpr, threshold = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
            auc_score = auc(fpr,tpr)
            plt.title(title)
            plt.plot(fpr, tpr,label=key+ " AUC: "+ str(np.round(auc_score,3)))
            plt.legend()
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, x_bound])
            plt.ylim([y_bound, 1.001])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
    plt.show()
plot_rocs(clfs,title = "ROC for our classifiers")
plot_rocs(clfs,y_bound=0.8,x_bound=1,size=(10,10), title = "ROC for our classifiers")
#bagging within the undersample
bagged_clfs = {}
for key,clf in clfs.items():
    if clf==model: #naive model, bag on X_train 
        bagged_clfs[key] = BaggingClassifier(clf,100).fit(X_train,y_train) 
    else: #undersample models, bag on our undersample
         bagged_clfs[key] = BaggingClassifier(clf,100).fit(U[0],U[1])
conf=plot_confusion_matrix(bagged_clfs["Decision Tree"],X_test,y_test,normalize='true',ax=ax,cmap='viridis')
plot_rocs(bagged_clfs,y_bound=0.8,omit=["KNN"],title= "ROC for our *BAGGED* classifiers")
conf=plot_confusion_matrix(bagged_clfs["Decision Tree"],X_test,y_test,normalize='true',ax=ax,cmap='viridis')
conf.plot()
plt.title('bagged Decision Tree')
plt.show()
from xgboost import XGBClassifier
#training on original 
xgb = XGBClassifier(random_state=77).fit(X_train,y_train)
#training on the undersample
xgb_u = XGBClassifier(random_state=77).fit(U[0],U[1])

d = {'XGBOOST (on original train)':xgb,'XGBOOST (on undersample)': xgb_u}
dic = {
    'XGBOOST (on original train)':xgb,'XGBOOST (on undersample)': xgb_u,
    'Naive Logistic Reg (no undersampling)': model,'LDA': LDA, 'Random Forest': RF, 
    'bagged DT':bagged_clfs['Decision Tree'] 
      }
plot_rocs(dic,0.8,title = "XGBOOST classifier ROC")
f, axs = plt.subplots(1, 2, figsize=(10,6))
axs= axs.flatten().tolist()
for key,clf in d.items():
    ax = axs.pop(0)
    cm=plot_confusion_matrix(clf,X_test,y_test,normalize='true',ax=ax,cmap='viridis')
    ax.set_title(key)
plt.show()
n_inputs = U[0].shape[1]
#function creates a Keras Neural network based on provided inputs
def create_model(dense_layer_sizes, epochs = 10, optimizer="adam", dropout=0.1, 
                 init='uniform', dense_nparams=256, batch_size = 2):
    model = Sequential()
    model.add(Dense(dense_nparams, activation='relu', input_shape=(n_inputs,), kernel_initializer=init,)) 
    model.add(Dropout(dropout), )
    for layer_size in dense_layer_sizes*[dense_nparams]:
        model.add(Dense(layer_size, activation='relu'))
        model.add(Dropout(dropout), )
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=["accuracy"])
    return model
keras_estimator = KerasClassifier(build_fn=create_model, verbose=1)
param_grid = {
    'dense_layer_sizes' : [1,2,3],
    'epochs': [50],
    'dense_nparams': [32, 64],
    'init': [ 'uniform'], 
    'batch_size':[4],
    'optimizer':['Adam'],
    'dropout': [0.2, 0.1,0.05, 0]
}
kfold_splits = 5
grid = GridSearchCV(estimator=keras_estimator,  
                    n_jobs=-1, 
                    verbose=1,
                    return_train_score=True,
                    cv=kfold_splits,  #StratifiedKFold(n_splits=kfold_splits, shuffle=True)
                    param_grid=param_grid,)
grid.fit(U[0],U[1])
nn_best = grid.best_estimator_
nn_best.get_params()
#for confusion matrix
x = nn_best.predict(X_test)
skplt.metrics.plot_confusion_matrix(
    y_test, 
    x,
    figsize=(4,4),normalize=True,title='Neural Net Confusion Matrix',cmap='Blues')
plt.show()
dic = {
    'Neural (on undersample)':nn_best,'XGBOOST (on original training)':xgb
      }
plot_rocs(dic,0.8,title="Neural Net and XGBOOST ROCs",size=(5,5))
