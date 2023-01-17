# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import warnings

warnings.filterwarnings("ignore")

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import IsolationForest

from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,roc_curve,confusion_matrix,log_loss,precision_score,recall_score,auc

from sklearn.utils import resample

from plotly.subplots import make_subplots

import plotly.graph_objects as go

import plotly.figure_factory as ff

import plotly.figure_factory as ff

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")

data.head()
print(data.shape)

data.columns
data.describe()
data.isnull().sum().sum()
plt.figure(figsize=(18,9))

sns.heatmap(data.corr(),vmax = .8, square = True)

plt.title("Correlation between features")

plt.show()
print(data["Class"].value_counts())

plt.style.use('dark_background')

sns.countplot("Class",data=data,color="cyan")

plt.xlabel("Class")

plt.ylabel("Count")

plt.show()
# Let's look at the Amount and Time distributions

fig, ax = plt.subplots(1, 2, figsize=(20,10))

sns.distplot(data["Amount"],ax=ax[0],color="orange")

sns.distplot(data["Time"],ax=ax[1],color="cyan")

ax[0].set_title("Distribution of Transaction Amount")

ax[1].set_title("Distribution of Transaction Time")
fraud = data.loc[data['Class'] == 1]

no_fraud = data.loc[data['Class'] == 0]

fig, ax = plt.subplots(2, 1, figsize=(16,14))

sns.scatterplot(fraud["Amount"],fraud["Time"],ax=ax[0],color="yellow")

ax[0].set_title("Amount-Time Distrubition for Fraud",fontsize=16)

sns.scatterplot(no_fraud["Amount"],no_fraud["Time"],ax=ax[1],color="green")

ax[1].set_title("Amount-Time Distrubition for No-Fraud", fontsize=16)

plt.show()
#confusion matrix

def conf_matrix(actual, predicted):

    plt.figure(figsize=(14,8))

    cm = confusion_matrix(actual, predicted)

    sns.heatmap(cm, xticklabels=['predicted_negative', 'predicted_positive'], 

                yticklabels=['true_negative', 'true_positive'], annot=True,

                fmt='d', annot_kws={'fontsize':26}, cmap='Blues');

    accuracy = accuracy_score(actual,predicted)

    precision = precision_score(actual,predicted)

    recall = recall_score(actual,predicted)

    f1 = f1_score(actual,predicted)



    cm_results = {"accuracy":accuracy, "precision":precision, "recall":recall, "f1":f1}

    return cm_results





#ROC Curve

def plot_roc_curve(model,X,y):

    plt.figure(figsize=(14,8))

    y_pred_prob = model.predict_proba(X)[:,1]

    fpr, tpr, thresholds = roc_curve(y, y_pred_prob)

    plt.plot([0, 1], [0, 1], 'k--',color="red")

    plt.plot(fpr, tpr,label='ROC curve (area = %0.2f)' % roc_auc_score(y,y_pred_prob))

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()  

    

    

#Precision-Recall Curve

from sklearn.metrics import precision_recall_curve

def plot_pre_recall(model,X,y):

    probs=model.predict_proba(X)[:,1]

    precision, recall, thresholds = precision_recall_curve(y, probs)

    auc_score = auc(recall, precision) 

    plt.figure(figsize=(16,8))

    plt.plot([0, 1], [0.5, 0.5], linestyle='--')

    plt.plot(recall, precision, marker='.')

    plt.title('Precision-Recall Curve')

    plt.show()

    print('AUC: %.3f' % auc_score)

    
eps=0.001

data['Amount'] = np.log(data.pop('Amount')+eps)

X=data.drop("Class",1)

y=data["Class"]

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=42)

from sklearn.preprocessing import RobustScaler

rbst=RobustScaler()

X_train=rbst.fit_transform(X_train)

X_test=rbst.transform(X_test)
logreg = LogisticRegression().fit(X_train, y_train)

log_pred = logreg.predict(X_test)

print("Accuracy score of Logistic Regression on Imbalanced Data:",accuracy_score(y_test,log_pred))

print(classification_report(y_test,log_pred))

conf_matrix(y_test,log_pred)
plot_roc_curve(logreg,X_test,y_test)
plot_pre_recall(logreg,X_test,y_test)
dectree=DecisionTreeClassifier().fit(X_train,y_train)

dec_pred=dectree.predict(X_test)

print("Accuracy score of Decision Tree Classifier on Imbalanced Data:",accuracy_score(y_test,dec_pred))

print(classification_report(y_test,dec_pred))

conf_matrix(y_test,dec_pred)

plot_roc_curve(dectree,X=X_test,y=y_test)
plot_pre_recall(dectree,X_test,y_test)
data_majority = data[data.Class==0]

data_minority = data[data.Class==1]

 

# Upsample minority class

data_major_downsampled = resample(data_majority, 

                                 replace=True, 

                                 n_samples=data_minority.shape[0], 

                                 random_state=123)



data_downsampled = pd.concat([data_major_downsampled, data_minority])

 

# Display new class counts

data_downsampled.Class.value_counts()
y_down = data_downsampled.Class

X_down = data_downsampled.drop('Class', axis=1)

X_train_down,X_test_down,y_train_down,y_test_down=train_test_split(X_down,y_down,test_size=0.3,random_state=0)

X_train_down=rbst.fit_transform(X_train_down)

X_test_down=rbst.transform(X_test_down)

logreg_down = LogisticRegression().fit(X_train_down, y_train_down)

log_pred_down = logreg_down.predict(X_test_down)



print("Accuracy score of Logistic Regression on Down-Sampling Data:", accuracy_score(y_test_down, log_pred_down) )

print(classification_report(y_test_down,log_pred_down))

conf_matrix(y_test_down, log_pred_down)
plot_roc_curve(logreg_down,X_test_down,y_test_down)
plot_pre_recall(logreg_down,X_test_down,y_test_down)
dectree_down=DecisionTreeClassifier()

dectree_down.fit(X_train_down, y_train_down)

dec_pred_down=dectree_down.predict(X_test_down)

print("Accuracy score of Decision Tree Classifier on DOWN-SAMPLING Data:",accuracy_score(y_test_down,dec_pred_down))

print(classification_report(y_test_down,dec_pred_down))

print("roc auc score of Decision Tree Classifier:",roc_auc_score(y_test_down,dec_pred_down))

conf_matrix(y_test_down,dec_pred_down)

plot_roc_curve(dectree_down,X_test_down,y_test_down)
plot_pre_recall(dectree_down, X_test_down,y_test_down)
from imblearn.over_sampling import SMOTE

oversample = SMOTE(sampling_strategy='auto', k_neighbors=6, random_state=42)

X_smt, y_smt = oversample.fit_resample(X, y)



X_train_smt,X_test_smt,y_train_smt,y_test_smt=train_test_split(X_smt,y_smt,test_size=0.3,random_state=42)

X_train_smt=rbst.fit_transform(X_train_smt)

X_test_smt=rbst.transform(X_test_smt)



logreg_smote=LogisticRegression()

logreg_smote.fit(X_train_smt,y_train_smt)

logreg_smt_pred=logreg_smote.predict(X_test_smt)

print("Accuracy score of Logistic Regression on SMOTE technique:",accuracy_score(y_test_smt,logreg_smt_pred))

print(classification_report(y_test_smt,logreg_smt_pred))

conf_matrix(y_test_smt,logreg_smt_pred)
plot_roc_curve(logreg_smote,X_test_smt,y_test_smt)
plot_pre_recall(logreg_smote, X_test_smt,y_test_smt)
dectree_smote=DecisionTreeClassifier()

dectree_smote.fit(X_train_smt, y_train_smt)

dec_pred_sm=dectree_smote.predict(X_test_smt)

print("Accuracy score of Decision Tree Classifier on SMOTE technique:",accuracy_score(y_test_smt,dec_pred_sm))

print(classification_report(y_test_smt,dec_pred_sm))

print("roc auc score of Decision Tree Classifier:",roc_auc_score(y_test_smt,dec_pred_sm))

conf_matrix(y_test_smt,dec_pred_sm)

plot_roc_curve(dectree_smote,X_test_smt,y_test_smt)
plot_pre_recall(dectree_smote, X_test_smt,y_test_smt)
#Since we will use the existing unprocessed data set, we will use the first X_train_im, y_train_im, X_test_im, y_test_im sets we created.

rf=RandomForestClassifier()

rf.fit(X_train,y_train)

rf_pred=rf.predict(X_test)

print("Accuracy score of Random Forest Classifier on Imbalanced Data:",accuracy_score(y_test,rf_pred))

print(classification_report(y_test,rf_pred))

conf_matrix(y_test,rf_pred)

plot_roc_curve(rf,X_test,y_test)
plot_pre_recall(rf,X_test,y_test)
from xgboost import XGBClassifier

xgb = XGBClassifier(scale_pos_weight=578) # SumofMajority/SumofMinority

xgb.fit(X_train, y_train)

xgb_pred=xgb.predict(X_test)

print("Accuracy score of XGB on Imbalanced Data:",accuracy_score(y_test,xgb_pred))

print(classification_report(y_test,xgb_pred))

conf_matrix(y_test,xgb_pred)
plot_roc_curve(xgb,X_test,y_test)
plot_pre_recall(xgb,X_test,y_test)