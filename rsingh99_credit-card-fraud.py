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
import seaborn as sns # For Plotting Graphs

import matplotlib.pyplot as plt # for Plotting Graphs

from sklearn.svm import SVC # Support Vector Machine Classifier

from sklearn.metrics import precision_score, recall_score,confusion_matrix, classification_report, accuracy_score, f1_score  ## Skearns Metrics

from sklearn.neighbors import KNeighborsClassifier ## KNN Classifier

from sklearn.model_selection import train_test_split ## Splitting Data set

from xgboost import XGBClassifier ## Boosting Algo

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_curve, auc ## Comparing Various Classifiers

import warnings # Removin Warnings

warnings.filterwarnings("ignore")
Credit_card_fraud_csv = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv",delimiter = ",") # Reading Csv file as Pandas DataFrame.

Credit_card_fraud_csv.head() # Showing Top 5 Datapoints
# V1,V2,V3 ... etc are PCA transform of Actual Features.

Credit_card_fraud_csv.shape
Credit_card_fraud_csv.info()
Credit_card_fraud_csv.describe()
def Pie(ratio, text = "CREDIT CARD FRAUDS IN EUROPE IN 2013 "):    

    fig, ax = plt.subplots()

    plt.rcParams['font.sans-serif'] = 'Arial'

    plt.rcParams['font.family'] = 'sans-serif'

    plt.rcParams['text.color'] = '#909090'

    plt.rcParams['axes.labelcolor']= '#909090'

    plt.rcParams['xtick.color'] = '#909090'

    plt.rcParams['ytick.color'] = '#909090'

    plt.rcParams['font.size']=12

    labels = ['Legit transaction', 

             'Fraud transaction']

    percentages = [1- ratio,ratio]

    explode=(0.1,0)

    ax.pie(percentages, explode=explode, labels=labels,  

           colors= ['#009ACD', '#ADD8E6'], autopct='%1.4f%%', 

           shadow=False, startangle=0,   

           pctdistance=1.4,labeldistance=1.8);

    ax.axis('equal')

    ax.set_title( text)

    ax.legend(frameon=False, bbox_to_anchor=(1.5,0.8));

Pie(ratio = 0.00172)
Amount = Credit_card_fraud_csv["Amount"]#[Credit_card_fraud_csv["Class"] == 1]

x = range(0,Amount.shape[0])

plt.style.use('seaborn-whitegrid')

plt.rcParams["figure.figsize"] = (20,8)

fig = plt.figure()

ax = plt.axes()

ax.plot(x,Amount,color='#0099CC')

#plt.xlabel("")

plt.ylabel("Amount")

ax.set_title("AMOUNT");
Fraud_Amount = Credit_card_fraud_csv[Credit_card_fraud_csv["Class"] == 1]["Amount"]

x = range(0,Fraud_Amount.shape[0])

plt.style.use('seaborn-whitegrid')

plt.rcParams["figure.figsize"] = (13,5)

fig = plt.figure()

ax = plt.axes()

ax.plot(x,Fraud_Amount,color='#0099CC')

#plt.xlabel("")

plt.ylabel("Amount")

ax.set_title("AMOUNT");
Fraud_Amount.describe()
from sklearn.preprocessing import  RobustScaler





rob_scaler = RobustScaler()



Sc_amount = rob_scaler.fit_transform(Credit_card_fraud_csv['Amount'].values.reshape(-1,1))

Sc_time   = rob_scaler.fit_transform(Credit_card_fraud_csv['Time'].values.reshape(-1,1))

Credit_card_fraud_csv.insert(0, 'scaled_amount', Sc_amount)

Credit_card_fraud_csv.insert(1, 'scaled_time', Sc_time)

Credit_card_fraud_csv.drop(['Time','Amount'], axis=1, inplace=True)

Credit_card_fraud_csv.head()
def concatenate(X,Y):

    return np.concatenate((X,Y))
legit_tranc = Credit_card_fraud_csv[Credit_card_fraud_csv["Class"] == 0]

fraud_tranc = Credit_card_fraud_csv[Credit_card_fraud_csv["Class"] == 1]

LeX, LeXte,LeT,LeTe = train_test_split(np.array(legit_tranc.iloc[:,0:-1]),np.array(legit_tranc.iloc[:,-1]),test_size  = 0.35)

FrX, FrXte,FrT,FrTe = train_test_split(np.array(fraud_tranc.iloc[:,0:-1]),np.array(fraud_tranc.iloc[:,-1]),test_size  = 0.5)

Xtrain = concatenate(LeX,FrX)

Xtest = concatenate(LeXte,FrXte)

Ytrain = concatenate(LeT,FrT)

Ytest = concatenate(LeTe,FrTe)

Xtrain.shape,Xtest.shape,Ytrain.shape,Ytest.shape,FrX.shape
neigh = KNeighborsClassifier(n_neighbors=1,algorithm='auto',n_jobs = 10)

neigh.fit(Xtrain, Ytrain)
def acc(y_test,prediction):

    # Printing Accuracy

    cm = confusion_matrix(y_test, prediction)

    recall = np.diag(cm) / np.sum(cm, axis = 1)

    precision = np.diag(cm) / np.sum(cm, axis = 0)

    

    print ('Recall:', recall)

    print ('Precision:', precision)

    print ('\n clasification report:\n', classification_report(y_test,prediction))

    print ('\n confussion matrix:\n',confusion_matrix(y_test, prediction))

    print("\n Accuracy Percentage  is : {}%".format(accuracy_score(Ytest,prediction) * 100))

    ax = sns.heatmap([precision,recall],linewidths= 0.5,cmap="YlGnBu")
y_pred = neigh.predict(Xtest)

acc(Ytest,y_pred);
Xtrain_un = concatenate(LeX[:246],FrX)

Ytrain_un = concatenate(LeT[:246],FrT)

Xtrain_un.shape,Xtest.shape,Ytrain_un.shape,Ytest.shape
Pie(Ytrain_un[Ytrain_un  == 1].sum()/Ytrain_un.shape[0],text = "Credit Card Fraud UnderSampled")
Pie(Ytest[Ytest == 1].sum()/Ytest.shape[0],text = "Credit Card Fraud Test Sample")
def train(clf,X,Y,x,y):

    clf.fit(X,Y)

    y_pred = clf.predict(x)

    acc(y_pred,y)

    return y_pred
clf = SVC(C = 5.3, cache_size=1000, class_weight="balanced", coef0=0.0,

    decision_function_shape='ovr', gamma='auto', kernel='rbf',

    max_iter=1000,  random_state=None, 

    tol=0.001, verbose=True)

yp_UNS_svc = train(clf,Xtrain_un,Ytrain_un,Xtest,Ytest)
neigh = KNeighborsClassifier(n_neighbors=7,algorithm='auto',n_jobs = 9)

yp_un_knn = train(neigh,Xtrain_un,Ytrain_un,Xtest,Ytest)
xgb = XGBClassifier(max_depth= 9,

                           learning_rate=0.001,

                           n_estimators=5000,

                           objective='binary:logistic',

                           gamma=0,

                           seed=1)

train(xgb,Xtrain_un,Ytrain_un,Xtest,Ytest)
rndmfor = RandomForestClassifier(n_estimators=1000, max_depth=13,random_state=0)

yp_uns_rndmfrs = train(rndmfor,Xtrain_un,Ytrain_un,Xtest,Ytest)
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=2)

Pie(Ytrain[Ytrain  == 1].sum()/Ytrain.shape[0],text = "Credit Card Fraud before Sampling")

Xtrain_smote, Ytrain_smote = sm.fit_sample(Xtrain, Ytrain)

Pie(Ytrain_smote[Ytrain_smote  == 1].sum()/Ytrain_smote.shape[0],text = "Credit Card Fraud after SMOTE")
clf = SVC(C = 5.3, cache_size=1000, class_weight="balanced", coef0=0.0,

    decision_function_shape='ovr', gamma='auto', kernel='rbf',

    max_iter=1000,  random_state=None, 

    tol=0.001, verbose=True)

yp_SMT_svc = train(clf,Xtrain_smote, Ytrain_smote,Xtest,Ytest)
neigh = KNeighborsClassifier(n_neighbors=3,algorithm='auto',n_jobs = 9)

yp_sm_knn = train(neigh,Xtrain_smote, Ytrain_smote,Xtest,Ytest)
logreg = LogisticRegression(C = 0.1)

yp_sm_lg = train(logreg,Xtrain_smote, Ytrain_smote,Xtest,Ytest)