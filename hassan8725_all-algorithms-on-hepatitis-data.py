# Import basic libraries 

import os

import pandas as pd 

import numpy as np

import seaborn as sns

from string import ascii_uppercase

from pandas import DataFrame

import matplotlib.pyplot as plt        

%matplotlib inline



import sklearn.preprocessing as skp

import sklearn.model_selection as skm

import os

#import classification modules

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier

# Selection

from sklearn.model_selection import GridSearchCV as gs

from sklearn.model_selection import RandomizedSearchCV as rs

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import RFECV

#import decision tree plotting libraries

# Metrics

from sklearn.metrics import accuracy_score, confusion_matrix,precision_score, recall_score, roc_auc_score,roc_curve, auc, f1_score 
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
# Loading Dataset

missing=["na","--",".",".."]

td= pd.read_csv("/kaggle/input/hepatitis/hepatitis.csv",na_values=missing)

td.head()
td.isnull().sum() # Checking for nulls
td["class"].replace((1,2),(0,1),inplace=True)
td["class"]=td["class"].astype("bool")
td.describe()
# Discretization of Age Column

td["age"]=np.where((td["age"]>10) & (td["age"]<20),"Teenagers",

                   np.where((td["age"]>=20) & (td["age"]<=30),"Adults",

                   np.where((td["age"]>30) & (td["age"]<=40),"Middle Aged",np.where((td["age"]<=10),"Children",

                            "Old"))))
td["age"]=pd.Categorical(td.age,["Children",'Teenagers','Adults', 'Middle Aged', 'Old'],ordered=True)
td["age"].value_counts() 
#draw a bar plot of Age vs. survival

sns.barplot(x="age", y="class", data=td)

plt.show()
td["sex"].replace((1,2),("Male","Female"),inplace=True)

td["sex"]=pd.Categorical(td.sex,["Male",'Female'],ordered=False)

td.head()
td.dropna(inplace=True) # Now dropping all nulls
td.dtypes
#We have categorical variables .getdummies seperates the different categories of categorical variables as separate 

#binary columns

td1 = pd.get_dummies(td,drop_first=True)

#List of new columns

print(td1.columns)

td1.head(5)
td1["bilirubin"]=np.abs((td1["bilirubin"]-td1["bilirubin"].mean())/(td1["bilirubin"].std()))

td1["albumin"]=np.abs((td1["albumin"]-td1["albumin"].mean())/(td1["albumin"].std()))
y=td1["class"].copy()

X=td1.drop(columns=["class"])

print(y.shape)

print(X.shape)
#Random Forest method for feature selection

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()    

#thit is how we get the feature importance with simple steps:

X_features=X.columns

model.fit(X, y)

# display the relative importance of each attribute

importances = np.around(model.feature_importances_,decimals=4)

imp_features= model.feature_importances_

feature_array=np.array(X_features)

sorted_features=pd.DataFrame(list(zip(feature_array,imp_features))).sort_values(by=1,ascending=False)

#print(sorted_features)

data_top=sorted_features[:X.shape[1]]

feature_to_rem=sorted_features[X.shape[1]:]

print("Unimportant Columms after simple Random Forrest\n",feature_to_rem)

rem_index=list(feature_to_rem.index)

print(rem_index)

print("Important Columms after simple Random Forrest\n",data_top)

data_top_index=list(data_top.index)

print("Important Columms after simple Random Forrest\n",data_top_index )

print(importances)

#0.0250 is a  selected threshold looking at the importance values this can be changed to any other value too

#cols_randfor_removed=[index for index,value in enumerate(importances) if value <= 0.0250]

#print(cols_randfor_removed)

X_randfor_sel = X.drop(X.columns[rem_index],axis=1)

#X_randfor_sel = X.drop(X.columns[cols_randfor_removed],axis=1)

features_randfor_select=X_randfor_sel.columns

print(features_randfor_select)
#creat train-test split parts for manual split



trainX, testX, trainy, testy= skm.train_test_split(X,y, test_size=0.25, random_state=99) #explain random state

print("\n shape of train split: ")

print(trainX.shape, trainy.shape)

print("\n shape of train split: ")

print(testX.shape, testy.shape)
### Making X Scalar for ML algorithms

X = skp.StandardScaler().fit(X).transform(X)
knn = KNeighborsClassifier()

knn.fit(trainX,trainy)

predictions = knn.predict(testX)

accknn=accuracy_score(testy, predictions)*100

print("Accuracy of KNN (%): \n", accknn)  

#get FPR

fprknn, tprknn, _ = roc_curve(testy, predictions)

aucknn=auc(fprknn, tprknn)*100

print("AUC OF KNN (%): \n", aucknn)

recallknn=recall_score(testy,predictions)*100

print("Recall of KNN is: \n",recallknn)

precknn=precision_score(testy,predictions)*100

print("Precision of KNN is: \n",precknn)
gnb=GaussianNB()

gnb.fit(trainX,trainy)

predictions = gnb.predict(testX)

accgnb=accuracy_score(testy, predictions)*100

print("Accuracy of Gaussian Naive Bayes (%): \n",accgnb)  

#get FPR

fprgnb, tprgnb, _ = roc_curve(testy, predictions)

aucgnb=auc(fprgnb, tprgnb)*100

print("AUC OF Gaussian Naive Bayes (%): \n", aucgnb)

recallgnb=recall_score(testy,predictions)*100

print("Recall of Gaussian Naive Bayes is: \n",recallgnb)

precgnb=precision_score(testy,predictions)*100

print("Precision of Gaussian Naive Bayes is: \n",precgnb)
lrg=LogisticRegression(solver='lbfgs')

lrg.fit(trainX,trainy)

predictions = lrg.predict(testX)

acclrg=accuracy_score(testy, predictions)*100

print("Accuracy of Logistic regression (%): \n",acclrg)  

#get FPR

fprlrg, tprlrg, _ = roc_curve(testy, predictions)

auclrg=auc(fprlrg, tprlrg)*100

print("AUC OF Logistic regression (%): \n", auclrg)

recalllrg=recall_score(testy,predictions)*100

print("Recall of Logistic regression is: \n",recalllrg)

preclrg=precision_score(testy,predictions)*100

print("Precision of Logistic regression is: \n",preclrg)
nn=MLPClassifier(solver='lbfgs',hidden_layer_sizes=20,batch_size=150,max_iter=100, random_state=1)

nn.fit(trainX,trainy)

predictions = nn.predict(testX)

accnn=accuracy_score(testy, predictions)*100

print("Accuracy of Neural Networks (%): \n",accnn)  

#get FPR

fprnn, tprnn, _ = roc_curve(testy, predictions)

aucnn=auc(fprnn, tprnn)*100

print("AUC OF Neural Networks (%): \n", aucnn)

recallnn=recall_score(testy,predictions)*100

print("Recall of Neural Networks is: \n",recallnn)

precnn=precision_score(testy,predictions)*100

print("Precision of Neural Networks is: \n",precnn)
svm=clf = SVC(gamma="auto",kernel='poly',degree=3)

svm.fit(trainX,trainy)

predictions = svm.predict(testX)

accsvm=accuracy_score(testy, predictions)*100

print("Accuracy of Support Vector Machine (%): \n",accsvm)  

#get FPR

fprsvm, tprsvm, _ = roc_curve(testy, predictions)

aucsvm=auc(fprsvm, tprsvm)*100

print("AUC OF Support Vector Machine (%): \n", aucsvm)

recallsvm=recall_score(testy,predictions)*100

print("Recall of Support Vector Machine is: \n",recallsvm)

precsvm=precision_score(testy,predictions)*100

print("Precision of Support Vector Machine is: \n",precsvm)
dt=DecisionTreeClassifier(max_depth=10,criterion="gini")

dt.fit(trainX,trainy)

predictions = dt.predict(testX)

accdt=accuracy_score(testy, predictions)*100

print("Accuracy of Decision Tree (%): \n",accdt)  

#get FPR

fprdt, tprdt, _ = roc_curve(testy, predictions)

aucdt=auc(fprdt, tprdt)*100

print("AUC OF Decision Tree (%): \n",aucdt)

recalldt=recall_score(testy,predictions)*100

print("Recall of Decision Tree is: \n",recalldt)

precdt=precision_score(testy,predictions)*100

print("Precision of Decision Tree is: \n",precdt)
rf=RandomForestClassifier()

rf.fit(trainX,trainy)

predictions = rf.predict(testX)

accrf=accuracy_score(testy, predictions)*100

print("Accuracy of Random Forest (%): \n",accrf)  

#get FPR

fprrf, tprrf, _ = roc_curve(testy, predictions)

aucrf=auc(fprrf, tprrf)*100

print("AUC OF Random Forest (%): \n", aucrf)

recallrf=recall_score(testy,predictions)*100

print("Recall of Random Forest is: \n",recallrf)

precrf=precision_score(testy,predictions)*100

print("Precision of Random Forest is: \n",precrf)
ab=AdaBoostClassifier()

ab.fit(trainX,trainy)

predictions = ab.predict(testX)

accab=accuracy_score(testy, predictions)*100

print("Accuracy of AdaBoost (%): \n",accab)  

#get FPR

fprab, tprab, _ = roc_curve(testy, predictions)

aucab=auc(fprab, tprab)*100

print("AUC OF AdaBoost (%): \n",aucab)

recallab=recall_score(testy,predictions)*100

print("Recall of AdaBoost is: \n",recallab)

precab=precision_score(testy,predictions)*100

print("Precision of AdaBoost is: \n",precab)
gb=GradientBoostingClassifier()

gb.fit(trainX,trainy)

predictions = gb.predict(testX)

accgb=accuracy_score(testy, predictions)*100

print("Accuracy of Gradient Descent Boosting (%): \n",accgb)  

#get FPR

fprgb, tprgb, _ = roc_curve(testy, predictions)

aucgb=auc(fprgb, tprgb)*100

print("AUC OF Gradient Descent Boosting (%): \n", aucgb)

recallgb=recall_score(testy,predictions)*100

print("Recall of Gradient Descent Boosting is: \n",recallgb)

precgb=precision_score(testy,predictions)*100

print("Precision of Gradient Descent Boosting is: \n",precgb)
algos=["K Nearest Neighbor","Guassian Naive Bayes","Logistic Regression","Neural Networks","Support Vector Machine","Decision Tree","Random Forrest","AdaBoost","Gradient Descent Boosting"]

acc=[accknn,accgnb,acclrg,accnn,accsvm,accdt,accrf,accab,accgb]

auc=[aucknn,aucgnb,auclrg,aucnn,aucsvm,aucdt,aucrf,aucab,aucgb]

recall=[recallknn,recallgnb,recalllrg,recallnn,recallsvm,recalldt,recallrf,recallab,recallgb]

prec=[precknn,precgnb,preclrg,precnn,precsvm,precdt,precrf,precab,precgb]

comp={"Algorithms":algos,"Accuracies":acc,"Area Under the Curve":auc,"Recall":recall,"Precision":prec}

compdf=pd.DataFrame(comp)

display(compdf.sort_values(by=["Accuracies","Area Under the Curve","Recall","Precision"], ascending=False))

import sklearn.metrics as metrics

roc_auc1=metrics.auc(fprknn,tprknn)

roc_auc2=metrics.auc(fprgnb,tprgnb)

roc_auc3=metrics.auc(fprlrg,tprlrg)

roc_auc4=metrics.auc(fprnn,tprnn)

roc_auc5=metrics.auc(fprsvm,tprsvm)

roc_auc6=metrics.auc(fprdt,tprdt)

roc_auc7=metrics.auc(fprrf,tprrf)

roc_auc8=metrics.auc(fprab,tprab)

roc_auc9=metrics.auc(fprgb,tprgb)



# Method-I: PLot

import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(20,10))

plt.title("Receiver Operating Curve")

plt.plot(fprknn,tprknn,"b",label="ROC of KNN = %0.2f" % roc_auc1)

plt.plot(fprgnb,tprgnb,"r",label="ROC of Guassian Naive Bayes = %0.2f" % roc_auc2)

plt.plot(fprlrg,tprlrg,"y",label="ROC of Logistic Regression = %0.2f" % roc_auc3)

plt.plot(fprnn,tprnn,"c",label="ROC of Neural Networks = %0.2f" % roc_auc4)

plt.plot(fprsvm,tprsvm,"k",label="ROC of SVM = %0.2f" % roc_auc5)

plt.plot(fprdt,tprdt,"m",label="ROC of Descision Tree= %0.2f" % roc_auc6)

plt.plot(fprrf,tprrf,"y--",label="ROC of Random Forrest= %0.2f" % roc_auc7)

plt.plot(fprab,tprab,"g--",label="ROC of Ada Boost= %0.2f" % roc_auc8)

plt.plot(fprgb,tprgb,"b--",label="ROC of Gradient Boost= %0.2f" % roc_auc9)

plt.rcParams.update({'font.size': 16})

plt.legend(loc="lower right")

plt.plot([0, 1],[0, 1],"r--")

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel("True Positive Rate")

plt.xlabel("False Positive Rate")



plt.rc('axes', labelsize=15)

plt.rc('axes', titlesize=22)
from sklearn.model_selection import RandomizedSearchCV as rs

# K Nearest Neighbor with random search

parameters={"algorithm":['auto','ball_tree','kd_tree','brute'],"n_neighbors":range(1,10,1),"p":[1,2],"weights":["uniform","distance"]}

clf_knn=KNeighborsClassifier()

clfknnrs=rs(clf_knn,parameters,cv=5,scoring="precision")

clfknnrs.fit(trainX,trainy)

predictions = clfknnrs.predict(testX)

accknnrs=accuracy_score(testy, predictions)*100

print("Accuracy of KNN after Hyperparameter Tuning (%): \n",accknnrs)  

#get FPR

fprknnrs, tprknnrs, _ = roc_curve(testy, predictions)

#aucdtrs=auc(fprdtrs, tprdtrs)*100

#print("AUC OF Decision Tree after Hyperparameter Tuning (%): \n",aucdtrs)

recallknnrs=recall_score(testy,predictions)*100

print("Recall of KNN after Hyperparameter Tuning is: \n",recallknnrs)

precknnrs=precision_score(testy,predictions)*100

print("Precision of KNN after Hyperparameter Tuning is: \n",precknnrs)



#examnine the best model

#single best score achieved accross all params

print("Best Score (%): \n",clfknnrs.best_score_*100)

#Dictionary Containing the parameters 

print("Best Parameters: \n",clfknnrs.best_params_)



print("Best Estimators: \n",clfknnrs.best_estimator_)
from sklearn.model_selection import RandomizedSearchCV as rs

# Logistic Regression with random search

parameters={"solver":['lbfgs','newton-cg','liblinear','sag','saga'],"max_iter":range(100,500,100)}

clf_lrg=LogisticRegression()

clflrgrs=rs(clf_lrg,parameters,cv=5,scoring="precision")

clflrgrs.fit(trainX,trainy)

predictions = clflrgrs.predict(testX)

acclrgrs=accuracy_score(testy, predictions)*100

print("Accuracy of Logistic Regression after Hyperparameter Tuning (%): \n",acclrgrs)  

#get FPR

fprlrgrs, tprlrgrs, _ = roc_curve(testy, predictions)

#aucdtrs=auc(fprdtrs, tprdtrs)*100

#print("AUC OF Decision Tree after Hyperparameter Tuning (%): \n",aucdtrs)

recalllrgrs=recall_score(testy,predictions)*100

print("Recall of Logistic Regression after Hyperparameter Tuning is: \n",recalllrgrs)

preclrgrs=precision_score(testy,predictions)*100

print("Precision of Logistic Regression after Hyperparameter Tuning is: \n",preclrgrs)



#examnine the best model

#single best score achieved accross all params

print("Best Score (%): \n",clflrgrs.best_score_*100)

#Dictionary Containing the parameters 

print("Best Parameters: \n",clflrgrs.best_params_)



print("Best Estimators: \n",clflrgrs.best_estimator_)
from sklearn.model_selection import RandomizedSearchCV as rs

# Decision Tree with random search

parameters={"min_samples_split":range(10,200,10),"max_depth":range(1,20,1)}

clf_treers=DecisionTreeClassifier()

clfrs=rs(clf_treers,parameters,cv=5,scoring="precision")

clfrs.fit(trainX,trainy)

predictions = clfrs.predict(testX)

accdtrs=accuracy_score(testy, predictions)*100

print("Accuracy of Decision Tree after Hyperparameter Tuning (%): \n",accdtrs)  

#get FPR

fprdtrs, tprdtrs, _ = roc_curve(testy, predictions)

#aucdtrs=auc(fprdtrs, tprdtrs)*100

#print("AUC OF Decision Tree after Hyperparameter Tuning (%): \n",aucdtrs)

recalldtrs=recall_score(testy,predictions)*100

print("Recall of Decision Tree after Hyperparameter Tuning is: \n",recalldtrs)

precdtrs=precision_score(testy,predictions)*100

print("Precision of Decision Tree after Hyperparameter Tuning is: \n",precdtrs)



#examnine the best model

#single best score achieved accross all params

print("Best Score (%): \n",clfrs.best_score_*100)

#Dictionary Containing the parameters 

print("Best Parameters: \n",clfrs.best_params_)



print("Best Estimators: \n",clfrs.best_estimator_)
from sklearn.model_selection import RandomizedSearchCV as rs

# Neural Networks with random search

parameters={"solver":['lbfgs','sgd','adam'],"hidden_layer_sizes":range(1,100,1),"batch_size":range(50,250,10),"max_iter":range(100,500,50),"learning_rate":['constant', 'invscaling', 'adaptive'],"activation":['identity', 'logistic', 'tanh', 'relu']}

clf_nn=MLPClassifier()

clfnnrs=rs(clf_nn,parameters,cv=5,scoring="precision")

clfnnrs.fit(trainX,trainy)

predictions = clfnnrs.predict(testX)

accnnrs=accuracy_score(testy, predictions)*100

print("Accuracy of Neural Networks after Hyperparameter Tuning (%): \n",accnnrs)  

#get FPR

fprnnrs, tprnnrs, _ = roc_curve(testy, predictions)

#aucdtrs=auc(fprdtrs, tprdtrs)*100

#print("AUC OF Decision Tree after Hyperparameter Tuning (%): \n",aucdtrs)

recallnnrs=recall_score(testy,predictions)*100

print("Recall of Neural Networks after Hyperparameter Tuning is: \n",recallnnrs)

precnnrs=precision_score(testy,predictions)*100

print("Precision of Neural Networks after Hyperparameter Tuning is: \n",precnnrs)



#examnine the best model

#single best score achieved accross all params

print("Best Score (%): \n",clfnnrs.best_score_*100)

#Dictionary Containing the parameters 

print("Best Parameters: \n",clfnnrs.best_params_)



print("Best Estimators: \n",clfnnrs.best_estimator_)
import sklearn.metrics as metrics

roc_auc1=metrics.auc(fprknnrs,tprknnrs)

roc_auc2=metrics.auc(fprdtrs,tprdtrs)

roc_auc3=metrics.auc(fprnnrs,tprnnrs)

roc_auc4=metrics.auc(fprlrgrs,tprlrgrs)



# Method-I: PLot

import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(15,10))

plt.title("Receiver Operating Curve")

plt.plot(fprknnrs,tprknnrs,"b",label="ROC of KNN after RS= %0.2f" % roc_auc1)

plt.plot(fprdtrs,tprdtrs,"r",label="ROC of Decision Tree after RS= %0.2f" % roc_auc2)

plt.plot(fprnnrs,tprnnrs,"g",label="ROC of Neural Networks after RS= %0.2f" % roc_auc3)

plt.plot(fprlrgrs,tprlrgrs,"k",label="ROC of Logistic Regression after RS= %0.2f" % roc_auc4)

plt.rcParams.update({'font.size': 20})



plt.legend(loc="lower right")

plt.plot([0, 1],[0, 1],"r--")

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel("True Positive Rate")

plt.xlabel("False Positive Rate")





plt.rc('axes', labelsize=15)

plt.rc('axes', titlesize=22)
algos1=["K Nearest Neighbor","Neural Networks","Decision Tree","Logistic Regression"]

acc1=[accknn,accnn,accdt,acclrg]

recall1=[recallknn,recallnn,recalldt,recalllrg]

prec1=[precknn,precnn,precdt,preclrg]

comp1={"Algorithms":algos1,"Accuracies before RS":acc1,"Recall before RS":recall1,"Precision before RS":prec1}

compdf1=pd.DataFrame(comp1)

display(compdf1.sort_values(by=["Accuracies before RS","Recall before RS","Precision before RS"], ascending=False))

acc2=[accknnrs,accnnrs,accdtrs,acclrgrs]

recall2=[recallknnrs,recallnnrs,recalldtrs,recalllrgrs]

prec2=[precknnrs,precnnrs,precdtrs,preclrgrs]

comp2={"Algorithms":algos1,"Accuracies after RS":acc2,"Recall after RS":recall2,"Precision after RS":prec2}

compdf2=pd.DataFrame(comp2)

display(compdf2.sort_values(by=["Accuracies after RS","Recall after RS","Precision after RS"], ascending=False))