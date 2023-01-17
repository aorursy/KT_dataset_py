# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#bringing the files
dataset = pd.read_csv('../input/train.csv')
X = dataset.iloc[:,[2,4,5,6,7,9]].values
Y = dataset.iloc[:,1].values
dataset1 = pd.read_csv('../input/test.csv')
X1 = dataset1.iloc[:,[1,3,4,5,6,8]].values

#Removing NAN values with mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,2:6])
X[:,2:6] = imputer.transform(X[:,2:6])
#Imputer = imputer.fit(X1[:,2:6])
X1[:,2:6] = imputer.transform(X1[:,2:6])

#encoading categorical variables
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:,1])
X1[:,1] = labelencoder_X.transform(X1[:,1])

#Converting data type from object to float
X = X.astype(float)
X1 = X1.astype(float)

import statsmodels.formula.api as sm
X_fork = X1
def backwardElimination(x,sl):
    numvars = len(x[0])
    global X1,X_fork
    temp = np.zeros(x.shape).astype(int)
    temp1 = np.zeros(X_fork.shape).astype(int)
    for i in range(0,numvars):
        regression_OLS = sm.OLS(Y,x).fit()
        maxVar = max(regression_OLS.pvalues).astype(float)
        adjR_before = regression_OLS.rsquared_adj.astype(float)
        if maxVar > sl:
            for j in range(0,numvars-i):
                if (regression_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:,j]
                    temp1[:,j] = X_fork[:,j]
                    x = np.delete(x,j,1)
                    X_fork = np.delete(X_fork,j,1)
                    temp_regressor = sm.OLS(Y,x).fit()
                    adjR_after = temp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x,temp[:,[0,j]]))
                        num = len(x_rollback)
                        x_rollback = np.delete(x_rollback,(num-1),1)
                        X1_rollback = np.hstack((X_fork,temp1[:,[0,j]]))
                        num = len(X1_rollback)
                        X1_rollback = np.delete(X1_rollback,(num-1),1)
                        X_fork = X1_rollback
                        print(regression_OLS.summary)
                        return x_rollback
                    else:
                        continue
    regression_OLS.summary()
    return x
#choosing the optimal features by backward elimination process with P-value and adjusted R-square as deciding factor
sl = 0.05
X_opt = X
X_modeled = backwardElimination(X_opt,sl)

#deviding into train and test sets
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X_modeled,Y,test_size = 0.25,random_state = 0)
 
#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_fork = sc.transform(X_fork)

#fitting Logistics Regression
from sklearn.linear_model import LogisticRegression
classifier_LR = LogisticRegression(random_state = 0)
classifier_LR.fit(X_train,Y_train)
#predicting test set
Y_pred_LR = classifier_LR.predict(X_test)

from sklearn.metrics import confusion_matrix
cm_LR = confusion_matrix(Y_test,Y_pred_LR)
LR_accuracy = ((cm_LR[0,0]+cm_LR[1,1])/(cm_LR[0,0]+cm_LR[0,1]+cm_LR[1,0]+cm_LR[1,1]))*100

#fitting Support Vector Machine
from sklearn.svm import SVC
classifier_svm = SVC()
classifier_svm.fit(X_train,Y_train)
#predicting test set using SVM
Y_pred_svm = classifier_svm.predict(X_test)
cm_svm = confusion_matrix(Y_test,Y_pred_svm)
SVM_accuracy = ((cm_svm[0,0]+cm_svm[1,1])/(cm_svm[0,0]+cm_svm[0,1]+cm_svm[1,0]+cm_svm[1,1]))*100

#fitting K-NN
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 5, metric='minkowski',p=2)
classifier_knn = KNeighborsClassifier()
classifier_knn.fit(X_train,Y_train)
#predicting test using K-NN
Y_pred_knn = classifier_knn.predict(X_test)
cm_knn = confusion_matrix(Y_test,Y_pred_knn)
knn_accuracy = ((cm_knn[0,0]+cm_knn[1,1])/(cm_knn[0,0]+cm_knn[0,1]+cm_knn[1,0]+cm_knn[1,1]))*100

#fitting Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier(criterion = 'gini')
classifier_dt.fit(X_train,Y_train)
#predicting test using Decision Tree
Y_pred_dt = classifier_dt.predict(X_test)
cm_dt = confusion_matrix(Y_test,Y_pred_dt)
dt_accuracy = ((cm_dt[0,0]+cm_dt[1,1])/(cm_dt[0,0]+cm_dt[0,1]+cm_dt[1,0]+cm_dt[1,1]))*100

from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 15,criterion='gini',random_state = 0)
classifier_rf.fit(X_train,Y_train)
#predicting test using Random Forest
Y_pred_rf = classifier_rf.predict(X_test)
cm_rf = confusion_matrix(Y_test,Y_pred_rf)                                    
rf_accuracy = ((cm_rf[0,0]+cm_rf[1,1])/(cm_rf[0,0]+cm_rf[0,1]+cm_rf[1,0]+cm_rf[1,1]))*100

from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(X_train,Y_train)
#predicting test using Naive bais
Y_pred_nb = classifier_nb.predict(X_test)
cm_nb = confusion_matrix(Y_test,Y_pred_nb)                                    
nb_accuracy = ((cm_nb[0,0]+cm_nb[1,1])/(cm_nb[0,0]+cm_nb[0,1]+cm_nb[1,0]+cm_nb[1,1]))*100

print("Logistics Regression accuracy is %.2f"%LR_accuracy,"\nSupport Vector Machine accuracy is %.2f" %SVM_accuracy,"\nK-NN accuracy is %.2f"%knn_accuracy,"\nDecision Tree accuracy is %.2f"%dt_accuracy,"\nRandom Forest accuracy is %.2f"%rf_accuracy,"\nNaive Bayes accuracy is %.2f"%nb_accuracy)

#from the above table we see that Random forest gives most accuracy hence predicting using random forest
Y_pred = classifier_rf.predict(X_fork)
#output to CSV file
#pd.DataFrame(Y_pred, columns = ['Predictions']).to_csv('C:\Users\roysamri\Downloads\Kaggle\Titanic\prediction.csv')
submission = pd.DataFrame({
        "PassengerId": dataset1["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('prediction.csv', encoding='utf-8', index=False)
#fitting Support Vector Machine