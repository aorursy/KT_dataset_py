import pandas as pd

import numpy as np

import copy

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression  # For Logistic Regression

from sklearn.ensemble import RandomForestClassifier # For RFC

from sklearn.svm import SVC                               #For SVM

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.metrics import matthews_corrcoef    

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.metrics import accuracy_score,roc_curve,auc

from sklearn.model_selection import GridSearchCV

sns.set(style="ticks", color_codes=True)
df = pd.read_csv("../input/phishing-data/combined_dataset.csv")

df.head()
df.isnull().sum()

df.isna().sum()

#df.info()
df.describe()
sns.countplot(df['label'])
X= df.drop(['label', 'domain'], axis=1)

Y= df.label

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.40)

print("Training set has {} samples.".format(x_train.shape[0]))

print("Testing set has {} samples.".format(x_test.shape[0]))

def LogReg(x_train, y_train, x_test, y_test):

    LogReg1=LogisticRegression()

    #Train the model using training data 

    LogReg1.fit(x_train,y_train)

    #Test the model using testing data

    y_pred_log1 = LogReg1.predict(x_test)

    cm=confusion_matrix(y_test,y_pred_log1)

    sns.heatmap(cm,annot=True)

    print("f1 score is ",f1_score(y_test,y_pred_log1,average='weighted'))

    print("matthews correlation coefficient is ",matthews_corrcoef(y_test,y_pred_log1))

    print("The accuracy Logistic Regression on testing data is: ",100.0 *accuracy_score(y_test,y_pred_log1))

    print( classification_report(y_test,y_pred_log1))

    print(cm)

    return;
LogReg(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
# Normalizing continuous variables

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range = (0,1))

scaler.fit(x_train)

X_train = scaler.transform(x_train)

X_test = scaler.transform(x_test)
LogReg(x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)
def LogReg22(x_train, y_train, x_test, y_test):

    LogReg2=LogisticRegression(penalty='l2')

    #Train the model using training data 

    LogReg2.fit(x_train,y_train)

    #Test the model using testing data

    y_pred_log2 = LogReg2.predict(x_test)

    cm=confusion_matrix(y_test,y_pred_log2)

    sns.heatmap(cm,annot=True)

    print("f1 score is ",f1_score(y_test,y_pred_log2,average='weighted'))

    print("matthews correlation coefficient is ",matthews_corrcoef(y_test,y_pred_log2))

    print("The accuracy Logistic Regression on testing data is: ",100.0 *accuracy_score(y_test,y_pred_log2))

    print( classification_report(y_test,y_pred_log2))

    print(cm)

    return;
LogReg22(x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)
def LogReg33(x_train, y_train, x_test, y_test):

    LogReg3=LogisticRegression(penalty='l2')

    #Train the model using training data 

    LogReg3.fit(x_train,y_train)

    #Test the model using testing data

    y_pred_log3 = LogReg3.predict(x_test)

    cm=confusion_matrix(y_test,y_pred_log3)

    sns.heatmap(cm,annot=True)

    print("f1 score is ",f1_score(y_test,y_pred_log3,average='weighted'))

    print("matthews correlation coefficient is ",matthews_corrcoef(y_test,y_pred_log3))

    print("The accuracy Logistic Regression on testing data is: ",100.0 *accuracy_score(y_test,y_pred_log3))

    print( classification_report(y_test,y_pred_log3))

    print(cm)

    return;
LogReg33(x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)
LogReg1=LogisticRegression(random_state= 0, multi_class='multinomial' , solver='newton-cg')
# Create first pipeline for base without reducing features.



#pipe = Pipeline([('classifier' , LogisticRegression())])

# pipe = Pipeline([('classifier', RandomForestClassifier())])



# Create param grid.



param_grid = [

    {'random_state' : (range(0,10,2)),

     'solver' : ['newton-cg', 'liblinear']}]



# Create grid search object

logreg = LogisticRegression()



clf = GridSearchCV(logreg, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)



# Fit on data



best_clf = clf.fit(X_train, y_train)
best_clf.best_params_

#grid_result.best_params_
def LogReg44(x_train, y_train, x_test, y_test):

    LogReg4=LogisticRegression(random_state=0, solver='newton-cg')

    #Train the model using training data 

    LogReg4.fit(x_train,y_train)

    #Test the model using testing data

    y_pred_log4 = LogReg4.predict(x_test)

    cm=confusion_matrix(y_test,y_pred_log4)

    sns.heatmap(cm,annot=True)

    print("f1 score is ",f1_score(y_test,y_pred_log4,average='weighted'))

    print("matthews correlation coefficient is ",matthews_corrcoef(y_test,y_pred_log4))

    print("The accuracy Logistic Regression on testing data is: ",100.0 *accuracy_score(y_test,y_pred_log4))

    print( classification_report(y_test,y_pred_log4))

    print(cm)

    return;
LogReg44(x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)
#For RFC

param_grid = [

    {'n_estimators' : list(range(10,101,10)),

    'max_features' : list(range(2,10,1))},

     ]



# Create grid search object

rfc = RandomForestClassifier()



clf = GridSearchCV(rfc, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)



# Fit on data



best_clf = clf.fit(X_train, y_train)
best_clf.best_params_
def RFC(x_train, y_train, x_test, y_test):

    #create RFC object

    RFClass1 = RandomForestClassifier(max_depth=3, n_estimators=100)

    #Train the model using training data 

    RFClass1.fit(x_train,y_train)



    #Test the model using testing data

    y_pred_rfc1 = RFClass1.predict(x_test)



    cm=confusion_matrix(y_test,y_pred_rfc1)

    sns.heatmap(cm,annot=True)

    print("f1 score is ",f1_score(y_test,y_pred_rfc1,average='weighted'))

    print("matthews correlation coefficient is ",matthews_corrcoef(y_test,y_pred_rfc1))

    print("The accuracy Random forest classifier on testing data is: ",100.0 *accuracy_score(y_test,y_pred_rfc1))

    print( classification_report(y_test,y_pred_rfc1))

    print(cm)

    return;
RFC(x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_test)