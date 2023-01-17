import pandas as pd

import numpy as np

import copy

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression  # For Logistic Regression

from sklearn.ensemble import RandomForestClassifier # For RFC

from sklearn.svm import SVC                               #For SVM

from sklearn.metrics import matthews_corrcoef    

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix,classification_report

from sklearn.metrics import accuracy_score,roc_curve,auc

sns.set(style="ticks", color_codes=True)
df = pd.read_csv("../input/phishing-data/combined_dataset.csv")
df.head()
df.isnull().sum()

df.isna().sum()
df.describe()
df.info()
sns.countplot(df['label'])
df.corr()

df.corr()['label'].sort_values()
#plt.figure(figsize = ('8','8'))

sns.heatmap(df.corr(),annot=True)
#sns.pairplot(df)
X= df.drop(['label', 'domain'], axis=1)

Y= df.label
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.40)

print("Training set has {} samples.".format(x_train.shape[0]))

print("Testing set has {} samples.".format(x_test.shape[0]))
#create logistic regression object

LogReg=LogisticRegression()



#Train the model using training data 

LogReg.fit(x_train,y_train)





#Test the model using testing data

y_pred = LogReg.predict(x_test)
cm=confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot=True)
print("f1 score is ",f1_score(y_test,y_pred,average='weighted'))

print("matthews correlation coefficient is ",matthews_corrcoef(y_test,y_pred))

print("The accuracy Logistic Regression on testing data is: ",100.0 *accuracy_score(y_test,y_pred))
LogReg1=LogisticRegression(random_state= 0, multi_class='multinomial' , solver='newton-cg')

#Train the model using training data 

LogReg1.fit(x_train,y_train)





#Test the model using testing data

y_pred_log = LogReg1.predict(x_test)



cm=confusion_matrix(y_test,y_pred_log)

sns.heatmap(cm,annot=True)

print("f1 score is ",f1_score(y_test,y_pred_log,average='weighted'))

print("matthews correlation coefficient is ",matthews_corrcoef(y_test,y_pred_log))

print("The accuracy Logistic Regression on testing data is: ",100.0 *accuracy_score(y_test,y_pred_log))
fpr,tpr,thresh = roc_curve(y_test,y_pred_log)

roc_auc = accuracy_score(y_test,y_pred_log)



# Plot ROC curve for Logistic Regression

plt.plot(fpr,tpr,'orange',label = 'Logistic Regression')

plt.legend("Logistic Regression", loc='lower right')

plt.xlabel("False positive rate")

plt.ylabel("True positive rate")

plt.legend(loc='lower right')
#create RFC object

RFClass = RandomForestClassifier()

#Train the model using training data 

RFClass.fit(x_train,y_train)



#Test the model using testing data

y_pred_rfc = RFClass.predict(x_test)



cm=confusion_matrix(y_test,y_pred_rfc)

sns.heatmap(cm,annot=True)

print("f1 score is ",f1_score(y_test,y_pred_rfc,average='weighted'))

print("matthews correlation coefficient is ",matthews_corrcoef(y_test,y_pred_rfc))

print("The accuracy Random forest classifier on testing data is: ",100.0 *accuracy_score(y_test,y_pred_rfc))
fpr,tpr,thresh = roc_curve(y_test,y_pred_rfc)

roc_auc = accuracy_score(y_test,y_pred_rfc)



# Plot ROC curve for Logistic Regression

plt.plot(fpr,tpr,'orange',label = 'Random Forest Classification')

plt.legend("Logistic Regression", loc='lower right')

plt.xlabel("False positive rate")

plt.ylabel("True positive rate")

plt.legend(loc='lower right')
#create SVM object



svc = SVC()



svc.fit(x_train,y_train)

y_pred_svc = svc.predict(x_test)



cm=confusion_matrix(y_test,y_pred_svc)

sns.heatmap(cm,annot=True)

print("f1 score is ",f1_score(y_test,y_pred_svc,average='weighted'))

print("matthews correlation coefficient is ",matthews_corrcoef(y_test,y_pred_svc))

print("The accuracy SVC on testing data is: ",100.0 *accuracy_score(y_test,y_pred_svc))
fpr,tpr,thresh = roc_curve(y_test,y_pred_svc)

roc_auc = accuracy_score(y_test,y_pred_svc)



# Plot ROC curve for SVC

plt.plot(fpr,tpr,'orange',label = 'Random Forest Classification')

plt.legend("Logistic Regression", loc='lower right')

plt.xlabel("False positive rate")

plt.ylabel("True positive rate")

plt.legend(loc='lower right')
print("The accuracy Logistic Regression on testing data is: ",100.0 *accuracy_score(y_test,y_pred_log))

print("The accuracy Random forest classifier on testing data is: ",100.0 *accuracy_score(y_test,y_pred_rfc))

print("The accuracy SVC on testing data is: ",100.0 *accuracy_score(y_test,y_pred_svc))