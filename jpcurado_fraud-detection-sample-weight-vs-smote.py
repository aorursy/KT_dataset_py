import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE
df = pd.read_csv('../input/creditcard.csv')
unbalance = df['Class'].sum() / df['Class'].count()
X = df.columns[1:29]

Y = 'Class'
X_train, X_test, Y_train, Y_test = train_test_split(df[X], df[Y], test_size=0.2, random_state=0)
weight = np.array([1/unbalance if i == 1 else 1 for i in Y_train])
rf = RandomForestClassifier(random_state=0)

rf.fit(X_train,Y_train, sample_weight=weight)
Y_predict=rf.predict(X_test)
confusion_matrix(Y_test,Y_predict)
print(classification_report(Y_test,Y_predict))
FP, TP, thresholds = roc_curve(Y_test,Y_predict)

roc_auc = auc(FP, TP)

print (roc_auc)
plt.plot(FP, TP, label='AUC = %0.2f'% roc_auc)

plt.title('ROC for Random Forest Classifier')

plt.plot([0,1],[0,1],'--')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
oversampler = SMOTE(random_state=0)

X_train_os,Y_train_os=oversampler.fit_sample(X_train,Y_train)
rf_os = RandomForestClassifier(random_state=0)

rf_os.fit(X_train_os,Y_train_os)
Y_predict_os=rf_os.predict(X_test)
confusion_matrix(Y_test,Y_predict_os)
print(classification_report(Y_test,Y_predict_os))
FP, TP, thresholds = roc_curve(Y_test,Y_predict_os)

roc_auc = auc(FP, TP)

print (roc_auc)
plt.plot(FP, TP, label='AUC = %0.2f'% roc_auc)

plt.title('ROC for Random Forest Classifier, with SMOTE')

plt.plot([0,1],[0,1],'--')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')