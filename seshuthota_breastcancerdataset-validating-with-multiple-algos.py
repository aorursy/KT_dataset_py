from sklearn import datasets



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

from sklearn.metrics import roc_curve, auc, confusion_matrix

import warnings

warnings.filterwarnings('ignore')
data = datasets.load_breast_cancer()

X = data.data

y = data.target 
print("Data Shape : ",X.shape)

print("targets Shape : ",y.shape)
print(data.keys())
X = pd.DataFrame(data=X, columns=data["feature_names"])
X = RobustScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
logreg = LogisticRegression()

logreg.fit(X_train, y_train)
print("Accuracry on training set: ",logreg.score(X_train, y_train))

print("Accuracry on test set: ",logreg.score(X_test, y_test))
svclinear = SVC(kernel='linear')

svclinear.fit(X_train, y_train)

print("Accuracry on training set: ",svclinear.score(X_train, y_train))

print("Accuracry on test set: ",svclinear.score(X_test, y_test))
svcrbf = SVC(kernel='rbf')

svcrbf.fit(X_train, y_train)

print("Accuracry on training set: ",svcrbf.score(X_train, y_train))

print("Accuracry on test set: ",svcrbf.score(X_test, y_test))
rf = RandomForestClassifier()

rf.fit(X_train, y_train)

print("Accuracry on training set: ",rf.score(X_train, y_train))

print("Accuracry on test set: ",rf.score(X_test, y_test))
bagC = BaggingClassifier()

bagC.fit(X_train, y_train)

print("Accuracry on training set: ",bagC.score(X_train, y_train))

print("Accuracry on test set: ",bagC.score(X_test, y_test))
y_pred_logreg = logreg.predict(X_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_logreg)

roc_auc = auc(false_positive_rate, true_positive_rate)



plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b',

label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
y_pred_svc_linear = svclinear.predict(X_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_svc_linear)

roc_auc = auc(false_positive_rate, true_positive_rate)



plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b',

label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
y_pred_rbf = svcrbf.predict(X_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_rbf)

roc_auc = auc(false_positive_rate, true_positive_rate)



plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b',

label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
y_pred_rf = rf.predict(X_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_rf)

roc_auc = auc(false_positive_rate, true_positive_rate)



plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b',

label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
y_pred_bagC = bagC.predict(X_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_bagC)

roc_auc = auc(false_positive_rate, true_positive_rate)



plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b',

label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
print("Confusion Matrix of Random Forests")

confusion_matrix(y_test, y_pred_rf)