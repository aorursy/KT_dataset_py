import numpy as np 
import pandas as pd 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv("../input/novartis-data/Train.csv")
test = pd.read_csv("../input/novartis-data/Test.csv")
train.head()

train.columns
train.shape
train.info()
train.count()
train.MULTIPLE_OFFENSE.value_counts()

X = train.drop(['MULTIPLE_OFFENSE', 'DATE', 'INCIDENT_ID'],axis=1)
eval_X = test.drop(['DATE','INCIDENT_ID'],axis=1)
Y = train['MULTIPLE_OFFENSE']

incident_ids_train = train['INCIDENT_ID']
incdent_ids_test = test['INCIDENT_ID']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=0.2)
X_train.shape, y_train.shape , X_test.shape

X_train.fillna(0, inplace=True)
X_cv.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)
eval_X.fillna(0, inplace=True)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = pd.DataFrame(scaler.transform(X_train), columns = X_train.columns)
X_cv = pd.DataFrame(scaler.transform(X_cv), columns = X_cv.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns)
eval_X = pd.DataFrame(scaler.transform(eval_X), columns = eval_X.columns)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

train_auc = []
cv_auc = []
k = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

for i in k:
    neigh = KNeighborsClassifier(n_neighbors = i)
    neigh.fit(X_train, y_train)
    
    y_train_pred = neigh.predict_proba(X_train)[:,1]
    y_cv_pred = neigh.predict_proba(X_cv)[:,1]
    
    train_auc.append(roc_auc_score(y_train, y_train_pred))
    cv_auc.append(roc_auc_score(y_cv, y_cv_pred))

plt.plot(k, train_auc, label = 'Training AUC')
plt.plot(k, cv_auc, label = 'CV AUC')
plt.legend()
plt.xlabel('k ------>')
plt.ylabel("AUC")
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

best_k_value = 13

neigh = svm.SVC(probability=True, class_weight={0: 10})


neigh.fit(X_train, y_train)

train_fpr, train_tpr, thresholds = roc_curve(y_train, neigh.predict_proba(X_train)[:,1])
test_fpr, test_tpr, thresholds = roc_curve(y_test, neigh.predict_proba(X_test)[:,1])

plt.plot(train_fpr, train_tpr, label = 'TRAIN')
plt.plot(test_fpr, test_tpr, label = 'TEST')
plt.legend()
plt.xlabel('K')
plt.ylabel('AUC')
plt.title('Error Plots')
plt.show()
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import f1_score

y_train_predict = neigh.predict(X_train)
y_test_predict = neigh.predict(X_test)

train_confusion_matrix = confusion_matrix(y_train, y_train_predict)
test_confusion_matrix = confusion_matrix(y_test, y_test_predict)
print("train CM:")
print(train_confusion_matrix)

print("test CM:")
print(test_confusion_matrix)

print("Training F1 score")
print(f1_score(y_train, y_train_predict))
print("Test F1 score")
print(f1_score(y_test, y_test_predict))
