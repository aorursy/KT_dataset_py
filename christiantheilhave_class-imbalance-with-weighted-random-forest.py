import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk

%matplotlib inline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
data = pd.read_csv("../input/creditcard.csv")
data.head()
count_classes = pd.value_counts(data['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Fraud class histogram")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.yscale("log")
from sklearn.model_selection import train_test_split

X = data.iloc[:,1:data.shape[1]-1]
y = data.iloc[:,data.shape[1]-1]

# Whole dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

print("Number transactions train dataset: ", len(X_train))
print("Number transactions test dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train)+len(X_test))
def confusion(classifier, X_test, y_test):
    y_pred  = classifier.predict(X_test)
    return confusion_matrix(y_test, y_pred).ravel()
def show(tn,fp,fn,tp):
    print("TN:" + str(tn) + " FP:" + str(fp) + " FN:" + str(fn) + " TP:" + str(tp) + 
          " FNR=" + str(fn/(fn+tp)) + " FPR=" + str(fp/(fp+tn)))
show(*confusion(RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=10).fit(X_train,y_train),X_test,y_test))
show(*confusion(RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=10, class_weight="balanced").fit(X_train,y_train),X_test,y_test))
show(*confusion(RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=10, class_weight="balanced_subsample").fit(X_train,y_train),X_test,y_test))
w_neg = 10**-4
w_pos_range = np.exp(np.arange(np.log(1), np.log(10**9)))
for w_pos in w_pos_range:
    print("w_pos: " + str(w_pos))
    show(*confusion(RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=10, class_weight={0: w_neg, 1: w_pos}).fit(X_train,y_train),X_test,y_test))