# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 





import matplotlib.pyplot as plt



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from scipy import stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import seaborn as sns



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import roc_curve, auc

from sklearn.preprocessing import label_binarize

from sklearn.linear_model import RidgeClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import average_precision_score

from sklearn.linear_model import RidgeClassifierCV

from sklearn import preprocessing
train_data=pd.read_csv(r"../input/bank-train.csv")

test_data=pd.read_csv(r"../input/bank-test.csv")
y_train=train_data["y"]

X_train=train_data.drop("y",axis=1)

cutoff=X_train.shape[0]

X=X_train.append(test_data)

X.drop("duration",axis=1,inplace=True)
X_dummies=pd.get_dummies(X,drop_first=True)
X=X_dummies.iloc[:cutoff]

X_valid=X_dummies.iloc[cutoff:]

y=y_train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
base_clf = LogisticRegression(random_state=0).fit(X_train, y_train)

y_pred=base_clf.predict(X_test)

y_score = base_clf.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class

fpr, tpr, _ = roc_curve(y_test, y_score)

roc_auc = auc(fpr, tpr)

C=confusion_matrix(y_test,y_pred)

sns.heatmap(C / C.astype(np.float).sum(axis=1))

plt.title("Confusion Matrix Normalized")
def print_classfiction_metrics(testy,yhat_classes):

    # accuracy: (tp + tn) / (p + n)

    accuracy = accuracy_score(testy, yhat_classes)

    print('Accuracy: %f' % accuracy)

    # precision tp / (tp + fp)

    precision = precision_score(testy, yhat_classes)

    print('Precision: %f' % precision)

    # recall: tp / (tp + fn)

    recall = recall_score(testy, yhat_classes)

    print('Recall: %f' % recall)

    # f1: 2 tp / (2 tp + fp + fn)

    f1 = f1_score(testy, yhat_classes)

    print('F1 score: %f' % f1)

    

    
y_test.value_counts(normalize=True)
print_classfiction_metrics(y_test,y_pred)

print("Area Under ROC Curve:", roc_auc)
y_train=train_data["y"]

X_train=train_data.drop("y",axis=1)

cutoff=X_train.shape[0]

X=X_train.append(test_data)

X.drop("duration",axis=1,inplace=True)
X_transform=X.copy()

X_transform["age"]=np.log(X_transform["age"])

#Transform the campaign varaible 

X_transform["campaign"]=np.log(X_transform["campaign"])

#Transform pdays to a category: 

bins=[0,7,14,21,28,1000]

X_transform["pdays"]=pd.cut(X_transform["pdays"],bins, labels=["OneWeek", "TwoWeek", "ThreeWeek","FourWeek","NoPreviousCampaign"],right=False)
X_transform=pd.get_dummies(X_transform,drop_first=True)
X_transform.head(1)
numerics=["age","campaign","previous","emp.var.rate","cons.price.idx","cons.conf.idx","euribor3m","nr.employed"]

X_transform[numerics] = preprocessing.scale(X_transform[numerics])
X_transform.head(1)
X_transform.shape
X=X_transform.iloc[:cutoff]

y=y_train

X_valid=X_transform.iloc[cutoff:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
y = label_binarize(y, classes=[0, 1])

n_classes = y.shape[1]

base_clf = LogisticRegression(random_state=0).fit(X_train, y_train)

y_pred=base_clf.predict(X_test)

y_score = base_clf.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class

fpr, tpr, _ = roc_curve(y_test, y_score)

roc_auc = auc(fpr, tpr)

C=confusion_matrix(y_test,y_pred)

sns.heatmap(C / C.astype(np.float).sum(axis=1))

plt.title("Confusion Matrix Normalized")
y_test.value_counts(normalize=True)
print_classfiction_metrics(y_test,y_pred)

print("Area Under ROC Curve:", roc_auc)
#Recall is kind of low. the model is not good at covering

#the successful phone calls. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

rg_base_clf = RidgeClassifier().fit(X_train, y_train)

y_score = rg_base_clf.fit(X_train, y_train).decision_function(X_test)

y_pred=rg_base_clf.predict(X_test)

fpr, tpr, _ = roc_curve(y_test, y_score)

roc_auc = auc(fpr, tpr)

print_classfiction_metrics(y_test,y_pred)

print("Area Under ROC Curve",roc_auc)
result=rg_base_clf.predict(X_valid)

submission = pd.concat([test_data["id"], pd.Series(result)], axis = 1)

submission.columns = ['id', 'Predicted']

submission.to_csv('submission.csv', index=False)
#Are Under curve also dropped 