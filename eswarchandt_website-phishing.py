import pandas as pd

import numpy as np
df=pd.read_csv(r"../input/phishing-website-detector/phishing.csv")
df.head()
df.columns
df.head()
df.shape
df.isnull().sum()
from sklearn.model_selection import train_test_split,cross_val_score
X= df.drop(columns='class')

X.head()
Y=df['class']

Y=pd.DataFrame(Y)

Y.head()
train_X,test_X,train_Y,test_Y=train_test_split(X,Y,test_size=0.3,random_state=2)
print(train_X.shape)

print(test_X.shape)

print(train_Y.shape)

print(test_Y.shape)
from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
logreg=LogisticRegression()

model_1=logreg.fit(train_X,train_Y)
logreg_predict= model_1.predict(test_X)
accuracy_score(logreg_predict,test_Y)
print(classification_report(logreg_predict,test_Y))
def plot_confusion_matrix(test_Y, predict_y):

 C = confusion_matrix(test_Y, predict_y)

 A =(((C.T)/(C.sum(axis=1))).T)

 B =(C/C.sum(axis=0))

 plt.figure(figsize=(20,4))

 labels = [1,2]

 cmap=sns.light_palette("blue")

 plt.subplot(1, 3, 1)

 sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)

 plt.xlabel('Predicted Class')

 plt.ylabel('Original Class')

 plt.title("Confusion matrix")

 plt.subplot(1, 3, 2)

 sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)

 plt.xlabel('Predicted Class')

 plt.ylabel('Original Class')

 plt.title("Precision matrix")

 plt.subplot(1, 3, 3)

 sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)

 plt.xlabel('Predicted Class')

 plt.ylabel('Original Class')

 plt.title("Recall matrix")

 plt.show()
plot_confusion_matrix(test_Y, logreg_predict)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)

model_2= knn.fit(train_X,train_Y)
knn_predict=model_2.predict(test_X)
accuracy_score(knn_predict,test_Y)
print(classification_report(test_Y,knn_predict))
plot_confusion_matrix(test_Y, knn_predict)
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()

model_3=dtree.fit(train_X,train_Y)
dtree_predict=model_3.predict(test_X)
accuracy_score(dtree_predict,test_Y)
print(classification_report(dtree_predict,test_Y))
plot_confusion_matrix(test_Y, dtree_predict)
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()

model_4=rfc.fit(train_X,train_Y)
rfc_predict=model_4.predict(test_X)
accuracy_score(rfc_predict,test_Y)
print(classification_report(rfc_predict,test_Y))
plot_confusion_matrix(test_Y, rfc_predict)
from sklearn.svm import SVC
svc=SVC()

model_5=svc.fit(train_X,train_Y)
svm_predict=model_5.predict(test_X)
accuracy_score(svm_predict,test_Y)
print(classification_report(svm_predict,test_Y))
plot_confusion_matrix(test_Y, svm_predict)
from sklearn.ensemble import AdaBoostClassifier
adc=AdaBoostClassifier(n_estimators=5,learning_rate=1)

model_6=adc.fit(train_X,train_Y)
adc_predict=model_6.predict(test_X)
accuracy_score(adc_predict,test_Y)
print(classification_report(adc_predict,test_Y))
plot_confusion_matrix(test_Y, adc_predict)
from xgboost import XGBClassifier
xgb=XGBClassifier()

model_7=xgb.fit(train_X,train_Y)
xgb_predict=model_7.predict(test_X)
accuracy_score(xgb_predict,test_Y)
plot_confusion_matrix(test_Y, xgb_predict)
print('Logistic Regression Accuracy:',accuracy_score(logreg_predict,test_Y))

print('K-Nearest Neighbour Accuracy:',accuracy_score(knn_predict,test_Y))

print('Decision Tree Classifier Accuracy:',accuracy_score(dtree_predict,test_Y))

print('Random Forest Classifier Accuracy:',accuracy_score(rfc_predict,test_Y))

print('support Vector Machine Accuracy:',accuracy_score(svm_predict,test_Y))

print('Adaboost Classifier Accuracy:',accuracy_score(adc_predict,test_Y))

print('XGBoost Accuracy:',accuracy_score(xgb_predict,test_Y))
df.columns
X=df[['PrefixSuffix-','AnchorURL']]

X.head()
train_X,test_X,train_Y,test_Y=train_test_split(X,Y,test_size=0.3,random_state=2)
print(train_X.shape)

print(test_X.shape)

print(train_Y.shape)

print(test_Y.shape)
model_8=logreg.fit(train_X,train_Y)
logreg_predict=model_8.predict(test_X)
accuracy_score(test_Y,logreg_predict)
logreg.classes_
x = np.array(X)

x
X = X.to_numpy()

y = df['class']

y= y.to_numpy()
from mlxtend.plotting import plot_decision_regions
plot_decision_regions(x, y, clf=model_1, legend=2)



# Adding axes annotations

plt.xlabel('features')

plt.ylabel('class')

plt.title('Logistic regression')

plt.show()