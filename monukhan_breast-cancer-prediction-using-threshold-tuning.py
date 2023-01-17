import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns


dataset = pd.read_csv('../input/breast-cancer/breast.csv')

dataset.head()
dataset.isnull().sum()
dataset1 = dataset.drop(["Unnamed: 32"], axis = 1)

dataset1


dataset1.isnull().values.any()
dataset1.info()
dataset1.describe()
#plot heat map

plt.figure(figsize=(20,20))

sns.heatmap(dataset.iloc[:,0:31].corr(),annot=True,cmap="RdYlGn")

X = dataset1.drop(['diagnosis'],axis = 1)

y = dataset1['diagnosis']

X.head()
y
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)
value_counts = pd.value_counts(y,sort = True)

value_counts.plot(kind = 'bar', rot = 0)

Label = [' 0 - Benign', '1 - Malignant']

plt.xticks(range(2), Label)

value_counts
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 1/3, random_state = 10)


from sklearn.preprocessing import StandardScaler

sc=  StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

#print(X_train)

print(X_train)
print(X_test)
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(X_train, y_train)
y_pred  = classifier.predict(X_test)
print("training  set accuracy : {}".format(classifier.score(X_train,y_train)*100))

print("test set accuracy : {}".format(classifier.score(X_test,y_test)*100))
from sklearn.metrics import confusion_matrix, accuracy_score



cm = confusion_matrix(y_test, y_pred)

print(cm)

print(accuracy_score(y_test, y_pred))

#   [TN , FP

#    FN , TP]
y_proba  = classifier.predict_proba(X_test)[:,1]
y_pred_th4 = np.where(classifier.predict_proba(X_test)[:,1]>0.4,1,0) # when  threshold is 0.4



y_pred_th3 = np.where(classifier.predict_proba(X_test)[:,1]>0.3,1,0) #  when threshold is 0.3



y_pred_th2 = np.where(classifier.predict_proba(X_test)[:,1]>0.26,1,0) #  when threshold is 0.26



y_pred_th1 = np.where(classifier.predict_proba(X_test)[:,1]>0.1,1,0) #  when threshold is 0.1

ct = pd.crosstab(y_test,y_pred_th2)

ct


from sklearn.model_selection import cross_val_score

cross_validation = cross_val_score(estimator = classifier, X = X_train,y = y_train, cv = 10)

print("\nCross validation mean accuracy of Logistic Regression = ", cross_validation.mean())

print("\nCross validation std. of Logistic Regression = ", cross_validation.std())

cross_val_res = pd.DataFrame(cross_validation*100)
cross_val_res
from sklearn.metrics import roc_curve, roc_auc_score
fpr,tpr, thresholds = roc_curve(y_test, y_proba)


fpr
tpr
thresholds
s = roc_auc_score(y_test, y_proba)

s
import matplotlib.pyplot as plt

plt.figure(figsize = (8,6))

from sklearn.metrics import  roc_curve, roc_auc_score

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.plot(fpr,tpr, color = 'darkorange', linewidth = 5, label = 'ROC Curve (area = %0.3f)' % (s))

plt.plot([0,1],[0,1], 'g--')



plt.title('Receiver Operating Characteristic')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc = 'lower right')