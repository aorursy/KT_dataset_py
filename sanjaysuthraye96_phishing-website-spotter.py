%matplotlib inline

import sys

import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns 

import missingno as msno

import statsmodels.api as sm

np.set_printoptions(threshold=sys.maxsize)



from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC





import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

print("Libraries Imported")
data = pd.read_csv('../input/phishing-website-detector/phishing.txt', sep=",", header=None)

data.columns = ['UsingIP', 'LongURL', 'ShortURL', 'Symbol@', 'Redirecting//',

                'PrefixSuffix-', 'SubDomains', 'HTTPS', 'DomainRegLen', 'Favicon',

                'NonStdPort', 'HTTPSDomainURL', 'RequestURL', 'AnchorURL',

                'LinksInScriptTags', 'ServerFormHandler', 'InfoEmail', 'AbnormalURL',

                'WebsiteForwarding', 'StatusBarCust', 'DisableRightClick',

                'UsingPopupWindow', 'IframeRedirection', 'AgeofDomain',

                'DNSRecording', 'WebsiteTraffic', 'PageRank', 'GoogleIndex',

                'LinksPointingToPage', 'StatsReport', 'class']
data.head(5)
print("The Shape of the Data: ", data.shape)
print("Value Count of Class : ")

print(data['class'].value_counts())
msno.matrix(data) 
data.describe()
X = data.drop(columns='class')

y = data['class']



X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3, random_state = 10)
log_reg = LogisticRegression()

log_reg.fit(X_train,y_train)

y_pred_log=log_reg.predict(X_test)
confusion_metric = metrics.confusion_matrix(y_test, y_pred_log)



class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(confusion_metric), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')



print("Accuracy:",metrics.accuracy_score(y_test, y_pred_log))

print("Precision:",metrics.precision_score(y_test, y_pred_log))

print("Recall:",metrics.recall_score(y_test, y_pred_log))
y_pred_proba = log_reg.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_log)

auc = metrics.roc_auc_score(y_test, y_pred_log)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

print("The AUC Value is", auc)

plt.legend(loc=4)

plt.show()
naive_bayes = GaussianNB()

naive_bayes.fit(X_train, y_train)

y_pred_nb=naive_bayes.predict(X_test)
confusion_metric = metrics.confusion_matrix(y_test, y_pred_nb)



class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(confusion_metric), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')



print("Accuracy:",metrics.accuracy_score(y_test, y_pred_nb))

print("Precision:",metrics.precision_score(y_test, y_pred_nb))

print("Recall:",metrics.recall_score(y_test, y_pred_nb))
y_pred_proba = log_reg.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_nb)

auc = metrics.roc_auc_score(y_test, y_pred_nb)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

print("The AUC Value is", auc)

plt.legend(loc=4)

plt.show()
sgd_model = SGDClassifier(loss='modified_huber', shuffle=True, random_state=90)

sgd_model.fit(X_train, y_train)

y_pred_sgd=sgd_model.predict(X_test)
confusion_metric = metrics.confusion_matrix(y_test, y_pred_sgd)



class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(confusion_metric), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')



print("Accuracy:",metrics.accuracy_score(y_test, y_pred_sgd))

print("Precision:",metrics.precision_score(y_test, y_pred_sgd))

print("Recall:",metrics.recall_score(y_test, y_pred_sgd))
y_pred_proba = log_reg.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_sgd)

auc = metrics.roc_auc_score(y_test, y_pred_sgd)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

print("The AUC Value is", auc)

plt.legend(loc=4)

plt.show()
rf_model = RandomForestClassifier(n_estimators=70, oob_score=True,

                                  n_jobs=1, random_state=90, 

                                  max_features=None, min_samples_leaf=30)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
confusion_metric = metrics.confusion_matrix(y_test, y_pred_rf)



class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(confusion_metric), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')



print("Accuracy:",metrics.accuracy_score(y_test, y_pred_rf))

print("Precision:",metrics.precision_score(y_test, y_pred_rf))

print("Recall:",metrics.recall_score(y_test, y_pred_rf))
y_pred_proba = log_reg.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_rf)

auc = metrics.roc_auc_score(y_test, y_pred_rf)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

print("The AUC Value is", auc)

plt.legend(loc=4)

plt.show()
print("Accuracy of Logistic Regression:",metrics.accuracy_score(y_test, y_pred_log))

print("Accuracy of Naive Bayes:",metrics.accuracy_score(y_test, y_pred_nb))

print("Accuracy of Stochastic graident descent:",metrics.accuracy_score(y_test, y_pred_sgd))

print("Accuracy of Random Forest:",metrics.accuracy_score(y_test, y_pred_rf))