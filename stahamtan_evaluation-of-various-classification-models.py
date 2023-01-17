# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/creditcard.csv")
data.head()
data.isnull().any().any()
data.dtypes
data['Class'] = data['Class'].astype('bool')

data.dtypes
class_zero = data.Class.value_counts().values[0]

class_one = data.Class.value_counts().values[1]

print (data["Class"].value_counts())

print ('(Class 1 / Class 0)% : ', round(class_one/class_zero, 5)*100, '%')

sns.barplot(

    x = data.Class.value_counts().index.values,

    y = data.Class.value_counts().values)

plt.title("Class Distribution")
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

sc_amount = StandardScaler()

data['normAmount'] = sc_amount.fit_transform(data['Amount'].values.reshape(-1,1))

data = data.drop(['Time', 'Amount'], axis = 1)

data.head()
X = data.iloc[:, data.columns != 'Class'].values

y = data.iloc[:, data.columns == 'Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
def plot_precision_recall_curve(y_test, y_score, model_name):

    precision, recall, _ = metrics.precision_recall_curve(y_test, y_score)

    curve_data = pd.DataFrame(columns = range(0, len(precision)))

    curve_data.loc['Precision'] = precision

    curve_data.loc['Recall'] = recall

    print (curve_data)

    plt.step(recall, precision, color='b', alpha=0.2, where='post')

    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    plt.title('Precision Recall Curve for {} Model'.format(model_name))

    plt.xlabel('Recall')

    plt.ylabel('Precision')

    plt.xlim([0, 1.05])

    plt.ylim([0, 1.0])
def evaluate_model(y_test, y_pred, y_score, model_name):

    cm = metrics.confusion_matrix(y_test, y_pred)

    print ('Confusion Matrix for {} Model'.format(model_name))

    print (cm)

    print ('Classification Report for {} Model'.format(model_name))

    print (metrics.classification_report(y_test, y_pred, digits=6))

    print ('Area under under ROC curve for {} Model'.format(model_name))

    print (metrics.roc_auc_score(y_test, y_score))

    plot_precision_recall_curve(y_test, y_score, model_name)
from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_neighbors=5, metric= 'minkowski', p=2)

knn_clf.fit(X_train, y_train.ravel())
y_pred_knn = knn_clf.predict(X_test)

y_prob_knn = knn_clf.predict_proba(X_test)
evaluate_model(y_test, y_pred_knn, y_prob_knn[:, [1]], 'KNN (n=5)')
from sklearn.naive_bayes import GaussianNB

nb_clf = GaussianNB()

nb_clf.fit(X_train, y_train.ravel())
y_pred_nb = nb_clf.predict(X_test)

y_prob_nb = nb_clf.predict_proba(X_test)
evaluate_model(y_test, y_pred_nb, y_prob_nb[:, [1]], 'Naive Bayes')
from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression()

lr_clf.fit(X_train, y_train.ravel())
y_pred_lr = lr_clf.predict(X_test)

y_score_lr = lr_clf.decision_function(X_test)

y_prob_lr = lr_clf.predict_proba(X_test)
evaluate_model(y_test, y_pred_lr, y_prob_lr[:,[1]], 'Logistic Regression')

evaluate_model(y_test, y_pred_lr, y_score_lr, 'Logistic Regression')
from sklearn.svm import SVC

svm_clf = SVC(kernel = 'rbf', probability=True)

svm_clf.fit(X_train, y_train.ravel())
y_pred_svm = svm_clf.predict(X_test)

y_prob_svm = svm_clf.predict_proba(X_test)

y_score_svm = svm_clf.decision_function(X_test)
evaluate_model(y_test, y_pred_svm, y_score_svm, 'Kernel SVM')
evaluate_model(y_test, y_pred_svm, y_prob_svm[:,[1]], 'Kernel SVM')