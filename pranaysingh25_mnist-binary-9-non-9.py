# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
mnist_train_data= pd.read_csv('/kaggle/input/mnist_train.csv')

mnist_train= mnist_train_data.drop("label", axis=1)

mnist_label= mnist_train_data["label"].copy()



mnist_test_data= pd.read_csv('/kaggle/input/mnist_test.csv')

mnist_test= mnist_test_data.drop("label", axis=1)

mnist_test_label= mnist_test_data["label"].copy()
X_train, y_train = mnist_train.values, mnist_label.values

X_test, y_test = mnist_test.values, mnist_test_label



X_train.shape, y_train.shape, X_test.shape, y_test.shape
%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt



some_digit = X_train[36000]

some_digit_image= some_digit.reshape(28,28)



plt.imshow(some_digit_image, cmap= matplotlib.cm.binary, interpolation= "nearest")

plt.axis("off")

plt.show()



print("actual output: " + str(y_train[36000]))
shuffle_index= np.random.permutation(60000)

X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
y_train_9 = (y_train==9)

y_test_9 = (y_test==9)
y_train_9[36000]
from sklearn.linear_model import SGDClassifier

sgd_clf= SGDClassifier(random_state=42)

sgd_clf.fit(X_train, y_train_9)
print(sgd_clf.predict([some_digit]))

sgd_clf.score(X_train, y_train_9)
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_9, scoring= "accuracy", cv=3)
from sklearn.model_selection import cross_val_predict

y_train_pred= cross_val_predict(sgd_clf, X_train, y_train_9, cv= 3)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_9, y_train_pred)
tn, fp, fn, tp = confusion_matrix(y_train_9, y_train_pred).ravel()

tn, fp, fn, tp
from sklearn.metrics import precision_score, recall_score

precision_score(y_train_9, y_train_pred), recall_score(y_train_9, y_train_pred)
from sklearn.metrics import f1_score

f1_score(y_train_9, y_train_pred)
y_scores = cross_val_predict(sgd_clf, X_train, y_train_9, cv=3, method= 'decision_function')
from sklearn.metrics import precision_recall_curve



precision, recall, threshold= precision_recall_curve(y_train_9, y_scores)

threshold
def PRvsT_curve(precision, recall, threshold):

    plt.plot(threshold, precision[:-1], "b--", label= "Precision")

    plt.plot(threshold, recall[:-1], "g-", label= "Recall")

    plt.xlabel("Threshold")

    plt.legend(loc= "upper left")

    plt.ylim([0,1.6])

    

PRvsT_curve(precision, recall, threshold)

plt.show()
plt.plot(recall[:-1], precision[:-1], "r-")

plt.xlabel("Recall")

plt.ylabel("Precision")

plt.show()
my_prec_pred  = (y_scores > 0)
precision_score(y_train_9, my_prec_pred)
recall_score(y_train_9, my_prec_pred)
from sklearn.metrics import roc_curve

fpr, tpr, threshold =roc_curve(y_train_9, y_scores)



def plot_roc_curve(fpr, tpr):

    plt.plot(fpr, tpr, "r-", linewidth= 2)

    plt.plot([0,1],[0,1], "g--")

    plt.axis([0,1,0,1])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    

plot_roc_curve(fpr, tpr)

plt.show()
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_9, y_scores)
from sklearn.ensemble import RandomForestClassifier



forest_clf= RandomForestClassifier(random_state=42)

y_probs_rf= cross_val_predict(forest_clf, X_train, y_train_9, cv=3, method='predict_proba')
y_scores_forest= y_probs_rf[:,1]

fpr_forest, tpr_forest, threshold_forest= roc_curve(y_train_9, y_scores_forest)
plt.plot(fpr, tpr, "b:", label= "SGD")

plt.plot(fpr_forest, tpr_forest, label= "Random Forest")

plt.legend(loc="lower right")

plt.show()
roc_auc_score(y_train_9, y_scores_forest)
forest_predictions= cross_val_predict(forest_clf, X_train, y_train_9, cv=3)
precision_score(y_train_9, forest_predictions)
recall_score(y_train_9, forest_predictions)
y_test_data = (y_test==9)
forest_clf.fit(X_train, y_train_9)
Final_predictions= forest_clf.predict(X_test)
confusion_matrix(y_test_data, Final_predictions)
final_precision= precision_score(y_test_data, Final_predictions)

final_recall= recall_score(y_test_data, Final_predictions)

final_precision, final_recall
from sklearn.externals import joblib

joblib.dump(forest_clf, "model_mnist_binary")
Final_predictions= pd.DataFrame(Final_predictions, columns=["Predictions"])

Final_predictions= Final_predictions.to_csv(index=False)
joblib.dump(Final_predictions, "final_predictions.csv")