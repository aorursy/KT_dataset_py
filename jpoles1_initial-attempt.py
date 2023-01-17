#Import general prereqs
import pandas as pd
import numpy as np
from ggplot import *
from matplotlib import pyplot as plt
#%matplotlib inline
#First round of analysis
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
def plotROC(true_list, prob_list):
    #ROC Curve
    false_positive_rate, true_positive_rate, thresholds = roc_curve(true_list, prob_list)
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
#Second round
from sklearn import svm
#Read in data
raw_data = pd.read_csv("../input/train.csv")
#Separate into explanatory and dependent vars (X and y respectively)
y = raw_data["TARGET"]
X = raw_data.ix[:, raw_data.columns != "TARGET"]
X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
target_names = ["Satisfied", "Unsatisfied"]
pca = PCA(n_components=20)
pca_model = pca.fit(X_train)
X_train_pca = pca_model.transform(X_train)
X_test_pca = pca_model.transform(X_test)
#Run SVC to classify
clf = svm.SVC(probability=True)
clf.fit(X_train_pca, y_train)
#Test classifier on training data
predict_train = clf.predict(X_train_pca)
predict_prob_train = clf.predict_proba(X_train_pca)
true_train = (predict_train == y_train)
correct_predictions = true_train.value_counts()
print(y_train.value_counts())
print(correct_predictions)
print("Percent accurate (train): "+str(100*correct_predictions[1]/(correct_predictions[0]+correct_predictions[1]))+"%")
plotROC(true_train, predict_prob_train[:,1])
#Test classifier on test data
predict_test = clf.predict(X_test_pca)
predict_prob_test = clf.predict_proba(X_test_pca)
true_test = (predict_test == y_test)
correct_predictions = true_test.value_counts()
print(correct_predictions)
print("Percent accurate (test): "+str(100*correct_predictions[1]/(correct_predictions[0]+correct_predictions[1]))+"%")
plotROC(true_test, predict_prob_test[:,1])
