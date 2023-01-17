# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

from sklearn.multiclass import OneVsRestClassifier # required for multiclass classification

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import label_binarize

from sklearn.pipeline import make_pipeline

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

from sklearn.metrics import f1_score

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import average_precision_score

from sklearn import tree

#for dirname, _, filenames in os.walk('/kaggle/input'):

#   for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('../input/fetal-health-classification/fetal_health.csv') 
# check if there is any null value

for i in train_df.columns:

    if (train_df[i].isnull().any()):

        print ("Column: {}".format(i))

    else:

        print("No null in column {}".format(i))
# number of unique classes

classes = train_df['fetal_health'].unique()

print(classes)

# percentage of each class in the dataset

count_1 = 0

count_2 = 0

count_3 = 0

for i in range(len(train_df)):

    if train_df["fetal_health"].iloc[i] == 1:

        count_1  = count_1 + 1

    elif train_df["fetal_health"].iloc[i] == 2:

        count_2  = count_2 + 1

    elif train_df["fetal_health"].iloc[i] == 3:

        count_3  = count_3 + 1

percent1 = (count_1/len(train_df))* 100

percent2 = (count_2/len(train_df))* 100

percent3= (count_3/len(train_df))* 100

print("Class 1 {}%".format(percent1))

print("Class 2 {}%".format(percent2))

print("Class 3 {}%".format(percent3))            
# the correlation between different columns in the 

corr = train_df.corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)

# Preparing the dataset and splitting into train , test and validation

# input: train dataframe

def preparing_training_data(train_df):

    training_data_array = train_df.to_numpy()

    print("[+] Training data shape: {}".format(training_data_array.shape))

    trainig = training_data_array[:,0:21]

    label   = training_data_array[:,-1]

    # binarizing the label

    label_binarized = label_binarize(label, classes=[1 ,2, 3]) # similar to one hotencoding

    n_classes = label_binarized.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(trainig, label_binarized, test_size=0.3, random_state=0)

    print("[+] X_train shape: {}".format(X_train.shape))

    print("[+] X_test shape: {}".format(X_test.shape))

    print("[+] Y_train shape: {}".format(y_train.shape))

    return (n_classes, X_train, y_train, X_test, y_test)
# calling the functio: preparing_training_data(train_df)

n_classes, X_train, y_train, X_test, y_test = preparing_training_data(train_df)
# Preparing an SVM model and trying

def linearSVM(X_train, y_train):

    clf = OneVsRestClassifier(LinearSVC(random_state=0, tol=1e-5))

    clf = make_pipeline(StandardScaler(), clf)

    clf.fit(X_train,y_train)

    return clf
# trying kernel method on SVM 

def kernelSVM(X_train, y_train):

    kernel_clf = OneVsRestClassifier(SVC(decision_function_shape = 'ovo', class_weight = 'balanced'))

    kernel_clf = make_pipeline(StandardScaler(), kernel_clf)

    kernel_clf.fit(X_train,y_train)

    return kernel_clf

    
# trying decision tree on the classifier

def decison_tree_classifer(X_train, y_train):

    decison_tree_clf = tree.DecisionTreeClassifier(criterion='entropy')

    decison_tree_clf = decison_tree_clf.fit(X_train, y_train)

    return decison_tree_clf
# training the linear and kernel classifier classifier

def training_classifier(X_train, y_train, classifier):

    model_classifer = None

    if classifier == "linear_svm":

        print("[+] ####### Testing with linear SVM ############")

        model_classifer = linearSVM(X_train, y_train)

        test_score = model_classifer.score(X_test, y_test)

        print("Test Score : {}".format(test_score))

    elif classifier == "kernel_svm":

        print("[+] ####### Testing with kernel SVM ############")

        model_classifer = kernelSVM(X_train, y_train)

        k_test_score = model_classifer.score(X_test, y_test)

        print("Test Score : {}".format(k_test_score))

    elif classifier == "decision_tree":

        print("[+] ####### Testing with decision tree ############")

        model_classifer = decison_tree_classifer(X_train, y_train)

        score = model_classifer.score(X_test, y_test)

        print("Test Score for decision ttree: {}".format(score))

        tree.plot_tree(model_classifer) 

    

    return model_classifer
# function to predict test set

def predict_classifier(X_test, y_test, classifier, model_classifier):

    if classifier == "linear_svm":

        pred = model_classifier.predict(X_test)

        print("[+] Pred Linear shape : {}".format(pred.shape))

    elif classifier == "kernel_svm":

        pred = model_classifier.predict(X_test)

        print("[+] Pred Kernel shape : {}".format(pred.shape))

    elif classifier == "decision_tree":

        pred = model_classifier.predict(X_test)

        print("[+] Pred Kernel shape : {}".format(pred.shape))



    return pred  
# calculating Area under the Precision-Recall Curve, FI Score and Area under the ROC Curve

# Compute ROC curve and ROC area for each class

classifier = "decision_tree"

model_classifier = training_classifier(X_train, y_train, classifier)

pred = predict_classifier(X_test, y_test, classifier, model_classifier)

fpr = dict()

tpr = dict()

roc_auc = dict()

for i in range(n_classes):

    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pred[:, i])

    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class

for i in range(n_classes):

    plt.figure()

    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic for class {}'.format(i))

    plt.legend(loc="lower right")

    plt.show()

    

# computing FI score

f1score = f1_score(y_test, pred, average = 'macro')

print("[+] F1 Score for {} {}".format(classifier, f1score))



# plotting precison and recall curves

average_precision_linear = average_precision_score(y_test, pred, average = "macro")

print("[+] Average Precision for {} is {}".format(classifier, average_precision_linear))



# plotting precision and recall curve

precision = dict()

recall = dict()

for i in range(n_classes):

    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],

                                                        pred[:, i])

    plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))



plt.xlabel("recall")

plt.ylabel("precision")

plt.legend(loc="best")

plt.title("precision vs. recall curve for kernel SVM")

plt.show()
