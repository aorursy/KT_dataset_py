# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from joblib import dump, load

from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score

from sklearn.linear_model import Perceptron

from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import AdaBoostClassifier

import matplotlib.pyplot as plt



from sklearn import svm, datasets

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels

from sklearn.metrics import roc_curve, auc

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import label_binarize

from sklearn.multiclass import OneVsRestClassifier

from scipy import interp

from itertools import cycle

import seaborn as sns

from sklearn.datasets import make_classification

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

import sklearn.metrics as metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Loading training set into dataframe

df = pd.read_csv('../input/nslkdd/nsl-kdd/KDDTrain+.txt', header=None)

df.head()
#Loading testing set into dataframe

qp = pd.read_csv('../input/nslkdd/nsl-kdd/KDDTest+.txt', header=None)

qp.head()
#Reset column names for training set

df.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',

'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',

'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',

'su_attempted', 'num_root', 'num_file_creations', 'num_shells',

'num_access_files', 'num_outbound_cmds', 'is_host_login',

'is_guest_login', 'count', 'srv_count', 'serror_rate',

'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',

'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',

'dst_host_srv_count', 'dst_host_same_srv_rate','dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',

'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',

'dst_host_srv_serror_rate', 'dst_host_rerror_rate',

'dst_host_srv_rerror_rate', 'subclass', 'difficulty_level']

df.head()

#Reset column names for testing set

qp.columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes',

'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',

'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',

'su_attempted', 'num_root', 'num_file_creations', 'num_shells',

'num_access_files', 'num_outbound_cmds', 'is_host_login',

'is_guest_login', 'count', 'srv_count', 'serror_rate',

'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',

'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',

'dst_host_srv_count', 'dst_host_same_srv_rate','dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',

'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',

'dst_host_srv_serror_rate', 'dst_host_rerror_rate',

'dst_host_srv_rerror_rate', 'subclass', 'difficulty_level']

qp.head()
#accessing names of training columns

lst_names = df.columns # returns a list of column names

lst_names

#accessing names of testing columns

testlst_names = qp.columns

testlst_names
#Dropping the last columns of training set

df = df.drop('difficulty_level', 1) # we don't need it in this project

df.shape
#Dropping the last columns of testing set

qp = qp.drop('difficulty_level', 1)

qp.shape
df.isnull().values.any()

qp.isnull().values.any()
#defining col list

cols = ['protocol_type','service','flag']

cols
#One-hot encoding

def one_hot(df, cols):

    """

    @param df pandas DataFrame

    @param cols a list of columns to encode

    @return a DataFrame with one-hot encoding

    """

    for each in cols:

        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)

        df = pd.concat([df, dummies], axis=1)

        df = df.drop(each, 1)

    return df

#Merging train and test data

combined_data = pd.concat([df,qp])
#Applying one hot encoding to combined data

combined_data = one_hot(combined_data,cols)
#Splitting the combined data back into training and testing

new_train_df = combined_data.iloc[:125973]

new_test_df = combined_data.iloc[125973:]
new_train_df
new_test_df
#Function to min-max normalize

def normalize(df, cols):

    """

    @param df pandas DataFrame

    @param cols a list of columns to encode

    @return a DataFrame with normalized specified features

    """

    result = df.copy() # do not touch the original df

    for feature_name in cols:

        max_value = df[feature_name].max()

        min_value = df[feature_name].min()

        if max_value > min_value:

            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)

    return result
#Dropping subclass column for training set

tmp = new_train_df.pop('subclass')
new_train_df.shape
#dropping subclass solumn for testing set

testtmp = new_test_df.pop('subclass')
new_test_df.shape
#Normalizing training set

new_train_df = normalize(new_train_df,new_train_df.columns)

new_train_df
#Normalizing testing set

new_test_df = normalize(new_test_df,new_test_df.columns)

new_test_df
#Fixing labels for training set

classlist = []

check1 = ("apache2","back","land","neptune","mailbomb","pod","processtable","smurf","teardrop","udpstorm","worm")

check2 = ("ipsweep","mscan","nmap","portsweep","saint","satan")

check3 = ("buffer_overflow","loadmodule","perl","ps","rootkit","sqlattack","xterm")

check4 = ("ftp_write","guess_passwd","httptunnel","imap","multihop","named","phf","sendmail","Snmpgetattack","spy","snmpguess","warezclient","warezmaster","xlock","xsnoop")



for item in tmp:

    if item in check1:

        classlist.append("DoS")

    elif item in check2:

        classlist.append("Probe")

    elif item in check3:

        classlist.append("U2R")

    elif item in check4:

        classlist.append("R2L")

    else:

        classlist.append("Normal")

    

    
#Appending class column to training set

new_train_df["Class"] = classlist
new_train_df
#Fixing labels for testing

testclasslist = []

testcheck1 = ("apache2","back","land","neptune","mailbomb","pod","processtable","smurf","teardrop","udpstorm","worm")

testcheck2 = ("ipsweep","mscan","nmap","portsweep","saint","satan")

testcheck3 = ("buffer_overflow","loadmodule","perl","ps","rootkit","sqlattack","xterm")

testcheck4 = ("ftp_write","guess_passwd","httptunnel","imap","multihop","named","phf","sendmail","Snmpgetattack","spy","snmpguess","warezclient","warezmaster","xlock","xsnoop")



for testitem in testtmp:

    if testitem in testcheck1:

        testclasslist.append("DoS")

    elif testitem in testcheck2:

        testclasslist.append("Probe")

    elif testitem in testcheck3:

        testclasslist.append("U2R")

    elif testitem in testcheck4:

        testclasslist.append("R2L")

    else:

        testclasslist.append("Normal")
#Appending Class column to testing set

new_test_df["Class"] = testclasslist
new_test_df
#Preparing X_train, Y_train

trainingdata = new_train_df.values

X_train = trainingdata[:,:-1]

Y_train = trainingdata[:,-1]
#Preparing X_test and Y_test

testingdata = new_test_df.values

X_test = testingdata[:,:-1]

Y_test = testingdata[:,-1]
# Saves training into ->   dump(clf,'filename.joblib')

# Loads the training ->   clf = load('filename.joblib')

#PLA

clf = Perceptron (tol=1e-3) # init classifier

clf.fit(X_train, Y_train) # fit data

#############use and evaluate model##################

train_acc = clf.score(X_train, Y_train) # mean acc on train data

test_acc = clf.score(X_test, Y_test) # mean acc on test data

y_pred = clf.predict(X_test) # make prediction

print("Training accuracy is:", train_acc )

print("Testing accuracy is:", test_acc)
#PLA performance metrics

PLA_f1score = f1_score(Y_test, y_pred, average="macro")

PLA_precision = precision_score(Y_test, y_pred, average="macro")

PLA_recall = recall_score(Y_test, y_pred, average="macro")

PLA_accuracy = accuracy_score(Y_test, y_pred)
#Logistic Regression

clf = LogisticRegression (solver='liblinear', multi_class='auto') # use all default parameters

# predication and performance evaluation are the same as Perceptron

clf.fit(X_train, Y_train) # fit data

#############use and evaluate model##################

train_acc = clf.score(X_train, Y_train) # mean acc on train data

test_acc = clf.score(X_test, Y_test) # mean acc on test data

y_pred = clf.predict(X_test) # make prediction

print("Training accuracy is:", train_acc )

print("Testing accuracy is:", test_acc)
#Logistic Regression performance metrics

Logistic_f1 = f1_score(Y_test, y_pred, average="macro")

Logistic_precision = precision_score(Y_test, y_pred, average="macro")

Logistic_recall = recall_score(Y_test, y_pred, average="macro")

Logistic_accuracy = accuracy_score(Y_test, y_pred)
#Neural Network

clf = MLPClassifier (hidden_layer_sizes =(4,6,8)) # init classifier with three hidden layers with 4, 6, and 8 hidden units, respectively.

clf = MLPClassifier (hidden_layer_sizes =(200,)) # init classifier with one hidden layer with 200 hidden units

clf.fit(X_train, Y_train) # fit data

train_acc = clf.score(X_train, Y_train) # mean acc on train data

test_acc = clf.score(X_test, Y_test) # mean acc on test data

y_pred = clf.predict(X_test) # make prediction

print("Training accuracy is:", train_acc )

print("Testing accuracy is:", test_acc)
#NN performance metrics

Neural_f1 = f1_score(Y_test, y_pred, average="macro")

Neural_precision = precision_score(Y_test, y_pred, average="macro")

Neural_recall = recall_score(Y_test, y_pred, average="macro")

Neural_accuracy = accuracy_score(Y_test, y_pred)
#Decision Tree

clf = DecisionTreeClassifier()

clf = clf.fit(X_train, Y_train)

train_acc = clf.score(X_train, Y_train) # mean acc on train data

test_acc = clf.score(X_test, Y_test) # mean acc on test data

y_pred = clf.predict(X_test) # make prediction

print("Training accuracy is:", train_acc )

print("Testing accuracy is:", test_acc)
#Decision Tree Performance Metrics

Tree_f1 = f1_score(Y_test, y_pred, average="macro")

Tree_precision = precision_score(Y_test, y_pred, average="macro")

Tree_recall = recall_score(Y_test, y_pred, average="macro")

Tree_accuracy = accuracy_score(Y_test, y_pred)
#Uniform Aggregation for PLA, logistic regression, NN and decision trees

clf1 = Perceptron (tol=1e-3)

clf2 = LogisticRegression (solver='liblinear', multi_class='auto')

clf3 = MLPClassifier (hidden_layer_sizes =(4,6,8))

clf4 = DecisionTreeClassifier()



eclf1 = VotingClassifier(estimators=[('PLA', clf1), ('LR', clf2), ('NN', clf3),('Tree',clf4)], voting='hard')

eclf1 = eclf1.fit(X_train,Y_train)

train_acc = eclf1.score(X_train, Y_train) # mean acc on train data

test_acc = eclf1.score(X_test, Y_test) # mean acc on test data

y_pred = eclf1.predict(X_test) # make prediction

print("Training accuracy is:", train_acc )

print("Testing accuracy is:", test_acc)

#Performance Metric for Uniform aggregation

Uniform_f1 = f1_score(Y_test, y_pred, average="macro")

Uniform_precision = precision_score(Y_test, y_pred, average="macro")

Uniform_recall = recall_score(Y_test, y_pred, average="macro")

Uniform_accuracy = accuracy_score(Y_test, y_pred)
#Bagging with PLA

PLA_clf = Perceptron (tol=1e-3)



clf = BaggingClassifier(base_estimator = PLA_clf)

clf = clf.fit(X_train,Y_train)

train_acc = clf.score(X_train, Y_train) # mean acc on train data

test_acc = clf.score(X_test, Y_test) # mean acc on test data

y_pred = clf.predict(X_test) # make prediction

print("Training accuracy is:", train_acc )

print("Testing accuracy is:", test_acc)
Bagging_f1 = f1_score(Y_test, y_pred, average="macro")

Bagging_precision = precision_score(Y_test, y_pred, average="macro")

Bagging_recall = recall_score(Y_test, y_pred, average="macro")

Bagging_accuracy = accuracy_score(Y_test, y_pred)
#AdaBoost with decision tree 



decision_clf = DecisionTreeClassifier()

clf = AdaBoostClassifier(decision_clf)

clf.fit(X_train,Y_train)

train_acc = clf.score(X_train, Y_train) # mean acc on train data

test_acc = clf.score(X_test, Y_test) # mean acc on test data

y_pred = clf.predict(X_test) # make prediction

print("Training accuracy is:", train_acc )

print("Testing accuracy is:", test_acc)

#Performance Metrics for Adaboost

Ada_f1 = f1_score(Y_test, y_pred, average="macro")

Ada_precision = precision_score(Y_test, y_pred, average="macro")

Ada_recall = recall_score(Y_test, y_pred, average="macro")

Ada_accuracy = accuracy_score(Y_test, y_pred)
#Random Forest

clf = RandomForestClassifier()

clf.fit(X_train,Y_train)

train_acc = clf.score(X_train, Y_train) # mean acc on train data

test_acc = clf.score(X_test, Y_test) # mean acc on test data

y_pred = clf.predict(X_test) # make prediction

print("Training accuracy is:", train_acc )

print("Testing accuracy is:", test_acc)
#Performance Metrics for Random Forest

RandomForest_f1 = f1_score(Y_test, y_pred, average="macro")

RandomForest_precision = precision_score(Y_test, y_pred, average="macro")

RandomForest_recall = recall_score(Y_test, y_pred, average="macro")

RandomForest_accuracy = accuracy_score(Y_test, y_pred)
#Displaying performance metrics 

data = [["PLA",PLA_accuracy,PLA_precision,PLA_recall,PLA_f1score],["Logistic Regression",Logistic_accuracy,Logistic_precision,Logistic_recall,Logistic_f1],["NN",Neural_accuracy,Neural_precision,Neural_recall,Neural_f1],["DTree",Tree_accuracy,Tree_precision,Tree_recall,Tree_f1],["Voting",Uniform_accuracy,Uniform_precision,Uniform_recall,Uniform_f1],["Bagging of PLA",Bagging_accuracy,Bagging_precision,Bagging_recall,Bagging_f1],["AdaBoost",Ada_accuracy,Ada_precision,Ada_recall,Ada_f1],["Random Forest",RandomForest_accuracy,RandomForest_precision,RandomForest_recall,RandomForest_f1]]

rt = pd.DataFrame(data,columns = ['','Mean Acc','Mean Precision','Mean Recall','Mean F1'])

rt
class_names = ["Normal","DoS","Probe","U2R","R2L"]
classifier = Perceptron (tol=1e-3)

y_pred = classifier.fit(X_train, Y_train).predict(X_test)

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None,cmap=plt.cm.Blues):

  

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



 

    cm = confusion_matrix(y_true, y_pred)



   

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax





np.set_printoptions(precision=2)





#PLA confusion matrix

plot_confusion_matrix(Y_test, y_pred, classes=class_names,title='PLA matrix, without normalization')

plt.show()
#Logistic Regression Matrix

classifier = LogisticRegression (solver='liblinear', multi_class='auto')

y_pred = classifier.fit(X_train, Y_train).predict(X_test)

plot_confusion_matrix(Y_test, y_pred, classes=class_names,title='Logistic Regression matrix, without normalization')

plt.show()
#NN confusion matrix

classifier = MLPClassifier (hidden_layer_sizes =(4,6,8))

y_pred = classifier.fit(X_train, Y_train).predict(X_test)

plot_confusion_matrix(Y_test, y_pred, classes=class_names,title='Neural Network matrix, without normalization')

plt.show()
#Confusion Matrix for decision tree

classifier = DecisionTreeClassifier()

y_pred = classifier.fit(X_train, Y_train).predict(X_test)

plot_confusion_matrix(Y_test, y_pred, classes=class_names,title='Decision Tree, without normalization')

plt.show()
classifier = VotingClassifier(estimators=[('PLA', clf1), ('LR', clf2), ('NN', clf3),('Tree',clf4)], voting='hard')

y_pred = classifier.fit(X_train, Y_train).predict(X_test)

plot_confusion_matrix(Y_test, y_pred, classes=class_names,title='Uniform Aggregation, without normalization')

plt.show()
#Bagging with PLA confusion matrix

classifier = BaggingClassifier(base_estimator = PLA_clf)

y_pred = classifier.fit(X_train, Y_train).predict(X_test)

plot_confusion_matrix(Y_test, y_pred, classes=class_names,title='Bagging with PLA, without normalization')

plt.show()
#Adaboost Confusion matrix



classifier = AdaBoostClassifier(decision_clf)

y_pred = classifier.fit(X_train, Y_train).predict(X_test)

plot_confusion_matrix(Y_test, y_pred, classes=class_names,title='Adaboost , without normalization')

plt.show()
#Random Forest Confusion matrix



classifier = RandomForestClassifier()

y_pred = classifier.fit(X_train, Y_train).predict(X_test)

plot_confusion_matrix(Y_test, y_pred, classes=class_names,title='Random Forest, without normalization')

plt.show()
def plot_roc_curve(fpr,tpr):

    plt.plot(fpr, tpr, color='orange', label='ROC')

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()
#ROC curve

classifier = LogisticRegression (solver='liblinear', multi_class='auto')

classifier.fit(X_train, Y_train)

probs = classifier.predict_proba(X_test)

probs = probs[:,1]

probs = label_binarize(probs, class_names)

fpr, tpr, threshold = metrics.roc_curve(Y_test,probs)

plot_roc_curve(fpr, tpr)