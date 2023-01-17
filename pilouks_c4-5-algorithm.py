import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train_raw = pd.read_csv("../input/dataset/x_train_gr_smpl.csv")

labels = pd.read_csv("../input/dataset/y_train_smpl.csv")
label_paths = [

    "../input/dataset/y_train_smpl_0.csv",

    "../input/dataset/y_train_smpl_1.csv",

    "../input/dataset/y_train_smpl_2.csv",

    "../input/dataset/y_train_smpl_3.csv",

    "../input/dataset/y_train_smpl_4.csv",

    "../input/dataset/y_train_smpl_5.csv",

    "../input/dataset/y_train_smpl_6.csv",

    "../input/dataset/y_train_smpl_7.csv",

    "../input/dataset/y_train_smpl_8.csv",

    "../input/dataset/y_train_smpl_9.csv",

]
from sklearn.tree import DecisionTreeClassifier 

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, f1_score

from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]

def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]



scoring = {'accuracy': make_scorer(accuracy_score), 'precision': make_scorer(precision_score),'roc_auc': make_scorer(roc_auc_score), 

           'recall': make_scorer(recall_score),'f1_score' : make_scorer(f1_score), 'tp': make_scorer(tp),'fp': make_scorer(fp)}
def C45(paths, max_depth):

    for i, path in enumerate(paths):

        label = pd.read_csv(path)

        print("\ny_train_smpl {}".format(i))

        for depth in max_depth:

            print("\n max_depth {}".format(depth))

            clf = DecisionTreeClassifier(max_depth = depth)

            scores = cross_validate(clf, train_raw, label, cv=10,scoring = scoring,return_train_score=True)

            print("Accuracy train :", (sum(scores['train_accuracy'])/len(scores['train_accuracy'])))

            print("Accuracy test :", (sum(scores['test_accuracy'])/len(scores['test_accuracy'])))

            print("Train Precision :", (sum(scores['train_precision'])/len(scores['train_precision'])))

            print("Test Precision :", (sum(scores['test_precision'])/len(scores['test_precision'])))

            print("Test Recall :", (sum(scores['test_recall'])/len(scores['test_recall'])))

            print("Test f1_score :", (sum(scores['test_f1_score'])/len(scores['test_f1_score'])))

            print("Test roc_auc :", (sum(scores['test_roc_auc'])/len(scores['test_roc_auc'])))

            print("TP :", scores['test_tp'])

            print("FP :", scores['test_fp'])

                                    

                                    

C45(label_paths, [3, 5, 10, 20, 50, 100])
test_raw = pd.read_csv("../input/testfile/x_test_gr_smpl.csv")

labels_test = pd.read_csv("../input/testfile/y_test_smpl.csv")



test = pd.read_csv("../input/testfile/x_test_gr_smpl.csv")

test['label'] = labels_test
label_paths_test = [

    "../input/testfile/y_test_smpl_0.csv",

    "../input/testfile/y_test_smpl_1.csv",

    "../input/testfile/y_test_smpl_2.csv",

    "../input/testfile/y_test_smpl_3.csv",

    "../input/testfile/y_test_smpl_4.csv",

    "../input/testfile/y_test_smpl_5.csv",

    "../input/testfile/y_test_smpl_6.csv",

    "../input/testfile/y_test_smpl_7.csv",

    "../input/testfile/y_test_smpl_8.csv",

    "../input/testfile/y_test_smpl_9.csv",

]
def C45_split(paths, max_depth):

    for i, path in enumerate(paths):

        label = pd.read_csv(path)

        print("\ny_train_smpl {}".format(i))

        for depth in max_depth:

            print("\n max_depth {}".format(depth))

            clf = DecisionTreeClassifier(max_depth = depth)

            clf = clf.fit(train_raw, label)

            y_pred = clf.predict(test_raw)

            y_pred1 = clf.predict(train_raw)

            y_test = pd.read_csv(label_paths_test[i])

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            print("Accuracy Train = ", accuracy_score(label, y_pred1))

            print("Precision Train = ", precision_score(label, y_pred1))

            print("Accuracy Test = ", accuracy_score(y_test, y_pred))

            print("Precision Test = ", precision_score(y_test, y_pred))

            print("Recall = ", recall_score(y_test, y_pred))

            print("ROC_AUC_Score = ", roc_auc_score(y_test, y_pred))

            print("f_score = ", f1_score(y_test, y_pred))

            print("TP Rate = ", tp)

            print("FP Rate = ", fp)

            



C45_split(label_paths, [3, 5, 10, 20, 50, 100])
from sklearn.model_selection import train_test_split
def C45_4000(paths, max_depth):

    for i, path in enumerate(paths):

        label = pd.read_csv(path)

        X_train, X_test, y_train, y_test = train_test_split(train_raw, label, test_size=0.35, random_state=42)

        print("\ny_train_smpl {}".format(i))

        for depth in max_depth:

            print("\n max_depth {}".format(depth))

            clf = DecisionTreeClassifier(max_depth = depth)

            clf = clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            print("Accuracy = ", accuracy_score(y_test, y_pred))

            print("Precision = ", precision_score(y_test, y_pred))

            print("Recall = ", recall_score(y_test, y_pred))

            print("ROC_AUC_Score = ", roc_auc_score(y_test, y_pred))

            print("f_score = ", f1_score(y_test, y_pred))

            print("TP Rate = ", tp)

            print("FP Rate = ", fp)

            



C45_4000(label_paths, [3, 5, 10, 20, 30 ,50])
def C45_9000(paths, max_depth):

    for i, path in enumerate(paths):

        label = pd.read_csv(path)

        X1_train, X1_test, y1_train, y1_test = train_test_split(train_raw, label, test_size=0.77, random_state=42)

        print("\ny_train_smpl {}".format(i))

        for depth in max_depth:

            print("\n max_depth {}".format(depth))

            clf = DecisionTreeClassifier(max_depth = depth)

            clf = clf.fit(X1_train, y1_train)

            y_pred = clf.predict(X1_test)

            tn, fp, fn, tp = confusion_matrix(y1_test, y_pred).ravel()

            print("Accuracy = ", accuracy_score(y1_test, y_pred))

            print("Precision = ", precision_score(y1_test, y_pred))

            print("Recall = ", recall_score(y1_test, y_pred))

            print("ROC_AUC_Score = ", roc_auc_score(y1_test, y_pred))

            print("f_score = ", f1_score(y1_test, y_pred))

            print("TP Rate = ", tp)

            print("FP Rate = ", fp)

            

C45_9000(label_paths, [3, 5, 10, 20, 30 ,50])