
import numpy as np
import pandas as pd 
import csv
import os
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn import tree, ensemble
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc, precision_recall_curve
from pandas import Series

import matplotlib.pyplot as plt
import seaborn as sns
import random
X_test_=[];y_test_=[];
import random
with open('../input/unnormalized/normal.csv') as normalfile:
    normal_reader=list(csv.reader(normalfile))
    normal_tmp=random.sample(range(len(normal_reader)),50);
    X_test_.append([normal_reader[x] for x in normal_tmp])
    y_test_.append([0]*50)
with open('../input/unnormalized/cancer.csv') as cancerfile:
    cancer_reader=list(csv.reader(cancerfile))
    cancer_tmp=random.sample(range(len(cancer_reader)),50);
    X_test_.append([cancer_reader[x] for x in cancer_tmp])
    y_test_.append([1]*50)
print(type(X_test_))
fig = plt.figure(figsize=(20,20))
ax2 = plt.subplot(221)
ax1 = plt.subplot(222)
ax = plt.subplot(223)
l=len(normal_reader)+len(cancer_reader);


def readFile(Cut=10):
    X=[];y=[];
    flag=0;
    Cnt=int(l/Cut);
    for i in range(Cnt):
        tmp=random.randint(1, 2) 
        if(tmp==1):
            tmp1=random.randint(1,len(normal_reader)-1);
            while(tmp1 in normal_tmp):
                tmp1=random.randint(1,len(normal_reader)-1);
            X.append(normal_reader[tmp1]);
            y.append(0);
        else:
            tmp1=random.randint(1,len(cancer_reader)-1);
            while(tmp1 in cancer_tmp):
                tmp1=random.randint(1,len(cancer_reader)-1);
            X.append(cancer_reader[tmp1]);
            y.append(1);
    return X,y

def roc_auc_plot(y_true, y_proba, label=' ', l='-', lw=1.0):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_true, y_proba[:,1])
    print(label);
    print("%s (area=%.3f)"%(label,roc_auc_score(y_true, y_proba[:,1])))
    ax.plot(fpr, tpr, linestyle=l, linewidth=lw,
            label="%s (area=%.3f)"%(label,roc_auc_score(y_true, y_proba[:,1])))

def precision_recall_plot(y_true, y_proba, label=' ', l='-', lw=1.0):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    precision, recall, _ = precision_recall_curve(y_true,
                                                  y_proba[:,1])
    average_precision = average_precision_score(y_true, y_proba[:,1],
                                                     average="micro")
    ax1.plot(recall, precision, label='%s (average=%.3f)'%(label,average_precision),
            linestyle=l, linewidth=lw)
    
def roc_curve_acc(Y_test, Y_pred,method):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, Y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    
    ax2.plot(false_positive_rate, true_positive_rate,label='%s AUC = %0.3f'%(method, roc_auc))


def trainForest(Cut=10, Penalty="l1"):
    X,y=readFile(Cut);
    lsvc = LinearSVC(C=0.01, penalty=Penalty, dual=False).fit(X, y)
    model = SelectFromModel(lsvc, prefit=True)
    X_train = model.transform(X)
    y_train = y;
    X_test = model.transform(X_test_);
    y_test = y_test_;
    forest = ensemble.RandomForestClassifier(criterion='entropy', max_depth=15, min_samples_leaf=5)
    forest.fit(X_train, y_train);
    print(str(Cut)+Penalty);
    roc_auc_plot(y_test,forest.predict_proba(X_test),label="RF "+"1/"+str(Cut)+" dataset, "+Penalty+" based ",l='--')
    precision_recall_plot(y_test,forest.predict_proba(X_test),label="RF "+"1/"+str(Cut)+" dataset, "+Penalty+" based ",l='-')
    RF = RandomForestClassifier()
    RF.fit(X_train, y_train)
    y_pred = RF.predict(X_test)
    print("Random Forest Classifier report \n", classification_report(y_test, y_pred))
    roc_curve_acc(y_test, y_pred, "RF "+"1/"+str(Cut)+" dataset, "+Penalty+" based ")
    # sns.set_style("white")

ax.plot([0,1], [0,1], color='k', linewidth=0.5, linestyle='--', 
    label='Random Classifier')    

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_title('Receiver Operator Characteristic curves')

ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.grid(True)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])
ax1.set_title('Precision-recall curves')


ax2.plot([0,1],[0,1],'b--')
ax2.set_xlim([-0.1,1.2])
ax2.set_ylim([-0.1,1.2])
ax2.set_ylabel('True Positive Rate')
ax2.set_xlabel('False Positive Rate')    
ax2.set_title('Receiver Operating Characteristic')
    
trainForest(1,"l1");
#trainForest(10,"l2");
#trainForest(5,"l1");
#trainForest(5,"l2");
ax.legend(loc='best')
ax1.legend(loc='best')
ax2.legend(loc='best')
plt.show()
asdfas=123124;
print(str(asdfas));
print("asdfas"+"asdf")