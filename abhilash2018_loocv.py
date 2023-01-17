# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra|

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
X=[None]*5

label=[None]*5

classifier=[None]*5

labels=np.load("../input/sublocalisation-of-protien/benchmark_pssm_label_multi.npy")

labels.shape

for i in range(0,5):

    X[i]=list(np.load("../input/fork-of-pssm-label-"+str(i+1)+"/benchmark_label"+str(i+1)+"_GA.npy"))

    for j in range(0,578):

        X[i][j]=list(X[i][j])

    label[i]=list(labels[:,i])

print(len(X[0][1]))
def other_scores(Y_test,prediction):

    n=prediction.shape[0]

    precision_score=0

    recall_score=0

    F1_score=0

    for i in range(n):

        actual_l=set(Y_test.rows[i])

        predict_l=set(prediction.rows[i])

        if len(predict_l)!=0:

            precision_score+=len(actual_l.intersection(predict_l))/len(predict_l)

        recall_score+=len(actual_l.intersection(predict_l))/len(actual_l)

        F1_score+=2*len(actual_l.intersection(predict_l))/(len(actual_l)+len(predict_l))

    

    return precision_score/n,recall_score/n,F1_score/n

    

        

import numpy as np

from sklearn.metrics import accuracy_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

import random

from sklearn.svm import SVC





for i in range(0,5):

    classifier[i] = SVC(kernel='rbf', C=30,gamma='auto')

# C=30 -precision,C+35 - accuracy

from sklearn.metrics import accuracy_score

acc_svm = [[],[],[],[],[]]

rand_y= [[],[],[],[],[]]

rand_x= [[],[],[],[],[]]

X_dash=[None]*5

label_dash=[None]*5



for i in range(0,5):

    X_dash[i]=X[i].copy()

    label_dash[i]=label[i].copy()

#print("X'",len(X_dash[0])  )

k=0

while len(X_dash[0])!=0:

    k=k+1

    Random=random.randint(0,len(X_dash[0])-1)

    for i in range(0,5):

        rand_x[i].append(X_dash[i].pop(Random))

        rand_y[i].append(label_dash[i].pop(Random))

#print(len(rand_x[1]))

it=len(rand_x[0])//10



for i in range(0,5):

    rand_x[i]=np.array(rand_x[i])

    #print(rand_x.shape)

    rand_y[i]=np.array(rand_y[i])



#print(rand_y.shape)

X_train=np.array([None]*5)

Y_train=np.array([None]*5)

X_test=np.array([None]*5)

Y_test=np.array([0]*5)



pred=np.array([0]*5)
from sklearn.model_selection import LeaveOneOut

#SVM

import numpy as np

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score,jaccard_similarity_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from skmultilearn.problem_transform import BinaryRelevance

from sklearn.svm import SVC

import random

from sklearn.svm import SVC

from scipy.sparse import lil_matrix

#kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

loo=LeaveOneOut()





import numpy as np

from sklearn.metrics import accuracy_score

acc_svm = []

jac=[]

precision=[]

recall=[]

F1_score=[]

macro_accuracy=[]

micro_accuracy=[]





acc_svm_training=[]

#print(rand_y.shape)

label1_acc=0

label1=0

label2=0

label3=0

label2_acc=0

label3_acc=0

label2_jac=0

label_count=[0]*5

true_pred_count=[0]*5

for train, test in loo.split(rand_x[0], rand_y[0]):



    for i in range(0,5):

        X_train[i], X_test[i] = rand_x[i][train], rand_x[i][test]

        Y_train[i], Y_test[i] = rand_y[i][train], rand_y[i][test]

        #print(train)

        classifier[i].fit(X_train[i], Y_train[i])

        pred[i] = classifier[i].predict(X_test[i])



    

    #label_measure.append(list(label_score(prediction,label_mat)))

    for j  in range(5):

        if Y_test[j]==1:

            label_count[j]+=1

        if Y_test[j]==1 and pred[j]==1:

            true_pred_count[j]+=1



    prediction_sparse=lil_matrix(pred)

    labels_sparse=lil_matrix(Y_test)

    prec,rec,F1=other_scores(labels_sparse,prediction_sparse)

    precision.append(prec)

    recall.append(rec)

    F1_score.append(F1)

    a=set(labels_sparse.rows[0])

    b=set(prediction_sparse.rows[0])

    if len(a)==1:

        label1+=1

        if a.intersection(b)==a.union(b):

            label1_acc+=1

            

    if len(a)==2:

        label2+=1

        label2_jac+= len(a.intersection(b))/len(a.union(b))

        if a.intersection(b)==a.union(b):

            label2_acc+=1

    if len(a)==3:

        label3+=1

        if a.intersection(b)==a.union(b):

            label3_acc+=1



 

    

    jac.append(jaccard_similarity_score(labels_sparse,prediction_sparse))

    

    acc_svm.append(accuracy_score(labels_sparse,prediction_sparse))

    #print(classification_report(Y_test,pred))

pred_sum=0

label_sum=0

macro_accuracy=[]

for i in range(5):

    if label_count[i]!=0:

        macro_accuracy.append(true_pred_count[i]/label_count[i])

    else:

        macro_accuracy.append('None')

    pred_sum=pred_sum+true_pred_count[i]

    label_sum=label_sum+label_count[i]

micro_accuracy=pred_sum/label_sum

    



    

print ('\n OAA: ', np.mean(acc_svm),np.std(acc_svm))

print('\n  Acc',np.mean(jac),np.std(jac))

print("\n  Precision",np.mean(precision),np.std(precision))

print("\n  Recall",np.mean(recall),np.std(recall))

print("\n F1 score ",np.mean(F1_score),np.std(F1_score))



print("\n Average micro accuracy",np.mean(micro_accuracy),np.std(micro_accuracy))

print("\n Average macro accuracy",macro_accuracy)
print("Label -1 accuracy",label1_acc/label1)

print("Label -2 oaa,jac",label2_acc/label2,label2_jac/label2)

print("Label -3 accuracy",label3_acc/label3)

label1,label2,label3

label1_acc,label2_acc,label3_acc

macro_accuracy[0]
def label_score(Y_test,pred):

    label_count=[0]*4

    true_pred_count=[0]*4

    for j  in range(4):

        for i in range(len(pred)):

            if Y_test[i][j]==1:

                label_count[j]+=1

            if Y_test[i][j]==1 and pred[i][j]==1:

                true_pred_count[j]+=1

    pred_sum=0

    label_sum=0

    macro_accuracy=[]

    for i in range(4):

        if label_count[i]!=0:

            macro_accuracy.append(true_pred_count[i]/label_count[i])

        else:

            macro_accuracy.append('None')

        pred_sum=pred_sum+true_pred_count[i]

        label_sum=label_sum+label_count[i]

    micro_accuracy=pred_sum/label_sum

    return macro_accuracy,micro_accuracy
X_test=np.load("../input/sublocalisation-of-protien/novel_pssm_feat_multi.npy")

Y_test=np.load('../input/sublocalisation-of-protien/novel_pssm_label_multi.npy')

X_support=[[],[],[],[],[]]

for i in range(0,5):

    X_support[i]=list(np.load("../input/fork-of-pssm-label-"+str(i+1)+"/benchmark_label"+str(i+1)+"_features_GA.npy"))

X_t=np.array([None]*4)

for i in range(0,4):

    X_t[i]=[]

    for k in range(0,122):

        

        temp=[]

        for j in range(400):

            if X_support[i][j]:

                temp.append(X_test[k][j])

        X_t[i].append(np.array(temp))

    X_t[i]=np.array(X_t[i])



        

    
rand_x[2].shape,X_t[2].shape

for  i in range(0,4):

    classifier[i].fit(rand_x[i],rand_y[i])
pred=[0]*5

prediction=np.zeros(122*4).reshape(122,4)

for i in range(0,4):

    pred[i]=classifier[i].predict(X_t[i])

    prediction[:,i]=pred[i]

    

macro_accuracy,micro_accuracy=label_score(Y_test,prediction)

OAA=accuracy_score(Y_test,prediction)

Y_test=lil_matrix(Y_test)

prediction=lil_matrix(prediction)

precision,recall,F1=other_scores(Y_test,prediction)



print("OAA ",OAA)

print("Accuracy",jaccard_similarity_score(Y_test,prediction))  

print("precision",precision)

print("recall",recall)

print("F1 score",F1)

print("Micro_accuracy",micro_accuracy)

print("Macro accuracy",macro_accuracy)



label1_acc=0

label1=0

label2=0

label3=0

label2_acc=0

label3_acc=0

label_jac=0

for i in range(122):

    a=set(Y_test.rows[i])

    b=set(prediction.rows[i])

    if len(a)==1:

        label1+=1

        if a.intersection(b)==a.union(b):

            label1_acc+=1

    if len(a)==2:

        label2+=1

        label_jac+=len(a.intersection(b))/len(a.union(b))

        if a.intersection(b)==a.union(b):

            label2_acc+=1

print("Label -1 accuracy",label1_acc/label1)

print("Label -2 accuracy",label2_acc/label2,label_jac/label2)



print(label1,label2)

print(label1_acc,label2_acc)
