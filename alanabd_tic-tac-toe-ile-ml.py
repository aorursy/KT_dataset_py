# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def precision_hesapla(class_id,TP, FP, TN, FN):

    sonuc=0

    for i in range(0,len(class_id)):

        sonuc+=(TP[i]/(TP[i]+FP[i]))

        

    sonuc=sonuc/2

    return sonuc



def recall_hesapla(class_id,TP, FP, TN, FN):

    sonuc=0

    for i in range(0,2):

        sonuc+=(TP[i]/(TP[i]+FN[i]))

       

    sonuc=sonuc/2

    return sonuc

def accuracy_hesapla(class_id,TP, FP, TN, FN):

    sonuc=0

    for i in range(0,1):

        sonuc+=((TP[i]+TN[i])/(TP[i]+FP[i]+TN[i]+FN[i]))

        

    sonuc=sonuc/2

    return sonuc

def specificity_hesapla(class_id,TP, FP, TN, FN):

    sonuc=0

    for i in range(0,2):

        sonuc+=(TN[i]/(FP[i]+TN[i]))

        

    sonuc=sonuc/2

    return sonuc

def NPV_hesapla(class_id,TP, FP, TN, FN):

    sonuc=0

    for i in range(0,2):

        sonuc+=(TN[i]/(TN[i]+FN[i]))

        

    sonuc=sonuc/2

    return sonuc

def perf_measure(y_actual, y_pred):

    class_id = set(y_actual).union(set(y_pred))

    TP = []

    FP = []

    TN = []

    FN = []



    for index ,_id in enumerate(class_id):

        TP.append(0)

        FP.append(0)

        TN.append(0)

        FN.append(0)

        for i in range(len(y_pred)):

            if y_actual[i] == y_pred[i] == _id:

                TP[index] += 1

            if y_pred[i] == _id and y_actual[i] != y_pred[i]:

                FP[index] += 1

            if y_actual[i] == y_pred[i] != _id:

                TN[index] += 1

            if y_pred[i] != _id and y_actual[i] != y_pred[i]:

                FN[index] += 1





    return class_id,TP, FP, TN, FN
df=pd.read_csv("../input/tictactoe-endgame-dataset-uci/tic-tac-toe-endgame.csv")
df.head()
df.V10.value_counts()
df.info()
mapping_for_moves = {'x':1, "o":0}
mapping_for_moves
mapping_for_wins = {"positive":1, "negative":0}
df.V10 = df.V10.map(mapping_for_wins)
df
x=df.V10
y=df.drop(columns=["V10"],axis=1)
y.columns
x
for i in y.columns:

    y[i] = y[i].map(mapping_for_moves)
y
features = y.values

labels = x.values
from sklearn.preprocessing import Imputer



features = (Imputer().fit_transform(features))

features
len(features) 

len(x)
features = features.astype(np.int)

labels = labels.astype(np.int)
from sklearn.metrics import classification_report,precision_score,recall_score,f1_score,roc_auc_score,accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



x_train, x_test, y_train, y_test = train_test_split(features, labels, random_state=3, shuffle=True)
score_liste=[]

auc_scor=[]

precision_scor=[]

recall_scor=[]

f1_scor=[]

LR_plus=[]

LR_eksi=[]

odd_scor=[]

NPV_scor=[]

youden_scor=[]

specificity_scor=[]

from sklearn.neighbors import KNeighborsClassifier
k=10

from sklearn.metrics import roc_curve

knn = KNeighborsClassifier(n_neighbors = k)

knn.fit(x_train,y_train)

y_head=knn.predict(x_test)

print("KNN Algoritması başarım sonucu: ",knn.score(x_test,y_test))



from sklearn.metrics import confusion_matrix

cmknn = confusion_matrix(y_test,y_head)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cmknn,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.title("KNN Algoritması Karmaşıklık Matrisi")

plt.show()



classid,tn,fp,fn,tp=perf_measure(y_test,y_head)

auc_scor.append(roc_auc_score(y_test,y_head))

knn_fpr,knn_tpr,knn_trr=roc_curve(y_test,y_head)

score_liste.append(accuracy_hesapla(classid,tn,fp,fn,tp))

precision_scor.append(precision_hesapla(classid,tn,fp,fn,tp))

recall_scor.append(recall_hesapla(classid,tn,fp,fn,tp))

f1_scor.append(f1_score(y_test,y_head,average='macro'))

NPV_scor.append(NPV_hesapla(classid,tn,fp,fn,tp))

specificity_scor.append(specificity_hesapla(classid,tn,fp,fn,tp))



LR_plus.append((recall_hesapla(classid,tn,fp,fn,tp)/(1-specificity_hesapla(classid,tn,fp,fn,tp))))

LR_eksi.append(((1-recall_hesapla(classid,tn,fp,fn,tp))/specificity_hesapla(classid,tn,fp,fn,tp)))

odd_scor.append(((recall_hesapla(classid,tn,fp,fn,tp)/(1-specificity_hesapla(classid,tn,fp,fn,tp))))/(((1-recall_hesapla(classid,tn,fp,fn,tp))/specificity_hesapla(classid,tn,fp,fn,tp))))

youden_scor.append((recall_hesapla(classid,tn,fp,fn,tp)+specificity_hesapla(classid,tn,fp,fn,tp)-1))

print("KNN algoritması için sınıflandırma raporu: \n",classification_report(y_test,y_head))
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier()

dtc.fit(x_train,y_train)

y_head=dtc.predict(x_test)

print("Karar Ağaçları Algoritması için başarım sonucu: ",dtc.score(x_test,y_test))



classid,tn,fp,fn,tp=perf_measure(y_test,y_head)

auc_scor.append(roc_auc_score(y_test,y_head))

dtc_fpr,dtc_tpr,dtc_trr=roc_curve(y_test,y_head)

score_liste.append(accuracy_hesapla(classid,tn,fp,fn,tp))

precision_scor.append(precision_hesapla(classid,tn,fp,fn,tp))

recall_scor.append(recall_hesapla(classid,tn,fp,fn,tp))

f1_scor.append(f1_score(y_test,y_head,average='macro'))

NPV_scor.append(NPV_hesapla(classid,tn,fp,fn,tp))

specificity_scor.append(specificity_hesapla(classid,tn,fp,fn,tp))

TPR=recall_hesapla(classid,tn,fp,fn,tp)

TNR=specificity_hesapla(classid,tn,fp,fn,tp)

FPR=1-TNR

if FPR==0:

    FPR=0.00001

FNR=1-TPR

lreksi=FNR/TNR

lrarti=TPR/FPR

if lreksi==0:

    lreksi=0.00000001

LR_plus.append(TPR/FPR)

LR_eksi.append(FNR/TNR)

odd_scor.append(lrarti/lreksi)

youden_scor.append(TPR+TNR-1)



print("DTC algoritması için sınıflandırma raporu: \n",classification_report(y_test,y_head))



cmdtc = confusion_matrix(y_test,y_head)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cmdtc,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("Tahminde edilen değer")

plt.ylabel("Gerçek Değer")

plt.title("Karar Ağaçları Algoritması Karmaşıklık Matrisi")

plt.show()
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=225,random_state=1)

rfc.fit(x_train,y_train)

y_head=rfc.predict(x_test)

print("Rastgele Orman Algoritması başarım sonucu: ",rfc.score(x_test,y_test))



classid,tn,fp,fn,tp=perf_measure(y_test,y_head)

auc_scor.append(roc_auc_score(y_test,y_head))

rfc_fpr,rfc_tpr,rfc_trr=roc_curve(y_test,y_head)

score_liste.append(accuracy_hesapla(classid,tn,fp,fn,tp))

precision_scor.append(precision_hesapla(classid,tn,fp,fn,tp))

recall_scor.append(recall_hesapla(classid,tn,fp,fn,tp))

f1_scor.append(f1_score(y_test,y_head,average='macro'))

NPV_scor.append(NPV_hesapla(classid,tn,fp,fn,tp))

specificity_scor.append(specificity_hesapla(classid,tn,fp,fn,tp))

TPR=recall_hesapla(classid,tn,fp,fn,tp)

TNR=specificity_hesapla(classid,tn,fp,fn,tp)

FPR=1-TNR

if FPR==0:

    FPR=0.00001

FNR=1-TPR

lreksi=FNR/TNR

lrarti=TPR/FPR

if lreksi==0:

    lreksi=0.00000001

LR_plus.append(TPR/FPR)

LR_eksi.append(FNR/TNR)

odd_scor.append(lrarti/lreksi)

youden_scor.append(TPR+TNR-1)

print("Rastgele Orman algoritması için sınıflandırma raporu: \n",classification_report(y_test,y_head))



cmrfc = confusion_matrix(y_test,y_head)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cmrfc,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.title("Rastgele Orman Algoritması Karmaşıklık Matrisi")

plt.show()
from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

nb.fit(x_train,y_train)

y_head=nb.predict(x_test)

print("Naive Bayes Algoritması başarım sonucu: ",nb.score(x_test,y_test))





classid,tn,fp,fn,tp=perf_measure(y_test,y_head)

auc_scor.append(roc_auc_score(y_test,y_head))

nb_fpr,nb_tpr,nb_trr=roc_curve(y_test,y_head)

score_liste.append(accuracy_hesapla(classid,tn,fp,fn,tp))

precision_scor.append(precision_hesapla(classid,tn,fp,fn,tp))

recall_scor.append(recall_hesapla(classid,tn,fp,fn,tp))

f1_scor.append(f1_score(y_test,y_head,average='macro'))

NPV_scor.append(NPV_hesapla(classid,tn,fp,fn,tp))

specificity_scor.append(specificity_hesapla(classid,tn,fp,fn,tp))

TPR=recall_hesapla(classid,tn,fp,fn,tp)

TNR=specificity_hesapla(classid,tn,fp,fn,tp)

FPR=1-TNR

if FPR==0:

    FPR=0.00001

FNR=1-TPR

lreksi=FNR/TNR

lrarti=TPR/FPR

if lreksi==0:

    lreksi=0.00000001

LR_plus.append(TPR/FPR)

LR_eksi.append(FNR/TNR)

odd_scor.append(lrarti/lreksi)

youden_scor.append(TPR+TNR-1)

print("Naive Bayes algoritması için sınıflandırma raporu: \n",classification_report(y_test,y_head))



cmnb = confusion_matrix(y_test,y_head)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cmnb,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("Tahmin Edilen")

plt.ylabel("Gerçek Değer")

plt.title("Naive Bayes Algoritması Karmaşıklık Matrisi")

plt.show()
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x_train,y_train)

y_head=lr.predict(x_test)

print("Logistic Regresyon Algoritması başarım sonucu: ",lr.score(x_test,y_test))





classid,tn,fp,fn,tp=perf_measure(y_test,y_head)

auc_scor.append(roc_auc_score(y_test,y_head))

lrc_fpr,lrc_tpr,lrc_trr=roc_curve(y_test,y_head)

score_liste.append(accuracy_hesapla(classid,tn,fp,fn,tp))

precision_scor.append(precision_hesapla(classid,tn,fp,fn,tp))

recall_scor.append(recall_hesapla(classid,tn,fp,fn,tp))

f1_scor.append(f1_score(y_test,y_head,average='macro'))

NPV_scor.append(NPV_hesapla(classid,tn,fp,fn,tp))

specificity_scor.append(specificity_hesapla(classid,tn,fp,fn,tp))

TPR=recall_hesapla(classid,tn,fp,fn,tp)

TNR=specificity_hesapla(classid,tn,fp,fn,tp)

FPR=1-TNR

if FPR==0:

    FPR=0.00001

FNR=1-TPR

lreksi=FNR/TNR

lrarti=TPR/FPR

if lreksi==0:

    lreksi=0.00000001

LR_plus.append(TPR/FPR)

LR_eksi.append(FNR/TNR)

odd_scor.append(lrarti/lreksi)

youden_scor.append(TPR+TNR-1)

print("Lojistik Regresyon algoritması için sınıflandırma raporu: \n",classification_report(y_test,y_head))



cmlr = confusion_matrix(y_test,y_head)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cmlr,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("Tahmin Edilen")

plt.ylabel("Gerçek Değer")

plt.title("Lojistik Regresyon Algoritması Karmaşıklık Matrisi")

plt.show()
from sklearn.svm import SVC

svc=SVC(random_state=1)

svc.fit(x_train,y_train)

y_head=svc.predict(x_test)

print("Destek Vektör Makineleri Algoritması başarım sonucu: ",svc.score(x_test,y_test))





classid,tn,fp,fn,tp=perf_measure(y_test,y_head)

auc_scor.append(roc_auc_score(y_test,y_head))

svc_fpr,svc_tpr,svc_trr=roc_curve(y_test,y_head)

score_liste.append(accuracy_hesapla(classid,tn,fp,fn,tp))

precision_scor.append(precision_hesapla(classid,tn,fp,fn,tp))

recall_scor.append(recall_hesapla(classid,tn,fp,fn,tp))

f1_scor.append(f1_score(y_test,y_head,average='macro'))

NPV_scor.append(NPV_hesapla(classid,tn,fp,fn,tp))

specificity_scor.append(specificity_hesapla(classid,tn,fp,fn,tp))

TPR=recall_hesapla(classid,tn,fp,fn,tp)

TNR=specificity_hesapla(classid,tn,fp,fn,tp)

FPR=1-TNR

if FPR==0:

    FPR=0.00001

FNR=1-TPR

lreksi=FNR/TNR

lrarti=TPR/FPR

if lreksi==0:

    lreksi=0.00000001

LR_plus.append(TPR/FPR)

LR_eksi.append(FNR/TNR)

odd_scor.append(lrarti/lreksi)

youden_scor.append(TPR+TNR-1)

print("Destek Vektör Makineleri algoritması için sınıflandırma raporu: \n",classification_report(y_test,y_head))



cmsvc = confusion_matrix(y_test,y_head)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cmsvc,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("Tahmin Edilen")

plt.ylabel("Gerçek Değer")

plt.title("Destek Vektör Makineleri Algoritması Karmaşıklık Matrisi")

plt.show()
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier

gfc=GradientBoostingClassifier(n_estimators= 1000, max_leaf_nodes= 4, max_depth=None,random_state= 2,min_samples_split= 5)

gfc.fit(x_train,y_train)

y_head=gfc.predict(x_test)

print("Gradient Boosting Classifier Algoritması başarım sonucu: ",gfc.score(x_test,y_test))



classid,tn,fp,fn,tp=perf_measure(y_test,y_head)

auc_scor.append(roc_auc_score(y_test,y_head))

gfc_fpr,gfc_tpr,gfc_trr=roc_curve(y_test,y_head)

score_liste.append(accuracy_hesapla(classid,tn,fp,fn,tp))

precision_scor.append(precision_hesapla(classid,tn,fp,fn,tp))

recall_scor.append(recall_hesapla(classid,tn,fp,fn,tp))

f1_scor.append(f1_score(y_test,y_head,average='macro'))

NPV_scor.append(NPV_hesapla(classid,tn,fp,fn,tp))

specificity_scor.append(specificity_hesapla(classid,tn,fp,fn,tp))

TPR=recall_hesapla(classid,tn,fp,fn,tp)

TNR=specificity_hesapla(classid,tn,fp,fn,tp)

FPR=1-TNR

if FPR==0:

    FPR=0.00001

FNR=1-TPR

lreksi=FNR/TNR

lrarti=TPR/FPR

if lreksi==0:

    lreksi=0.00000001

LR_plus.append(TPR/FPR)

LR_eksi.append(FNR/TNR)

odd_scor.append(lrarti/lreksi)

youden_scor.append(TPR+TNR-1)

print("Gradient Boosting Classifier algoritması için sınıflandırma raporu: \n",classification_report(y_test,y_head))



cmgfc = confusion_matrix(y_test,y_head)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cmgfc,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("Tahmin Edilen")

plt.ylabel("Gerçek Değer")

plt.title("Gradient Boosting Classifier Algoritması Karmaşıklık Matrisi")

plt.show()
abc=AdaBoostClassifier(n_estimators=100, random_state=0)

abc.fit(x_train,y_train)

y_head=abc.predict(x_test)

print("AdaBoosting Classifier Algoritması başarım sonucu: ",abc.score(x_test,y_test))





classid,tn,fp,fn,tp=perf_measure(y_test,y_head)

auc_scor.append(roc_auc_score(y_test,y_head))

abc_fpr,abc_tpr,abc_trr=roc_curve(y_test,y_head)

score_liste.append(accuracy_hesapla(classid,tn,fp,fn,tp))

precision_scor.append(precision_hesapla(classid,tn,fp,fn,tp))

recall_scor.append(recall_hesapla(classid,tn,fp,fn,tp))

f1_scor.append(f1_score(y_test,y_head,average='macro'))

NPV_scor.append(NPV_hesapla(classid,tn,fp,fn,tp))

specificity_scor.append(specificity_hesapla(classid,tn,fp,fn,tp))

TPR=recall_hesapla(classid,tn,fp,fn,tp)

TNR=specificity_hesapla(classid,tn,fp,fn,tp)

FPR=1-TNR

if FPR==0:

    FPR=0.00001

FNR=1-TPR

lreksi=FNR/TNR

lrarti=TPR/FPR

if lreksi==0:

    lreksi=0.00000001

LR_plus.append(TPR/FPR)

LR_eksi.append(FNR/TNR)

odd_scor.append(lrarti/lreksi)

youden_scor.append(TPR+TNR-1)

print("AdaBoosting Classifier algoritması için sınıflandırma raporu: \n",classification_report(y_test,y_head))



cmabc = confusion_matrix(y_test,y_head)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cmabc,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("Tahmin Edilen")

plt.ylabel("Gerçek Değer")

plt.title("AdaBoosting Classifier Algoritması Karmaşıklık Matrisi")

plt.show()
ysa=MLPClassifier(alpha=1, max_iter=1000)

ysa.fit(x_train,y_train)

y_head=ysa.predict(x_test)

print("Yapay Sinir Ağları Algoritması başarım sonucu: ",ysa.score(x_test,y_test))





classid,tn,fp,fn,tp=perf_measure(y_test,y_head)

auc_scor.append(roc_auc_score(y_test,y_head))

ysa_fpr,ysa_tpr,ysa_trr=roc_curve(y_test,y_head)

score_liste.append(accuracy_hesapla(classid,tn,fp,fn,tp))

precision_scor.append(precision_hesapla(classid,tn,fp,fn,tp))

recall_scor.append(recall_hesapla(classid,tn,fp,fn,tp))

f1_scor.append(f1_score(y_test,y_head,average='macro'))

NPV_scor.append(NPV_hesapla(classid,tn,fp,fn,tp))

specificity_scor.append(specificity_hesapla(classid,tn,fp,fn,tp))

TPR=recall_hesapla(classid,tn,fp,fn,tp)

TNR=specificity_hesapla(classid,tn,fp,fn,tp)

FPR=1-TNR

if FPR==0:

    FPR=0.00001

FNR=1-TPR

lreksi=FNR/TNR

lrarti=TPR/FPR

if lreksi==0:

    lreksi=0.00000001

LR_plus.append(TPR/FPR)

LR_eksi.append(FNR/TNR)

odd_scor.append(lrarti/lreksi)

youden_scor.append(TPR+TNR-1)

print("AdaBoosting Classifier algoritması için sınıflandırma raporu: \n",classification_report(y_test,y_head))



cmysa = confusion_matrix(y_test,y_head)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cmysa,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("Tahmin Edilen")

plt.ylabel("Gerçek Değer")

plt.title("AdaBoosting Classifier Algoritması Karmaşıklık Matrisi")

plt.show()
algo_liste=["KNN","Decision Tree","Random Forest","Naive Bayes","Linear Regression","Support Vector Machine","Gradient Boosting Classifier","AdaBoosting Classifier","Neural Network"]

score={"algo_list":algo_liste,"score_liste":score_liste,"precision":precision_scor,"recall":recall_scor,"f1_score":f1_scor,"AUC":auc_scor,"LR+":LR_plus,"LR-":LR_eksi,"ODD":odd_scor,"YOUDEN":youden_scor,"Specificity":specificity_scor}
df=pd.DataFrame(score)
df
f,ax1 = plt.subplots(figsize =(15,15))

sns.pointplot(x=df['algo_list'], y=df['score_liste'],data=df,color='lime',alpha=0.8,label="score_liste")

sns.pointplot(x=df['algo_list'], y=df['precision'],data=df,color='red',alpha=0.8,label="precision")

sns.pointplot(x=df['algo_list'], y=df['recall'],data=df,color='black',alpha=0.8,label="recall")

sns.pointplot(x=df['algo_list'], y=df['f1_score'],data=df,color='blue',alpha=0.8,label="f1_score")

sns.pointplot(x=df['algo_list'], y=df['AUC'],data=df,color='yellow',alpha=0.8,label="AUC")



sns.pointplot(x=df['algo_list'], y=df['LR-'],data=df,color='orange',alpha=0.8,label="YOUDEN")



sns.pointplot(x=df['algo_list'], y=df['YOUDEN'],data=df,color='brown',alpha=0.8,label="LR-")

sns.pointplot(x=df['algo_list'], y=df['Specificity'],data=df,color='purple',alpha=0.8,label="Specificity")

plt.xlabel('Algoritma ismi',fontsize = 15,color='blue')

plt.ylabel('Score',fontsize = 15,color='blue')

plt.xticks(rotation= 45)

plt.title('Tic-Tac-Toe Dataset ile Sınıflandırma',fontsize = 20,color='blue')

plt.grid()

plt.legend()

plt.show()
def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr,rfc_fpr,rfc_tpr,nb_fpr,nb_tpr,gbc_fpr,gbc_tpr,abc_fpr,abc_tpr,ysa_fpr,ysa_tpr):

    plt.figure(figsize=(16,8))

    plt.title('ROC Curve \n Top 9 Classifiers', fontsize=18)

    plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_test, lr.predict(x_test))))

    plt.plot(knear_fpr, knear_tpr, label='KNears Neighbors Classifier Score: {:.4f}'.format(roc_auc_score(y_test, knn.predict(x_test))))

    plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(y_test, svc.predict(x_test))))

    plt.plot(tree_fpr, tree_tpr, label='Decision Tree Classifier Score: {:.4f}'.format(roc_auc_score(y_test, dtc.predict(x_test))))

    plt.plot(rfc_fpr, rfc_tpr, label='Random Forest Classifier Score: {:.4f}'.format(roc_auc_score(y_test, rfc.predict(x_test))))

    plt.plot(nb_fpr, nb_tpr, label='Naive Bayes Classifier Score: {:.4f}'.format(roc_auc_score(y_test, nb.predict(x_test))))

    plt.plot(gbc_fpr, gbc_tpr, label='Gradient Boosting Classifier Score: {:.4f}'.format(roc_auc_score(y_test, gfc.predict(x_test))))

    plt.plot(abc_fpr, abc_tpr, label='AdaBoosting Classifier Score: {:.4f}'.format(roc_auc_score(y_test, abc.predict(x_test))))

    plt.plot(ysa_fpr, ysa_tpr, label='Neural Network Score: {:.4f}'.format(roc_auc_score(y_test, ysa.predict(x_test))))

    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([-0.01, 1, 0, 1])

    plt.xlabel('False Positive Rate', fontsize=16)

    plt.ylabel('True Positive Rate', fontsize=16)

    plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),

                arrowprops=dict(facecolor='#6E726D', shrink=0.05),

                )

    plt.legend()

    

graph_roc_curve_multiple(lrc_fpr, lrc_tpr, knn_fpr, knn_tpr, svc_fpr, svc_tpr, dtc_fpr, dtc_tpr,rfc_fpr,rfc_tpr,nb_fpr,nb_tpr,gfc_fpr,gfc_tpr,abc_fpr,abc_tpr,ysa_fpr,ysa_tpr)

plt.show()