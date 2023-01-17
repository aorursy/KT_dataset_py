import pandas as pd

import numpy as np



# Plots

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.tools as tls

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)

import squarify



# Data processing, metrics and modeling

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.metrics import precision_score, recall_score, confusion_matrix,  roc_curve, precision_recall_curve, accuracy_score, roc_auc_score

import lightgbm as lgbm

from sklearn.ensemble import VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_curve,auc

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_predict

from yellowbrick.classifier import DiscriminationThreshold



# Stats

import scipy.stats as ss

from scipy import interp

from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform



# Time

from contextlib import contextmanager

@contextmanager

def timer(title):

    t0 = time.time()

    yield

    print("{} - done in {:.0f}s".format(title, time.time() - t0))



#ignore warning messages 

import warnings

warnings.filterwarnings('ignore') 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def precision_hesapla(class_id,TP, FP, TN, FN):

    sonuc=0

    for i in range(0,len(class_id)):

        if (TP[i]==0 or FP[i]==0):

            TP[i]=0.00000001

            FP[i]=0.00000001

        sonuc+=(TP[i]/(TP[i]+FP[i]))

        

    sonuc=sonuc/len(class_id)

    return sonuc



def recall_hesapla(class_id,TP, FP, TN, FN):

    sonuc=0

    for i in range(0,len(class_id)):

        if (TP[i]==0 or FN[i]==0):

            TP[i]=0.00000001

            FN[i]=0.00000001

        sonuc+=(TP[i]/(TP[i]+FN[i]))

       

    sonuc=sonuc/len(class_id)

    return sonuc

def accuracy_hesapla(class_id,TP, FP, TN, FN):

    sonuc=0

    for i in range(0,len(class_id)):

        sonuc+=((TP[i]+TN[i])/(TP[i]+FP[i]+TN[i]+FN[i]))

        

    sonuc=sonuc/len(class_id)

    return sonuc

def specificity_hesapla(class_id,TP, FP, TN, FN):

    sonuc=0

    for i in range(0,len(class_id)):

        if (TN[i]==0 or FP[i]==0):

            TN[i]=0.00000001

            FP[i]=0.00000001

        sonuc+=(TN[i]/(FP[i]+TN[i]))

        

    sonuc=sonuc/len(class_id)

    return sonuc

def NPV_hesapla(class_id,TP, FP, TN, FN):

    sonuc=0

    for i in range(0,len(class_id)):

        if (TN[i]==0 or FN[i]==0):

            TN[i]=0.00000001

            FN[i]=0.00000001

        sonuc+=(TN[i]/(TN[i]+FN[i]))

        

    sonuc=sonuc/len(class_id)

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

df = pd.read_csv("../input/portuguese-bank-marketing-data-set/bank_cleaned.csv")

df = df.drop(['Unnamed: 0'], axis=1)

df.drop(['marital'],axis=1, inplace=True)

df1 = df.iloc[:, 0:7]
df2 = pd.get_dummies(df1, columns = ['job'])

df2 = pd.get_dummies(df2, columns = ['education'])

df2['housing'] = df2['housing'].map({'yes': 1, 'no': 0})

df2['default'] = df2['default'].map({'yes': 1, 'no': 0})

df2['loan'] = df2['loan'].map({'yes': 1, 'no': 0})

df_response = pd.DataFrame(df['response_binary'])

df2 = pd.merge(df2, df_response, left_index = True, right_index = True)
array = df2.values



# Features: first 20 columns

x = array[:,0:-1]



# Target variable: 'response_binary'

y = array[:,-1]
from sklearn.metrics import classification_report,precision_score,recall_score,f1_score,roc_auc_score,accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import cross_val_score
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

from sklearn.metrics import roc_curve
def model_olustur2(isim,model,x,y):

    

    y_head=cross_val_predict(model,x,y,cv=10)

    print(isim," Algoritması başarım sonucu: ",np.mean(cross_val_score(model,x,y,cv=10)))

    cm = confusion_matrix(y,y_head)

    f, ax = plt.subplots(figsize =(5,5))

    sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

    plt.xlabel("Tahmin Edilen Değer")

    plt.ylabel("Gerçek Değer")

    baslik=isim+" Algoritması Karmaşıklık Matrisi"

    plt.title(baslik)

    plt.show()

    classid,tn,fp,fn,tp=perf_measure(y,y_head)

    auc_scor.append(roc_auc_score(y,y_head))

    fpr,tpr,trr=roc_curve(y,y_head)

    score_liste.append(accuracy_hesapla(classid,tn,fp,fn,tp))

    precision_scor.append(precision_hesapla(classid,tn,fp,fn,tp))

    recall_scor.append(recall_hesapla(classid,tn,fp,fn,tp))

    f1_scor.append(f1_score(y,y_head,average='macro'))

    NPV_scor.append(NPV_hesapla(classid,tn,fp,fn,tp))

    specificity_scor.append(specificity_hesapla(classid,tn,fp,fn,tp))



    LR_plus.append((recall_hesapla(classid,tn,fp,fn,tp)/(1-specificity_hesapla(classid,tn,fp,fn,tp))))

    LR_eksi.append(((1-recall_hesapla(classid,tn,fp,fn,tp))/specificity_hesapla(classid,tn,fp,fn,tp)))

    odd_scor.append(((recall_hesapla(classid,tn,fp,fn,tp)/(1-specificity_hesapla(classid,tn,fp,fn,tp))))/(((1-recall_hesapla(classid,tn,fp,fn,tp))/specificity_hesapla(classid,tn,fp,fn,tp))))

    youden_scor.append((recall_hesapla(classid,tn,fp,fn,tp)+specificity_hesapla(classid,tn,fp,fn,tp)-1))

    print(isim," algoritması için sınıflandırma raporu: \n",classification_report(y,y_head))

    return cm,fpr,tpr,trr
knn=KNeighborsClassifier(n_neighbors=8)

cmknn,knn_fpr,knn_tpr,knn_trr=model_olustur2("KNN",knn,x,y)
dtc=DecisionTreeClassifier()

cmdtc,dtc_fpr,dtc_tpr,dtc_trr=model_olustur2("Karar Ağaçları",dtc,x,y)
rfc=RandomForestClassifier(n_estimators=225,random_state=1)

cmrfc,rfc_fpr,rfc_tpr,rfc_trr=model_olustur2("Rastgele Orman",rfc,x,y)
nb=GaussianNB()

cmnb,nb_fpr,nb_tpr,nb_trr=model_olustur2("Naive Bayes",nb,x,y)
lrc=LogisticRegression()

cmlrc,lrc_fpr,lrc_tpr,lrc_trr=model_olustur2("Lojistik Regresyon",lrc,x,y)
svc=SVC(random_state=1)

cmsvc,svc_fpr,svc_tpr,svc_trr=model_olustur2("Destek Vektör Makineleri",svc,x,y)
gfc=GradientBoostingClassifier(n_estimators= 1000, max_leaf_nodes= 4, max_depth=None,random_state= 2,min_samples_split= 5)

cmgfc,gfc_fpr,gfc_tpr,gfc_trr=model_olustur2("Gradien Boosting",gfc,x,y)
abc=AdaBoostClassifier(n_estimators=100, random_state=0)

cmabc,abc_fpr,abc_tpr,abc_trr=model_olustur2("AdaBoosting",abc,x,y)
ysa=MLPClassifier(alpha=1, max_iter=1000)

cmysa,ysa_fpr,ysa_tpr,ysa_trr=model_olustur2("Yapay Sinir Ağları",ysa,x,y)
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

plt.title('Portuguese Bank Marketing Dataset ile Sınıflandırma',fontsize = 20,color='blue')

plt.grid()

plt.legend()

plt.show()
def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr,rfc_fpr,rfc_tpr,nb_fpr,nb_tpr,gbc_fpr,gbc_tpr,abc_fpr,abc_tpr,ysa_fpr,ysa_tpr):

    plt.figure(figsize=(16,8))

    plt.title('ROC Curve \n Top 9 Classifiers', fontsize=18)

    plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y, cross_val_predict(lrc,x,y,cv=10))))

    plt.plot(knear_fpr, knear_tpr, label='KNears Neighbors Classifier Score: {:.4f}'.format(roc_auc_score(y, cross_val_predict(knn,x,y,cv=10))))

    plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(y, cross_val_predict(svc,x,y,cv=10))))

    plt.plot(tree_fpr, tree_tpr, label='Decision Tree Classifier Score: {:.4f}'.format(roc_auc_score(y, cross_val_predict(dtc,x,y,cv=10))))

    plt.plot(rfc_fpr, rfc_tpr, label='Random Forest Classifier Score: {:.4f}'.format(roc_auc_score(y, cross_val_predict(rfc,x,y,cv=10))))

    plt.plot(nb_fpr, nb_tpr, label='Naive Bayes Classifier Score: {:.4f}'.format(roc_auc_score(y, cross_val_predict(nb,x,y,cv=10))))

    plt.plot(gbc_fpr, gbc_tpr, label='Gradient Boosting Classifier Score: {:.4f}'.format(roc_auc_score(y, cross_val_predict(gfc,x,y,cv=10))))

    plt.plot(abc_fpr, abc_tpr, label='AdaBoosting Classifier Score: {:.4f}'.format(roc_auc_score(y, cross_val_predict(abc,x,y,cv=10))))

    plt.plot(ysa_fpr, ysa_tpr, label='Neural Network Score: {:.4f}'.format(roc_auc_score(y, cross_val_predict(ysa,x,y,cv=10))))

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