# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import missingno

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
df=pd.read_csv("../input/travel-insurance/travel insurance.csv")

df1=df

df.head(5)
len(df['Destination'].unique().tolist())
df.info()
missingno.matrix(df)
df['Gender'].isnull().sum()
df.fillna('Not Specified',inplace=True)
df.isnull().sum()
df_numerical=df._get_numeric_data()

df_numerical.info()
for i, col in enumerate(df_numerical.columns):

    plt.figure(i)

    sns.distplot(df_numerical[col])
df['Duration'].describe()
df10=df['Duration']<0

df10.sum()
df.loc[df['Duration'] < 0, 'Duration'] = 49.317
df6= df['Net Sales']<df['Commision (in value)']

df6.sum()
df.loc[df['Net Sales'] == 0.0, 'Commision (in value)'] = 0
test=[(df[df['Gender']=='Not Specified']['Claim'].value_counts()/len(df[df['Gender']=='Not Specified']['Claim']))[1],(df[df['Gender']=='M']['Claim'].value_counts()/len(df[df['Gender']=='M']['Claim']))[1],

      (df[df['Gender']=='F']['Claim'].value_counts()/len(df[df['Gender']=='F']['Claim']))[1]]

test
fig, axes=plt.subplots(1,3,figsize=(24,9))

sns.countplot(df[df['Gender']=='Not Specified']['Claim'],ax=axes[0])

axes[0].set(title='Distribution of claims for null gender')

axes[0].text(x=1,y=30000,s=f'% of 1 class: {round(test[0],2)}',fontsize=16,weight='bold',ha='center',va='bottom',color='navy')

sns.countplot(df[df['Gender']=='M']['Claim'],ax=axes[1])

axes[1].set(title='Distribution of claims for Male')

axes[1].text(x=1,y=6000,s=f'% of 1 class: {round(test[1],2)}',fontsize=16,weight='bold',ha='center',va='bottom',color='navy')

sns.countplot(df[df['Gender']=='F']['Claim'],ax=axes[2])

axes[2].set(title='Distribution of claims for Female')

axes[2].text(x=1,y=6000,s=f'% of 1 class: {round(test[2],2)}',fontsize=16,weight='bold',ha='center',va='bottom',color='navy')

plt.show()
pd.crosstab(df['Agency'],df['Agency Type'],margins=True)
table1=pd.crosstab(df['Agency'],df['Claim'],margins=True)



table1.drop(index=['All'],inplace=True)

table1=(table1.div(table1['All'],axis=0))*100



table1['mean commision']=df.groupby('Agency')['Commision (in value)'].mean()

table1
table1.columns
fig,ax1=plt.subplots(figsize=(18,9))

sns.barplot(table1.index,table1.Yes,ax=ax1)

plt.xticks(rotation=90)

ax1.set(ylabel='Acceptance %')

ax2=ax1.twinx()

sns.lineplot(table1.index,table1['mean commision'],ax=ax2,linewidth=3)
table2=pd.crosstab(df['Product Name'],df['Claim'],margins=True)

table2=(table2.div(table2['All'],axis=0))*100



table2['mean commision']=df.groupby('Product Name')['Commision (in value)'].mean()

table2.drop(index=['All'],inplace=True)

table2
fig,ax1=plt.subplots(figsize=(20,11))

sns.barplot(table2.index,table2.Yes,ax=ax1)

plt.xticks(rotation=90)

ax1.set(ylabel='Acceptance %')

ax2=ax1.twinx()

sns.lineplot(table2.index,table2['mean commision'],ax=ax2,linewidth=3)
tests=df.copy()

tests['Duration_label']=pd.qcut(df['Duration'],q=35)

table3=pd.crosstab(tests['Duration_label'],tests['Claim'],normalize='index')

table3
table3.columns
plt.figure(figsize=(10,7))

sns.barplot(table3.index,table3.Yes)

plt.xticks(rotation=90)
table4=pd.crosstab(df['Destination'],df['Claim'],margins=True,normalize='index')

table4
table4 = table4.sort_values(by=['Yes'], ascending=[False])

table4
sns.countplot(df['Claim'])
from scipy.stats import chi2_contingency



class ChiSquare:

    def __init__(self, df):

        self.df = df

        self.p = None #P-Value

        self.chi2 = None #Chi Test Statistic

        self.dof = None

        self.dfObserved = None

        self.dfExpected = None

        

    def _print_chisquare_result(self, colX, alpha):

        result = ""

        if self.p<alpha:

            result="{0} is IMPORTANT for Prediction".format(colX)

        else:

            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)



        print(result)

        

    def TestIndependence(self,colX,colY, alpha=0.05):

        X = self.df[colX].astype(str)

        Y = self.df[colY].astype(str)

        

        self.dfObserved = pd.crosstab(Y,X) 

        chi2, p, dof, expected = ss.chi2_contingency(self.dfObserved.values)

        self.p = p

        self.chi2 = chi2

        self.dof = dof 

        

        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)

        

        self._print_chisquare_result(colX,alpha)
X = df.drop(['Claim'], axis=1)
df.drop(columns=['Distribution Channel','Agency Type'],axis=1,inplace=True)
y=df['Claim']
x=df

x.drop(columns='Claim',axis=1,inplace=True)
x_dummy=pd.get_dummies(x,columns=['Agency','Gender','Product Name','Destination'],drop_first=True)
x=x_dummy
y
y.value_counts()
y = [ 1 if each == "Yes" else 0 for each in y]
x_data=x

x=(x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))
x
from sklearn.metrics import classification_report,precision_score,recall_score,f1_score,roc_auc_score,accuracy_score

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=1)
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

from sklearn.metrics import roc_curve

k=4

knn = KNeighborsClassifier(n_neighbors = k)

knn.fit(x_train,y_train)

y_head=knn.predict(x_test)

print("KNN Algoritmas?? ba??ar??m sonucu: ",knn.score(x_test,y_test))



from sklearn.metrics import confusion_matrix

cmknn = confusion_matrix(y_test,y_head)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cmknn,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.title("KNN Algoritmas?? Karma????kl??k Matrisi")

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

print("KNN algoritmas?? i??in s??n??fland??rma raporu: \n",classification_report(y_test,y_head))
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier()

dtc.fit(x_train,y_train)

y_head=dtc.predict(x_test)

print("Karar A??a??lar?? Algoritmas?? i??in ba??ar??m sonucu: ",dtc.score(x_test,y_test))



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



print("DTC algoritmas?? i??in s??n??fland??rma raporu: \n",classification_report(y_test,y_head))



cmdtc = confusion_matrix(y_test,y_head)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cmdtc,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("Tahminde edilen de??er")

plt.ylabel("Ger??ek De??er")

plt.title("Karar A??a??lar?? Algoritmas?? Karma????kl??k Matrisi")

plt.show()
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=225,random_state=1)

rfc.fit(x_train,y_train)

y_head=rfc.predict(x_test)

print("Rastgele Orman Algoritmas?? ba??ar??m sonucu: ",rfc.score(x_test,y_test))



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

print("Rastgele Orman algoritmas?? i??in s??n??fland??rma raporu: \n",classification_report(y_test,y_head))



cmrfc = confusion_matrix(y_test,y_head)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cmrfc,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.title("Rastgele Orman Algoritmas?? Karma????kl??k Matrisi")

plt.show()
from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

nb.fit(x_train,y_train)

y_head=nb.predict(x_test)

print("Naive Bayes Algoritmas?? ba??ar??m sonucu: ",nb.score(x_test,y_test))





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

print("Naive Bayes algoritmas?? i??in s??n??fland??rma raporu: \n",classification_report(y_test,y_head))



cmnb = confusion_matrix(y_test,y_head)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cmnb,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("Tahmin Edilen")

plt.ylabel("Ger??ek De??er")

plt.title("Naive Bayes Algoritmas?? Karma????kl??k Matrisi")

plt.show()
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x_train,y_train)

y_head=lr.predict(x_test)

print("Logistic Regresyon Algoritmas?? ba??ar??m sonucu: ",lr.score(x_test,y_test))





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

print("Lojistik Regresyon algoritmas?? i??in s??n??fland??rma raporu: \n",classification_report(y_test,y_head))



cmlr = confusion_matrix(y_test,y_head)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cmlr,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("Tahmin Edilen")

plt.ylabel("Ger??ek De??er")

plt.title("Lojistik Regresyon Algoritmas?? Karma????kl??k Matrisi")

plt.show()
from sklearn.svm import SVC

svc=SVC(random_state=1)

svc.fit(x_train,y_train)

y_head=svc.predict(x_test)

print("Destek Vekt??r Makineleri Algoritmas?? ba??ar??m sonucu: ",svc.score(x_test,y_test))





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

print("Destek Vekt??r Makineleri algoritmas?? i??in s??n??fland??rma raporu: \n",classification_report(y_test,y_head))



cmsvc = confusion_matrix(y_test,y_head)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cmsvc,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("Tahmin Edilen")

plt.ylabel("Ger??ek De??er")

plt.title("Destek Vekt??r Makineleri Algoritmas?? Karma????kl??k Matrisi")

plt.show()
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

from sklearn.neural_network import MLPClassifier
gfc=GradientBoostingClassifier(n_estimators= 1000, max_leaf_nodes= 4, max_depth=None,random_state= 2,min_samples_split= 5)

gfc.fit(x_train,y_train)

y_head=gfc.predict(x_test)

print("Gradient Boosting Classifier Algoritmas?? ba??ar??m sonucu: ",gfc.score(x_test,y_test))



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

print("Gradient Boosting Classifier algoritmas?? i??in s??n??fland??rma raporu: \n",classification_report(y_test,y_head))



cmgfc = confusion_matrix(y_test,y_head)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cmgfc,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("Tahmin Edilen")

plt.ylabel("Ger??ek De??er")

plt.title("Gradient Boosting Classifier Algoritmas?? Karma????kl??k Matrisi")

plt.show()
abc=AdaBoostClassifier(n_estimators=100, random_state=0)

abc.fit(x_train,y_train)

y_head=abc.predict(x_test)

print("AdaBoosting Classifier Algoritmas?? ba??ar??m sonucu: ",abc.score(x_test,y_test))





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

print("AdaBoosting Classifier algoritmas?? i??in s??n??fland??rma raporu: \n",classification_report(y_test,y_head))



cmabc = confusion_matrix(y_test,y_head)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cmabc,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("Tahmin Edilen")

plt.ylabel("Ger??ek De??er")

plt.title("AdaBoosting Classifier Algoritmas?? Karma????kl??k Matrisi")

plt.show()
ysa=MLPClassifier(alpha=1, max_iter=1000)

ysa.fit(x_train,y_train)

y_head=ysa.predict(x_test)

print("Yapay Sinir A??lar?? Algoritmas?? ba??ar??m sonucu: ",ysa.score(x_test,y_test))





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

print("AdaBoosting Classifier algoritmas?? i??in s??n??fland??rma raporu: \n",classification_report(y_test,y_head))



cmysa = confusion_matrix(y_test,y_head)

f, ax = plt.subplots(figsize =(5,5))

sns.heatmap(cmysa,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("Tahmin Edilen")

plt.ylabel("Ger??ek De??er")

plt.title("AdaBoosting Classifier Algoritmas?? Karma????kl??k Matrisi")

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

plt.title('HCC Survival Dataset ile S??n??fland??rma',fontsize = 20,color='blue')

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