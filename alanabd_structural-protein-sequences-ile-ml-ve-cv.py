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

        if TP[i]==0 or FP[i]==0:

            TP[i]=0.00000000001

            FP[i]=0.00000000001

        sonuc+=(TP[i]/(TP[i]+FP[i]))

        

    sonuc=sonuc/len(class_id)

    return sonuc



def recall_hesapla(class_id,TP, FP, TN, FN):

    sonuc=0

    for i in range(0,len(class_id)):

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

        sonuc+=(TN[i]/(FP[i]+TN[i]))

        

    sonuc=sonuc/len(class_id)

    return sonuc

def NPV_hesapla(class_id,TP, FP, TN, FN):

    sonuc=0

    for i in range(0,len(class_id)):

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
def yenimetot(y_test,y_score):

    from sklearn.preprocessing import label_binarize

    from sklearn.metrics import roc_curve, auc

    y_test = label_binarize(y_test, classes=[0,1,2])

    y_score = label_binarize(y_score, classes=[0,1,2])

    n_classes = 3

    fpr = dict()

    tpr = dict()

    thr = dict()

    roc_auc = dict()

    for i in range(n_classes):

        fpr[i], tpr[i], thr[i] = roc_curve(y_test[:, i], y_score[:, i])

        roc_auc[i] = auc(fpr[i], tpr[i])

    return roc_auc[2],fpr[2],tpr[2],thr[2]
df_dup = pd.read_csv('../input/protein-data-set/pdb_data_no_dups.csv')

df_seq = pd.read_csv('../input/protein-data-set/pdb_data_seq.csv')
df_merge = df_dup.merge(df_seq,how='inner',on='structureId')

df_merge.rename({'macromoleculeType_x':'macromoleculeType','residueCount_y':'residueCount'},axis=1,inplace=True)

df_merge.drop(['macromoleculeType_y','residueCount_x'],axis=1,inplace=True)
df_isnull = pd.DataFrame(round((df_merge.isnull().sum().sort_values(ascending=False)/df_merge.shape[0])*100,1)).reset_index()

df_isnull.columns = ['Columns', '% of Missing Data']

df_isnull.style.format({'% of Missing Data': lambda x:'{:.1%}'.format(abs(x))})

cm = sns.light_palette("skyblue", as_cmap=True)

df_isnull = df_isnull.style.background_gradient(cmap=cm)

df_isnull
df_pub_year = df_merge.dropna(subset=['publicationYear']) #dropping the missing values from the publicationYear only

#graph

x1= df_pub_year.publicationYear.value_counts().sort_index().index

y1 = df_pub_year.publicationYear.value_counts().sort_index().values
def ph_scale (ph):

    if ph < 7 :

        ph = 'Acidic'

    elif ph > 7:

        ph = 'Bacis'

    else:

        ph = 'Neutral'

    return ph

print('The pH Scale are group into 3 Categories: BASIC if [ pH > 7 ], ACIDIC if [ pH < 7 ] and NEUTRAL if pH [ is equal to 7 ]')



#Transform the dataset

df_ph = df_merge.dropna(subset=['phValue']) # dropping missing values in the phValue column only

df_ph['pH_scale'] = df_ph['phValue'].apply(ph_scale)

#Graph

labels= df_ph['pH_scale'].value_counts().index

values = df_ph['pH_scale'].value_counts().values
# The result of this cell Show the Top 10 most used crystallization method

df_cry_meth = df_merge.dropna(subset=['crystallizationMethod']) # this will drop all missing values in

#the crystallizationMethod column



cry_me = pd.DataFrame(df_cry_meth.crystallizationMethod.value_counts(ascending=False).head(10)).reset_index()

cry_me.columns = ['Crystallization Method','Values']



f,ax = plt.subplots(figsize=(10,8))

cry_me.plot(kind = 'barh',ax=ax,color='gray',legend=None,width= 0.8)

# get_width pulls left or right; get_y pushes up or down

for i in ax.patches:

    ax.text(i.get_width()+.1, i.get_y()+.40, \

            str(round((i.get_width()), 2)), fontsize=12, color='black',alpha=0.8)  

#Set ylabel

ax.set_yticklabels(cry_me['Crystallization Method'])

# invert for largest on top 

ax.invert_yaxis()

kwargs= {'length':3, 'width':1, 'colors':'black','labelsize':'large'}

ax.tick_params(**kwargs)

x_axis = ax.axes.get_xaxis().set_visible(False)

ax.set_title ('Top 10 Crystallization Method',color='black',fontsize=16)

sns.despine(bottom=True)
popular_exp_tech = df_merge.experimentalTechnique.value_counts()[:3] # Extract the 3 top used Exp Tech 

popular_exp_tech_df = pd.DataFrame(popular_exp_tech).reset_index()

popular_exp_tech_df.columns=['Experimental Technique','values']

# ADDING A ROW FOR THE ORTHER EXPERIMENTAL TECHNIQUE USED. PLEASE PUT IN MIND THAT TO ORTHER TECHNIQUES 

#IS JUST A GROUP OF THE REST OF THECNIQUES USED

popular_exp_tech_df.loc[3]  = ['OTHER TECHNIQUE', 449]

print ('The X-RAY DIFFRACTION is by far the most used Experimental Technique during the Study of the Protein Sequences')



labels = popular_exp_tech_df['Experimental Technique']

values = popular_exp_tech_df['values']

a = 'Exp Tech'
print ('There are more than 10 macro molecules used in this dataset but PROTEIN is widely used than the others')



ex = df_merge.macromoleculeType.value_counts()

a = 'Macro Mol Type'

colors = ['SlateGray','Orange','Green','DodgerBlue','DodgerBlue','DodgerBlue','DodgerBlue','DodgerBlue','DodgerBlue',

        'DodgerBlue','DodgerBlue','DodgerBlue','DodgerBlue']
#classification distribution

clasific =df_merge.classification.value_counts(ascending=False)

df_class = pd.DataFrame(round(((clasific/df_merge.shape[0])*100),2).head(10)).reset_index()

df_class.columns = ['Classification', 'percent_value']

print('There are {} Unique Classification Types and the top 10 Classification type accounts for more than 50% of the classification in the dataset'.format(df_merge.classification.nunique()))

f,ax = plt.subplots(figsize=(10,8))



df_class.plot(kind = 'barh',ax=ax,color='slategray',legend=None,width= 0.8)

# get_width pulls left or right; get_y pushes up or down

for i in ax.patches:

    ax.text(i.get_width()+.1, i.get_y()+.40, \

            str(round((i.get_width()), 2))+'%', fontsize=12, color='black',alpha=0.8)  

#Set ylabel

ax.set_yticklabels(df_class['Classification'])

# invert for largest on top 

ax.invert_yaxis()

kwargs= {'length':3, 'width':1, 'colors':'black','labelsize':'large'}

ax.tick_params(**kwargs)

x_axis = ax.axes.get_xaxis().set_visible(False)

ax.set_title ('Top 10 Classification Types',color='black',fontsize=16)

sns.despine(bottom=True)
df_class.Classification.values.tolist()[1:4]

# Reduce the df_merge to df_protein which is compose of macromolecule type [Protein and Protein#RNA]

macrotype = ['Protein','Protein#RNA']

df_protein = df_merge[(df_merge['experimentalTechnique'] =='X-RAY DIFFRACTION') & 

                      (df_merge['macromoleculeType'].isin(macrotype))&

                     (df_merge['classification'].isin(df_class.Classification.values.tolist()[1:4]))]



df_protein.reset_index(drop=True,inplace=True)

columns = ['crystallizationMethod' ,'pdbxDetails', 'publicationYear','phValue','crystallizationTempK']

#Dropping columns with missing value above 15%

df_protein.drop(columns=columns,inplace=True)

# Classification Type that will be used from now on

f,ax= plt.subplots(figsize=(10,5))

sns.countplot('classification',data=df_protein, ax=ax)

ax.set_title('Classification Types Selected',fontsize=14,color='black')

ax.tick_params(length =3,labelsize=11,color='black')

ax.set_xlabel('Classification',color='black',fontsize=13)

sns.despine()
from scipy import stats

from scipy.stats import norm, skew, kurtosis

def stat_kde_plot(input1,input2,input3):

    f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,5))

    sns.kdeplot(df_protein[input1],ax = ax1,color ='blue',shade=True,

                label=("Skewness : %.2f"%(df_protein[input1].skew()),

                       "Kurtosis: %.2f"%(df_protein[input1].kurtosis())))

    sns.kdeplot(df_protein[input2], ax = ax2,color='r',shade=True,

                label=("Skewness : %.2f"%(df_protein[input2].skew()),

                       "Kurtosis: %.2f"%(df_protein[input2].kurtosis())))

    sns.kdeplot(df_protein[input3], ax = ax3,color='gray',shade=True,

                label=("Skewness : %.2f"%(df_protein[input3].skew()),

                       "Kurtosis: %.2f"%(df_protein[input3].kurtosis())))

    axes = [ax1,ax2,ax3]

    input = [input1,input2,input3]

    for j in range(len(axes)):

        axes[j].set_xlabel(input[j],color='black',fontsize=12)

        axes[j].set_title(input[j] + ' Kdeplot',fontsize=14)

        axes[j].axvline(df_protein[input[j]].mean() , color ='g',linestyle = '--')

        axes[j].legend(loc ='upper right',fontsize=12,ncol=2)

    sns.despine()

    return plt.show()



stat_kde_plot('resolution','residueCount','structureMolecularWeight')
for i in ['resolution','residueCount','structureMolecularWeight']:

    df_protein[i] = df_protein[i].map(lambda i: np.log(i) if i > 0 else 0)

stat_kde_plot('resolution','residueCount','structureMolecularWeight')
# Drop all null values from this columns

def stat_plot (input):

    (mu, sigma) = norm.fit(df_protein[input])

    f, (ax1, ax2)= plt.subplots(1,2,figsize=(15,5))

    # Apply the log transformation on the column

    sns.distplot(df_protein[input],ax = ax1,fit=norm,color ='blue',hist=False)

    ax1.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')

    ax1.set_ylabel('Frequency')

    ax1.set_title(input +' Distribution',color='black',fontsize=14)

    #Get also the QQ-plot

    res = stats.probplot(df_protein[input], plot=ax2)

    sns.despine()

    return plt.show()

stat_plot('structureMolecularWeight')

stat_plot('residueCount')

stat_plot('resolution')
def box_plot(input):

    g = sns.factorplot(x="classification", y = input,data = df_protein, kind="box",size =4,

                  aspect=2)

    plt.title(input, fontsize=14,color='black')

    return plt.show()



box_plot('residueCount')

box_plot('resolution')

box_plot('structureMolecularWeight')
#class_dict = {'RIBOSOME':1,'HYDROLASE':2,'TRANSFERASE':3} 

class_dict = {'HYDROLASE':1,'TRANSFERASE':2,'OXIDOREDUCTASE':3}

df_protein['class'] = df_protein.classification.map(class_dict)

#Reduce the dataset to only numerical column and clssification column

columns = ['resolution','structureMolecularWeight','densityMatthews','densityPercentSol',

           'residueCount','class']

df_ml = df_protein[columns]

df_ml.dropna(inplace=True)

df_ml.head()
colormap = plt.cm.RdBu

f, ax = plt.subplots(figsize=(18,7))

sns.heatmap(df_ml.corr(),cmap= colormap,annot=True,ax=ax,annot_kws ={'fontsize':12})

kwargs= {'length':3, 'width':1, 'colors':'black','labelsize':13}

ax.tick_params(**kwargs)

ax.tick_params(**kwargs,axis='x')

plt.title ('Pearson Correlation Matrix', color = 'black',fontsize=18)

plt.tight_layout()

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

X = df_ml.drop('class',axis = 1)

y = df_ml['class']
y=y.values

sc = StandardScaler()

x=sc.fit_transform(X)
from sklearn.metrics import classification_report,precision_score,recall_score,f1_score,roc_auc_score,accuracy_score
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

    

    from sklearn.metrics import matthews_corrcoef

    y_head=cross_val_predict(model,x,y,cv=10)

    print(matthews_corrcoef(y, y_head))

    print(isim," Algoritması başarım sonucu: ",np.mean(cross_val_score(model,x,y,cv=10)))

    cm = confusion_matrix(y,y_head)

    f, ax = plt.subplots(figsize =(5,5))

    sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

    plt.xlabel("Tahmin Edilen Değer")

    plt.ylabel("Gerçek Değer")

    baslik=isim+" Algoritması Karmaşıklık Matrisi"

    plt.title(baslik)

    plt.show()

    

    aucdegeri,fpr,tpr,trr=yenimetot(y,y_head)

    classid,tn,fp,fn,tp=perf_measure(y,y_head)

    auc_scor.append(aucdegeri)

    #fpr,tpr,trr=roc_curve(y,y_head)

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

    return aucdegeri,cm,fpr,tpr,trr
knn=KNeighborsClassifier(n_neighbors=11)

knnauc,cmknn,knn_fpr,knn_tpr,knn_trr=model_olustur2("KNN",knn,x,y)
dtc=DecisionTreeClassifier()

dtcauc,cmdtc,dtc_fpr,dtc_tpr,dtc_trr=model_olustur2("Karar Ağaçları",dtc,x,y)
rfc=RandomForestClassifier(n_estimators=225,random_state=1)

rfcauc,cmrfc,rfc_fpr,rfc_tpr,rfc_trr=model_olustur2("Rastgele Orman",rfc,x,y)
nb=GaussianNB()

nbauc,cmnb,nb_fpr,nb_tpr,nb_trr=model_olustur2("Naive Bayes",nb,x,y)
lrc=LogisticRegression()

lrcauc,cmlrc,lrc_fpr,lrc_tpr,lrc_trr=model_olustur2("Lojistik Regresyon",lrc,x,y)
svc=SVC(random_state=1)

svcauc,cmsvc,svc_fpr,svc_tpr,svc_trr=model_olustur2("Destek Vektör Makineleri",svc,x,y)
gfc=GradientBoostingClassifier(n_estimators= 1000, max_leaf_nodes= 4, max_depth=None,random_state= 2,min_samples_split= 5)

gfcauc,cmgfc,gfc_fpr,gfc_tpr,gfc_trr=model_olustur2("Gradien Boosting",gfc,x,y)
abc=AdaBoostClassifier(n_estimators=100, random_state=0)

abcauc,cmabc,abc_fpr,abc_tpr,abc_trr=model_olustur2("AdaBoosting",abc,x,y)
ysa=MLPClassifier(alpha=1, max_iter=1000)

ysaauc,cmysa,ysa_fpr,ysa_tpr,ysa_trr=model_olustur2("Yapay Sinir Ağları",ysa,x,y)
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

plt.title('Structural Protein Sequences Dataset ile Sınıflandırma',fontsize = 20,color='blue')

plt.grid()

plt.legend()

plt.show()
def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr,rfc_fpr,rfc_tpr,nb_fpr,nb_tpr,gbc_fpr,gbc_tpr,abc_fpr,abc_tpr,ysa_fpr,ysa_tpr):

    plt.figure(figsize=(16,8))

    plt.title('ROC Curve \n Top 9 Classifiers', fontsize=18)

    plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: %0.2f'% lrcauc)

    plt.plot(knear_fpr, knear_tpr, label='KNears Neighbors Classifier Score: %0.2f'% knnauc)

    plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: %0.2f'% svcauc)

    plt.plot(tree_fpr, tree_tpr, label='Decision Tree Classifier Score: %0.2f'% dtcauc)

    plt.plot(rfc_fpr, rfc_tpr, label='Random Forest Classifier Score: %0.2f'% rfcauc)

    plt.plot(nb_fpr, nb_tpr, label='Naive Bayes Classifier Score: %0.2f'% nbauc)

    plt.plot(gbc_fpr, gbc_tpr, label='Gradient Boosting Classifier Score: %0.2f'% gfcauc)

    plt.plot(abc_fpr, abc_tpr, label='AdaBoosting Classifier Score: %0.2f'% abcauc)

    plt.plot(ysa_fpr, ysa_tpr, label='Neural Network Score: %0.2f'% ysaauc)

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