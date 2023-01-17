

import numpy as np

import matplotlib.pyplot as plt

from sklearn import linear_model ,neighbors,preprocessing,svm,tree

from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import StandardScaler,LabelEncoder

import pandas as pd

from IPython.display import display

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

import warnings

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score,accuracy_score,make_scorer

import seaborn as sns

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier,VotingClassifier,ExtraTreesClassifier,BaggingClassifier

import sys

from xgboost import XGBRegressor





import vecstack

from vecstack import stacking

from xgboost import XGBClassifier

from scipy.stats import probplot

from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import r2_score,accuracy_score,make_scorer,log_loss,precision_score

from sklearn.model_selection import GridSearchCV,KFold,cross_val_score

from sklearn.svm import NuSVC,SVC

from scipy import std ,mean

from scipy.stats import norm

from scipy import stats

warnings.filterwarnings('ignore')







pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 100)





# plt.style.use('dark_background')



sns.set_palette('dark')





plt.rcParams['figure.figsize'] = [14, 7]

sns.set_context("paper", rc={"font.size":15,"axes.titlesize":22,"axes.labelsize":20,"figure.suptitlesize":25}) 





data=pd.read_csv('../input/heart-disease-uci/heart.csv')



scale=['age','chol','trestbps','thalach','oldpeak']

f=data[scale]

scaler=StandardScaler().fit(f.values)

f=scaler.transform(f.values)

data[scale]=f





plt.rcParams['figure.figsize'] = [14, 7]

sns.countplot(x="target", data=data);
data.target.value_counts().plot.pie(autopct='%1.1f%%',fontsize=25)
sns.barplot(data.sex,data.target)

plt.xlabel('Sex:  0:Female, 1:Male');
sns.heatmap(data.corr())

sns.boxplot(data.slope,data.oldpeak)
data.corr().target.sort_values(ascending=False)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_curve,auc



def model(algorithm,dtrain_x,dtrain_y,dtest_x,dtest_y,of_type):

    print('*'*120)

    print ("MODEL")

    algorithm.fit(dtrain_x,dtrain_y)

    predictions = algorithm.predict(dtest_x)

    

    print (algorithm)

    print ("\naccuracy_score :",accuracy_score(dtest_y,predictions))

    

    print ("\nclassification report :\n",(classification_report(dtest_y,predictions)))

        

    plt.figure(figsize=(15,12))

    plt.subplot(221)

    sns.heatmap(confusion_matrix(dtest_y,predictions),annot=True, annot_kws={"size": 23,'va':'top'})

    plt.title("CONFUSION MATRIX")

    

    predicting_probabilites = algorithm.predict_proba(dtest_x)[:,1]

    fpr,tpr,thresholds = roc_curve(dtest_y,predicting_probabilites)

    plt.subplot(222)

    plt.plot(fpr,tpr,label = ("Area_under the curve :",auc(fpr,tpr)),color = "r")

    plt.plot([1,0],[1,0],linestyle = "dashed",color ="k")

    plt.legend(loc = "best")

    plt.title("ROC - CURVE & AREA UNDER CURVE")

    

    if  of_type == "feat":

        

        dataframe = pd.DataFrame(algorithm.feature_importances_,dtrain_x.columns).reset_index()

        dataframe = dataframe.rename(columns={"index":"features",0:"coefficients"})

        dataframe = dataframe.sort_values(by="coefficients",ascending = False)

        plt.subplot(223)

        ax = sns.barplot(x = "coefficients" ,y ="features",data=dataframe,palette="husl")

        plt.title("FEATURE IMPORTANCES",fontsize =17)

        for i,j in enumerate(dataframe["coefficients"]):

            ax.text(.011,i,j,weight = "bold")

    

    elif of_type == "coef" :

        

        dataframe = pd.DataFrame(algorithm.coef_.ravel(),dtrain_x.columns).reset_index()

        dataframe = dataframe.rename(columns={"index":"features",0:"coefficients"})

        dataframe = dataframe.sort_values(by="coefficients",ascending = False)

        plt.subplot(223)

        ax = sns.barplot(x = "coefficients" ,y ="features",data=dataframe,palette="husl")

        plt.title("FEATURE IMPORTANCES",fontsize =20)

        for i,j in enumerate(dataframe["coefficients"]):

            ax.text(.011,i,j,weight = "bold")

            

    elif of_type == "none" :

        return (algorithm)

x=data.drop('target',1)

y=data['target']

x,y=shuffle(x,y)

xtr,xtest,ytr,ytest=train_test_split(x,y, test_size=0.2)



model(LogisticRegression(),xtr,ytr,xtest,ytest,'coef')



model(XGBClassifier(),xtr,ytr,xtest,ytest,'feat')

from sklearn.tree import DecisionTreeClassifier

dec_tree=DecisionTreeClassifier()

cross_val_score(dec_tree,x,y, cv=4)
fig,axes=plt.subplots(1,3,figsize=(20,11))



sns.barplot(data.slope,data.target,ax=axes[0])

sns.boxplot(data.target,data.oldpeak,ax=axes[1])

sns.boxplot(data.target,data.chol,ax=axes[2])
fig,axes=plt.subplots(1,2,figsize=(15,8))

sns.boxplot(data.target,data.trestbps,ax=axes[0])

sns.boxplot(data.target,data.thalach,ax=axes[1])
fig,axes=plt.subplots(1,4,figsize=(23,12))



sns.barplot(data.fbs,data.target,ax=axes[0])

sns.barplot(data.thal,data.target,ax=axes[1])

sns.barplot(data.cp,data.target,ax=axes[2])

sns.barplot(data.ca,data.target,ax=axes[3])
data.drop(data[(data.target==1) & (data.oldpeak>1)].index,axis=0,inplace=True)

data.drop(data[(data.target==1) & (data.chol>2)].index,axis=0,inplace=True)

data.drop(data[(data.target==1) & (data.thalach<-1.5)].index,axis=0,inplace=True)
data=pd.get_dummies(data, columns=['cp','restecg','slope','thal'],drop_first=True)
x=data.drop('target',1)

y=data.target

x,y=shuffle(x,y)

xtr,xtest,ytr,ytest=train_test_split(x,y, test_size=0.2)

model(LogisticRegression(),xtr,ytr,xtest,ytest,'coef')


rf=RandomForestClassifier()

reg=LogisticRegression()

k=KNeighborsClassifier()

xg=XGBClassifier(max_depth=3,learning_rate=0.1)

svc=SVC()

model=[rf,reg,k,xg,dec_tree,svc]

voting=VotingClassifier([('rf',model[0]),('reg',model[1]),('k',model[2]),('xg',model[3]),('dec_tree',model[4]),('svc',model[5])],weights=[1,2,1,1,1,1])







Str,Ste=stacking(model,xtr,ytr,xtest,regression=False,metric=accuracy_score,shuffle=True,verbose=0,stratified=True)

print(

    

     'LogisticRegression:',cross_val_score(LogisticRegression(),x,y,cv=5).mean(),

     

     '\n DecisionTreeClassifier:' ,cross_val_score(DecisionTreeClassifier(),x,y, cv=5).mean(),

      

      

      '\n XGB:',cross_val_score(xg,x,y, cv=5).mean(),

      

      

      '\n KNeighbors:',cross_val_score(KNeighborsClassifier(n_neighbors=6),x,y, cv=5).mean(),

      

      

      

      '\n RandomForest:',cross_val_score(RandomForestClassifier(),x,y, cv=5).mean(),

      

      

      

      '\n SVC:',cross_val_score(SVC(),x,y, cv=5).mean(),

      

      

      '\n Voting:',cross_val_score(voting,x,y, cv=5).mean(),

      

      

      

      '\n Stacked w SVC ',cross_val_score(SVC(),Str,ytr,cv=4).mean(),

      

      

      

       '\n Stacked w Logistic:', cross_val_score(LogisticRegression(),Str,ytr,cv=4).mean(),

      

      

      

      '\n Stacked w voting ',cross_val_score(voting,Str,ytr,cv=4).mean()

      

    

    )

      

      

    

li=[]

for i in range(1,25):

    k = KNeighborsClassifier(n_neighbors = i) 

    li.append(cross_val_score(k,x,y,cv=5).mean())

    

plt.plot(range(1,25),li)



plt.xlabel("K")



plt.ylabel("acc")



acc=max(li)*100

best_k=li.index(max(li))+1

print(f"Best acc :{acc}",best_k)