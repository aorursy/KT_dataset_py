# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# -*- coding: utf-8 -*-
import pandas as pd
import numpy  as np 
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter





#data convert to object
data=pd.read_csv('../input/asdfvef/bank-full.csv',sep=';')
head=data.head()
print(head)
print("************************")
columns=data.columns
print(columns)
print("************************")
dtypes=data.dtypes
print(dtypes)
print("************************")
bosverivarmı=data.columns[data.isnull().any()]
print(bosverivarmı)
#the data not contains null
print("************************")
print(data.info())


#firstly we have to make the data numerical
#marital
evlilik=data["marital"].value_counts(dropna=False)
print(evlilik)
data["marital"]=[1 if i == 'single' else 0 if  i=='married' else 2 for i in data["marital"]]

'''
married = 0
single = 1
else = 2
'''
#get dummiesleri görselleştirmeden sonra predict algoritmasından önce yazalım
#data=pd.get_dummies(data,columns=['marital'])

#job
jobsayısı=data["job"].value_counts(dropna=False)
print(jobsayısı)
data['job'].value_counts()
data['job']=data['job'].map({'blue-collar':1,'management':2,'technician':3,'admin.':4,'services':5,'retired':6,'self-employed':7,'entrepreneur':8,'unemployed':9,'housemaid':10,'student':11,'unknown':np.nan})

'''
'blue-collar':1
'management':2
'technician':3
'admin.':4
'services':5
'retired':6
'self-employed':7
'entrepreneur':8
'unemployed':9
'housemaid':10
'student':11
'unknown':12
'''

#days
print("*********************groupby1***********************")
groupby1=data.groupby(['day','y']).size().reset_index().groupby('day')[[0]].max()
print(groupby1)

#camping
numbercamping=data["campaign"].value_counts(dropna=False)

#eğitim
eğitimsayısı=data["education"].value_counts(dropna= False)
print(eğitimsayısı)
data['education']=data['education'].map({'unknown':np.nan,'primary':1,'secondary':2,'tertiary':3})
#unknown kolonunu ortalamayı bozmadan doldurmaya çalışalım

#defaut : temerrüte kalmış borcu var mı
defaultsayısı=data['default'].value_counts(dropna=False)
print(defaultsayısı)

def defaultbelirleme(value):
    if value=='no':
        return 0
    else:
        return 1   
data['default']=data['default'].apply(defaultbelirleme)


#housing borçla ev var mı?
housing=data['housing'].value_counts(dropna=False)
print(housing)
#data['housing'] = data['housing'].map({})
def defaultbelirleme(value):
    if value=='no':
        return 0
    else:
        return 1
data['housing']=data['housing'].apply(defaultbelirleme)

print("*********************groupby2***********************")
groupby2=data.groupby(['housing','age']).size().reset_index().groupby('age')[[0]].max()
print(groupby2)
print(groupby2.sum())
'''0= no 
   1= yes
'''
plt.style.use("seaborn-whitegrid")
g=sns.FacetGrid(data,col='housing', height=4, aspect=2)
g.map(sns.distplot, 'age',bins=25)
plt.show()
#60 tan sonra ev borucu olan yok.ev borcu olanlar 25 -50 arası çoğunluk

#loan 
data['loan']=data['loan'].replace({'no':0, 'yes':1})  #replace and map same methods
#borç varmı ?

#mounth
aylar=data['month'].value_counts()
print(aylar)
data['month']=data['month'].map({'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12})

#contact 
contact1=data['contact'].value_counts()
data['contact']=data['contact'].map({'cellular':0,'telephone':1,'unknown':np.nan})

#balance
#öncelikle datamızı normalize edelim üstüne outlier hesaplayalım ve gerekirse çıkaralım
print(data['balance'].min())
print(data['balance'].max())
#formülümüz =Xsc=(X−Xmin)/(Xmax−Xmin)

from sklearn.preprocessing import MinMaxScaler
scaling=MinMaxScaler()
data['balance']=scaling.fit_transform(data[['balance']])
#duration
#his attribute highly affects the output target (e.g., if duration=0 then y='no')

print("******************************")
print(data.info())

#preoutcome :önceki pazarlama durumu
pout=data['poutcome'].unique()
print(pout)
data['poutcome']=data['poutcome'].map({'failure':0,'success':1,'other':2,'unknown':np.nan})

#durationu normalize edelim
# formülümüz:Xsc=(X−Xmin)/(Xmax−Xmin)
data['duration']=data['duration'].apply(lambda v :((v-data['duration'].min()) /(data['duration'].max()-data['duration'].min())))
print("*********************groupby3***********************")
groupby3=data.groupby(['month','duration']).size().reset_index().groupby('month')[[0]].mean()
print(groupby3)
#5-6-7-8 month have most duration
#y encoder
data['y']=data['y'].replace({'no':0, 'yes':1})
print("y sayıları")
numbery=data['y'].value_counts()
print(numbery)


#değiştirdiğimiz datayı farklı bir csv dosyasına aldık.
data.to_csv("bank.preprocessed.csv",index=False)
new_df=pd.read_csv("bank.preprocessed.csv")
print(new_df.head())

describe=new_df.describe()
print(describe)

'''***************** DATA VİSUALİZATİON *************'''


plt.style.use('classic')
plt.subplots(figsize=(15,12))
sns.heatmap(new_df.corr(),annot=True,fmt='.2f',color='red',cmap='magma')
plt.show()
#çıkarımlar
#1-)duration süresi yüzde 40 corr gerekirse yeni featurelar üretilebilir
#2-)housing yüzde -14
#3-)poutcome yüzde +14
#4-)pdays yüzde 10 pday müşteri ile iletişime geçilen günden geçen zaman 999=daha önce ulaşılamadı anlamına gelir
#5-)hpusing ile month arasında yüzde 17
corry=new_df.corr()['y'].sort_values(ascending=False)
print(corry)
plt.show()

#outlier için;
plt.style.use("seaborn-whitegrid")
new_df.plot.box(figsize=(12,10))
plt.xticks(list(range(len(new_df.columns))), new_df.columns,rotation='vertical')
plt.show()




#age
plt.subplots(figsize=(12,10))
plt.hist(new_df['age'], bins=50,color='blue')
plt.show()
def bar_graph(variable):
    var=new_df[variable]
    varValue=var.value_counts()
    plt.style.use("seaborn-whitegrid")
    plt.figure(figsize=(9,3))
    plt.bar(varValue.index, varValue)
    plt.xticks(varValue.index, varValue.values)
    plt.ylabel('frequence')
    plt.title(variable)
    plt.show()
    print("{}: \n {}".format(variable, varValue))

category1=["marital","education","default"]
for each in category1:
    bar_graph(each)
#age and y
plt.style.use("seaborn-whitegrid")
g=sns.FacetGrid(new_df,col='y')
g.map(sns.distplot, 'age',bins=25)
plt.show()


#housing age an y
g=sns.FacetGrid(new_df,col='y',row='housing')
g.map(plt.hist,'age',bins=25)
g.add_legend()
plt.show()
sns.countplot(x='housing', data=new_df)
plt.show()
print("*********************housing and y = 1 *************************")
numberhousi=((new_df['housing'] == 1) & (new_df['y'] == 1)).value_counts()
print(numberhousi)
#ikisininde bir olduğu sadece 1935 müşteri var bu bilgi ile kolon inşa etmeliyiz.



#duration
g=sns.FacetGrid(new_df,col='y',size=4, aspect=2)
g.map(sns.distplot, 'duration',bins=50)
plt.show()
#0.1 ve aşağısı çok yüksek ihtimal 0
#0.2 den yukarısı 1 e çok yakın
sns.jointplot(new_df.loc[:,'duration'], new_df.loc[:,'housing'], kind="regg", color="#ce1414")
plt.show()


#poutcome
sns.countplot(x='poutcome', data=new_df)
plt.show()
print("*********************poutcome and y = 1 *************************")
numberpout=((new_df['poutcome'] == 1) & (new_df['y'] == 1)).value_counts()
print(numberpout)
#True       978 bununla ilgili de feature engineering yap
#failure':0,'success':1,'other':2



#pdays
new_df['pdays']=new_df['pdays'].replace(-1,999)
#999 means client was not previously contacted)
plt.subplots(figsize=(12,10))
plt.hist(new_df['pdays'], bins=50,color='blue')
plt.xlabel("How many days ago was the connect passed?")
plt.show()


'''outlier tespit etmek bulmak ve doldurmak'''
def detect_outliers(df,features):
    outlier_indices = []
    
    for c in features:
        #1st quartile
        Q1=np.percentile(df[c],25)
        #3st quartile
        Q3=np.percentile(df[c],75)
        #IQR 
        IQR=Q3-Q1
        #outlier step
        outlier_step=IQR*1.5
        #detect outlier
        outlier_list_col=df[(df[c] < Q1-outlier_step) | (df[c] > Q3 + outlier_step)].index
        #store indeces
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list(i for i , v in outlier_indices.items() if v > 2)
        
    return multiple_outliers

#veriseti dengesiz olduğu için outliersları çıkarmanın mantıklı olmadığını düşünüyorum.


'''               *********FEATURE ENGİNEERİNG**************           '''
#most correlation with y.
#we create new two column
print("***********number of y************")
print(new_df['y'].value_counts())

new_df['low_duration']=[0 if i<0.08 else 1 for i in new_df['duration']]
print("***********number of low y************")
print(new_df['low_duration'].value_counts())


new_df['high_duration']=[1 if i>0.22 else 0 for i in new_df['duration']]
print("***********number of high y************")
print(new_df['high_duration'].value_counts())

#doğru kullandığımdan emin değilim buraya previousu da ekle
#pdays and previous
new_df['accessible customerspart1']=[1 if i<200 else 0 for i in new_df['pdays']]
new_df['accessible customerspart2']=[1 if i>=2 else 0 for i in new_df['previous']]
new_df['accessible customers']=new_df['accessible customerspart1']+new_df['accessible customerspart2']
#we use only accessible customers columns because it contains other parts so:
new_df.drop(columns=['accessible customerspart1','accessible customerspart2'],axis=1,inplace=True)

#day  7 20 arasını bir yerde öteki değerleri bir yerde ayrıştıralım
new_df['d-day']=[1 if (i>11 & i<22) else 0 for i in new_df['day']]
#housing 
#60 tan sonra ev borucu olan yok.ev borcu olanlar 25 -50 arası çoğunluk
new_df['d-age']=[0 if (i<22 | i>60) else 1 if (i>=25 & i<=50) else 2 for i in new_df['age']]
new_df['d-housing1']=new_df['d-age']+new_df['housing']
new_df['d-housing']=[1 if i==2 else 0 for i in new_df['d-housing1']]


#new features corr with y
only_NewFeatures=new_df.drop(columns=['age','job','marital','education','default','day','contact','d-housing1'],axis=1)
plt.subplots(figsize=(15,12))
sns.heatmap(only_NewFeatures.corr(),annot=True,fmt='.2f',color='red',cmap='coolwarm')
plt.show()

new_df.drop(columns=['day','d-age'],axis=1,inplace=True)
#GET DUMMİES
new_df=pd.get_dummies(new_df,columns=['job','marital','education','housing','month','default','contact','loan','poutcome','accessible customers'])

'''                         **********BUİLD MODELS**********             '''
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix


X=new_df.drop(columns=['y'],axis=1)
y=new_df.iloc[:,6]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=35)
print("x_train: " , len(X_train))
print("X_test: " , len(X_test))
print("y_train: " , len(X_train))
print("y_test: " , len(y_test))


#SİMPLE LOGİSTİC REGRESSİON
print("****************   Tahminler      *****************")

print("**********************************************************")
print("LogisticRegression")
logreg=LogisticRegression(C=0.01)
logreg.fit(X_train,y_train)
acc_log_train=logreg.score(X_train,y_train)*100
acc_log_test=logreg.score(X_test,y_test)*100
print('Training accuracy : % {}'.format(acc_log_train))
print('Test accuracy : % {}'.format(acc_log_test))
print('Logistic Regression confusion matrix')
y_pred1=logreg.predict(X_test)
cm=confusion_matrix(y_test,y_pred1)
print(cm)
skor1=metrics.accuracy_score(y_pred1,y_test)
print("skor1:",skor1)


print("**********************************************************")
print("RandomForestClassifier")
rdc=RandomForestClassifier(max_features=15,n_estimators=100,criterion='gini',random_state=42)
rdc.fit(X_train,y_train)
acc_rdc_train=rdc.score(X_train,y_train)*100
acc_rdc_test=rdc.score(X_test,y_test)*100
print('Training accuracy : % {}'.format(acc_rdc_train))
print('Test accuracy : % {}'.format(acc_rdc_test))
print('RandomForestClassifier confusion matrix')
y_pred2=rdc.predict(X_test)
cm=confusion_matrix(y_test,y_pred2)
print(cm)
skor2=metrics.accuracy_score(y_pred2,y_test)
print("skor2:",skor2)

print("**********************************************************")
print("DecisionTreeClassifier")
dc=DecisionTreeClassifier(max_depth =20,min_samples_split=400,criterion="gini")
dc.fit(X_train,y_train)
acc_dc_train=dc.score(X_train,y_train)*100
acc_dc_test=dc.score(X_test,y_test)*100
print('Training accuracy : % {}'.format(acc_dc_train))
print('Test accuracy : % {}'.format(acc_dc_test))
print('DecisionTreeClassifier confusion matrix')
y_pred3=dc.predict(X_test)
cm=confusion_matrix(y_test,y_pred3)
print(cm)
skor3=metrics.accuracy_score(y_pred3,y_test)
print("skor3:",skor3)

print("**********************************************************")
print("KNeighborsClassifier")
knn=KNeighborsClassifier(metric='manhattan', n_neighbors=19, weights='uniform')
knn.fit(X_train,y_train)
acc_knn_train=knn.score(X_train,y_train)*100
acc_knn_test=knn.score(X_test,y_test)*100
print('Training accuracy : % {}'.format(acc_knn_train))
print('Test accuracy : % {}'.format(acc_knn_test))
print('KNeighborsClassifier confusion matrix')
y_pred4=knn.predict(X_test)
cm=confusion_matrix(y_test,y_pred4)
print(cm)
skor4=metrics.accuracy_score(y_pred4,y_test)
print("skor4:",skor4)


print("**********************************************************")
print("SVC")
svc=SVC()
svc.fit(X_train,y_train)
acc_svc_train=svc.score(X_train,y_train)*100
acc_svc_test=svc.score(X_test,y_test)*100
print('Training accuracy : % {}'.format(acc_svc_train))
print('Test accuracy : % {}'.format(acc_svc_test))
print('SVC confusion matrix')
y_pred5=svc.predict(X_test)
cm=confusion_matrix(y_test,y_pred5)
print(cm)
skor5=metrics.accuracy_score(y_pred5,y_test)
print("skor5:",skor5)


print("**********************************************************")
print("GaussianNB")
gaussian=GaussianNB()
gaussian.fit(X_train,y_train)
acc_gaussian_train=gaussian.score(X_train,y_train)*100
acc_gaussian_test=gaussian.score(X_test,y_test)*100
print('Training accuracy : % {}'.format(acc_gaussian_train))
print('Test accuracy : % {}'.format(acc_gaussian_test))
print('SVC confusion matrix')
y_pred6=gaussian.predict(X_test)
cm=confusion_matrix(y_test,y_pred6)
print(cm)
skor6=metrics.accuracy_score(y_pred6,y_test)
print("skor6:",skor6)

print("**********************************************************")

'''
print("************HYPERPARAMETER TURİNG-GRİD SEARCH-CROSS VALİDATİON******************")
print("SVC")
logreg_param_grid={"C":np.logspace(-3,3,7),
                   "penalty":["l1","l2"]}
gs=GridSearchCV(estimator=logreg,
                param_grid=logreg_param_grid,
                scoring='accuracy',
                cv =10,
                n_jobs=-1)
grid_search=gs.fit(X_train,y_train)
eniyisonuc=grid_search.best_score_
eniyiparametreler=grid_search.best_params_
print(eniyisonuc)
print(eniyiparametreler)

print("RANDOMFORESTCLASSİFİER")
rfc_param_grid={"max_features": [1,3,10],
                "min_samples_split":[1,3,10],
                "min_samples_leaf":[1,3,10],
                "bootstrap":[False],
                "n_estimators":[100,300],
                "criterion":["gini"]}
gs=GridSearchCV(estimator=rdc,
                param_grid=rfc_param_grid,
                scoring='accuracy',
                cv =10,
                n_jobs=-1)
grid_search=gs.fit(X_train,y_train)
eniyisonuc=grid_search.best_score_
eniyiparametreler=grid_search.best_params_
print(eniyisonuc)
print(eniyiparametreler)

print("DECİSİONTREE")
dt_param_grid ={"min_samples_split" : range(10,500,20),
                "max_depth": range(1,20,2)}
gs=GridSearchCV(estimator=dc,
                param_grid=dt_param_grid,
                scoring='accuracy',
                cv =10,
                n_jobs=-1)
grid_search=gs.fit(X_train,y_train)
eniyisonuc=grid_search.best_score_
eniyiparametreler=grid_search.best_params_
print(eniyisonuc)
print(eniyiparametreler)

print("KNeighborsClassifier")
knn_param_grid={"n_neighbors": np.linspace(1,19,10, dtype =int).tolist(),
                "weights":["uniform","distance"],
                "metric":["euclidean","manhattan"]}
gs=GridSearchCV(estimator=knn,
                param_grid=knn_param_grid,
                scoring='accuracy',
                cv =10,
                n_jobs=-1)
grid_search=gs.fit(X_train,y_train)
eniyisonuc=grid_search.best_score_
eniyiparametreler=grid_search.best_params_
print(eniyisonuc)
print(eniyiparametreler)

print("SUPPORtVECTORMACHİNE")
svc_param_grid={"kernel" : ["rbf"],
                "gamma":[0.001,0.01,0.1,1],
                "C":[1,10,50,100,200,300,100]}
gs=GridSearchCV(estimator=svc,
                param_grid=svc_param_grid,
                scoring='accuracy',
                cv =10,
                n_jobs=-1)
grid_search=gs.fit(X_train,y_train)
eniyisonuc=grid_search.best_score_
eniyiparametreler=grid_search.best_params_
print(eniyisonuc)
print(eniyiparametreler)
'''


'''ENSEMBLE'''
votingC = VotingClassifier(estimators=[("rdc",rdc),
                                       ("dc",dc),
                                       ("knn",knn)],
                                        voting= 'hard', n_jobs = -1)
'''when voting=soft ==  0.9027691763248695'''
votingC=votingC.fit(X_train,y_train)
ensemble=accuracy_score(votingC.predict(X_test),y_test)
print("Ensemble uygulanmış yüzdelik :",accuracy_score(votingC.predict(X_test),y_test))


import pandas as pd 
  
# intialise data of lists. 
lastdata = {'Name':['RFC', 'DTC', 'KNN', 'Ensemble'], 'skorlar':[skor2, skor3,skor4,ensemble]} 
  
# Create DataFrame 
dflast = pd.DataFrame(lastdata) 


g = sns.barplot(x='Name',y='skorlar',data=dflast)
g.set_xlabel("Models")
g.set_xlabel("Models numbers")
g.set_title("models graph")
plt.show()