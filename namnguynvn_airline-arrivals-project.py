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
# Libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import iplot
from mpl_toolkits.basemap import Basemap

import scipy
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn import metrics
from IPython.display import Image
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
# Input Dataset
df=pd.read_csv("../input/delayed/DelayedFlights.csv")
print("Count Columns  :",df.shape[0])
print("Count Rows     :",df.shape[1])
df.head(5)
df["UniqueCarrier"].value_counts()
df=df.drop(columns=["Unnamed: 0","Year"])

df.head(5)
# print feature 
print(df.columns)
print(df.info())
df=df.drop(columns=["DepTime","ArrTime","ActualElapsedTime","CRSArrTime","CRSDepTime","CarrierDelay","WeatherDelay","NASDelay","SecurityDelay","LateAircraftDelay"])
df["Cancelled"].value_counts()
print(df[df["Cancelled"]==1].head(5))
print(df[df["Diverted"]==1].head(5))
df=df.drop(columns=["Cancelled","Diverted","CancellationCode","DepDelay"])
df.head(5)
number_feature = df.dtypes[df.dtypes != "object"].index
print(number_feature)
# Category
category_feature = df.dtypes[df.dtypes == "object"].index
print(category_feature)
df.describe(percentiles=[0.01, 0.25,0.75, 0.99])
list_feature_number=["ArrDelay","FlightNum","Distance","AirTime","TaxiIn","TaxiOut"]

fig,ax = plt.subplots(3,2, figsize=(12,12)) 
i=0 
for x in range(3):
    for y in range(2):
        sns.distplot(df[list_feature_number[i]], ax = ax[x,y])
        i+=1
plt.tight_layout()
plt.show()
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
df2=df
count_outlier=[]
for i in range(len(IQR)):
    index = df2[(df2[IQR.index[i]] < (Q1[i]-1.5*IQR[i])) | (df2[IQR.index[i]] >(Q3[i]+1.5*IQR[i]))].index
    count_outlier.append(len(index))
percent=[]
for count in count_outlier:
    percent.append(100*(count/(df2.shape[0])))
out_lier=pd.DataFrame({"count outlier":count_outlier,"percent":percent},
                      index=IQR.index)
print(out_lier)
for i in range(len(IQR)):
    index = list(df2[(df2[IQR.index[i]] >= (Q3[i]+1.5*IQR[i]))|(df2[IQR.index[i]] <=(Q1[i]-1.5*IQR[i]))].index)
    df2.drop(index, inplace=True)
df2.shape
df2.head(5)
number_feature = df2.dtypes[df2.dtypes != "object"].index
print(number_feature)
#sns.pairplot(df2, x_vars=["Distance"],y_vars=["ArrDelay"],height=8, aspect=.8, kind="reg")
category_feature = df2.dtypes[df2.dtypes == "object"].index
print(category_feature)
df2["UniqueCarrier"].value_counts()

sns.catplot(x='UniqueCarrier', y='ArrDelay', data=df2,height=12,kind='box')
plt.plot([0, 20], [30, 30], linewidth=2)
# convert month to category
df2["season"]=df2["Month"]
# spring
df2["season"]=df2["season"].replace(1,"spring")
df2["season"]=df2["season"].replace(2,"spring")
df2["season"]=df2["season"].replace(3,"spring")
# summer
df2["season"]=df2["season"].replace(4,"summer")
df2["season"]=df2["season"].replace(5,"summer")
df2["season"]=df2["season"].replace(6,"summer")
# autumn
df2["season"]=df2["season"].replace(7,"autumn")
df2["season"]=df2["season"].replace(8,"autumn")
df2["season"]=df2["season"].replace(9,"autumn")
# winter
df2["season"]=df2["season"].replace(10,"winter")
df2["season"]=df2["season"].replace(11,"winter")
df2["season"]=df2["season"].replace(12,"winter")

df2["season"].value_counts()

sns.catplot(x='season', y='ArrDelay', data=df2,height=12,kind='box')
plt.plot([0, 10], [30,30], linewidth=2)
# convert DayOfMonth to category
df2["week"]=df2["DayofMonth"]
# week1
df2["week"]=df2["week"].replace(1,"week1")
df2["week"]=df2["week"].replace(2,"week1")
df2["week"]=df2["week"].replace(3,"week1")
df2["week"]=df2["week"].replace(4,"week1")
df2["week"]=df2["week"].replace(5,"week1")
df2["week"]=df2["week"].replace(6,"week1")
df2["week"]=df2["week"].replace(7,"week1")
# week2
df2["week"]=df2["week"].replace(8,"week2")
df2["week"]=df2["week"].replace(9,"week2")
df2["week"]=df2["week"].replace(10,"week2")
df2["week"]=df2["week"].replace(11,"week2")
df2["week"]=df2["week"].replace(12,"week2")
df2["week"]=df2["week"].replace(13,"week2")
df2["week"]=df2["week"].replace(14,"week2")
# week3
df2["week"]=df2["week"].replace(15,"week3")
df2["week"]=df2["week"].replace(16,"week3")
df2["week"]=df2["week"].replace(17,"week3")
df2["week"]=df2["week"].replace(18,"week3")
df2["week"]=df2["week"].replace(19,"week3")
df2["week"]=df2["week"].replace(20,"week3")
df2["week"]=df2["week"].replace(21,"week3")
# week4
df2["week"]=df2["week"].replace(22,"week4")
df2["week"]=df2["week"].replace(23,"week4")
df2["week"]=df2["week"].replace(24,"week4")
df2["week"]=df2["week"].replace(25,"week4")
df2["week"]=df2["week"].replace(26,"week4")
df2["week"]=df2["week"].replace(27,"week4")
df2["week"]=df2["week"].replace(28,"week4")
df2["week"]=df2["week"].replace(29,"week4")
df2["week"]=df2["week"].replace(30,"week4")
df2["week"]=df2["week"].replace(31,"week4")
df2["week"].value_counts()
sns.catplot(x='week', y='ArrDelay', data=df2,height=12,kind='box')
plt.plot([0, 10], [30,30], linewidth=2)
df2.DayOfWeek.value_counts()
# convert DayOfWeek to category
df2["day"]=df2["DayOfWeek"]
# week1
df2["day"]=df2["day"].replace(1,"sunday")
df2["day"]=df2["day"].replace(2,"monday")
df2["day"]=df2["day"].replace(3,"tuesday")
df2["day"]=df2["day"].replace(4,"wednesday")
df2["day"]=df2["day"].replace(5,"thursday")
df2["day"]=df2["day"].replace(6,"friday")
df2["day"]=df2["day"].replace(7,"saturday")
df2["day"].value_counts()
sns.catplot(x='day', y='ArrDelay', data=df2,height=12,kind='box')
plt.plot([0, 10], [30,30], linewidth=2)
df2=df2.drop(columns=["Month",'DayofMonth', 'DayOfWeek'])
df2.head(5)
df2["Origin"].value_counts()
Origin_counts=df2["Origin"].value_counts()<30
Origin_index=Origin_counts[Origin_counts==True].index
print(len(Origin_index))
print(Origin_index)
for index in Origin_index:
    df2["Origin"]=df2["Origin"].replace(index,"OT")
df2["Origin"].value_counts()
df2[df2.Origin=="OT"].head(5)
df2["Dest"].value_counts()
Dest_counts=df2["Dest"].value_counts()<30
Dest_index=Dest_counts[Dest_counts==True].index
print(len(Dest_index))
print(Dest_index)
for index in Dest_index:
    df2["Dest"]=df2["Dest"].replace(index,"OT")
df2["Dest"].value_counts()
#index=df2.index
#df2["tail"]=df2["TailNum"]
#for index in index:
#    df2["tail"]=df2["tail"].replace(df2.TailNum[index],df2.TailNum[index][4:6])
#df2["tail"].value_counts()
df2.head()
df2.ArrDelay.isnull().sum()
df3=df2[df.ArrDelay.isnull()==False]
df3.shape
df3.isnull().sum()
row_missing=df3[df3.TailNum.isnull()==True].index
row_missing=list(row_missing)
df3=df3.drop(row_missing)
df3.shape
df3.isnull().sum().sum()
df3.head(5)
df3=df3.drop(columns=["TailNum"])

df3_corr=df3.corr()
mask = np.zeros_like(df3_corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f , ax = plt.subplots(figsize=(20,12))
    ax = sns.heatmap(df3_corr, mask=mask, vmax=1,annot=True, square=True)
df3.head()
UniqueCarrier_set=set(df3["UniqueCarrier"])
print("Number of unique of UniqueCarrier is :",len(UniqueCarrier_set))
sample=[]
for uni in UniqueCarrier_set:
    po=list(df3["ArrDelay"][df3[df3["UniqueCarrier"]==uni].index])
    sample.append(po)
# có n sample dùng anova cho từng cặp 0 và 1 ,1 và 2 xem thử có 1 cặp khác hay là tất cả đều same nhau
# nếu có 1 cặp khác 
khac=0
for i in range(len(sample)-1):
    f,p=stats. f_oneway(sample[i],sample[i+1])
    if p <0.05:
        khac=1
        break
print("have reject H0 : ",khac)
season_set=set(df3["season"])
print("Number of unique of season is :",len(season_set))
sample=[]
for uni in season_set:
    po=list(df3["ArrDelay"][df3[df3["season"]==uni].index])
    sample.append(po)
# có n sample dùng anova cho từng cặp 0 và 1 ,1 và 2 xem thử có 1 cặp khác hay là tất cả đều same nhau
# nếu có 1 cặp khác 
khac=0
for i in range(len(sample)-1):
    f,p=stats. f_oneway(sample[i],sample[i+1])
    if p <0.05:
        khac=1
        break
print("have reject H0 : ",khac)
Origin_set=set(df3["Origin"])
print("Number of unique of Origin is :",len(Origin_set))
sample=[]
for uni in Origin_set:
    po=list(df3["ArrDelay"][df3[df3["Origin"]==uni].index])
    sample.append(po)
# có n sample dùng anova cho từng cặp 0 và 1 ,1 và 2 xem thử có 1 cặp khác hay là tất cả đều same nhau
# nếu có 1 cặp khác 
khac=0
for i in range(len(sample)-1):
    f,p=stats. f_oneway(sample[i],sample[i+1])
    if p <0.05:
        khac=1
        break
print("have reject H0 : ",khac)
Dest_set=set(df3["Dest"])
print("Number of unique of Dest is :",len(Dest_set))
sample=[]
for uni in Dest_set:
    po=list(df3["ArrDelay"][df3[df3["Dest"]==uni].index])
    sample.append(po)
# có n sample dùng anova cho từng cặp 0 và 1 ,1 và 2 xem thử có 1 cặp khác hay là tất cả đều same nhau
# nếu có 1 cặp khác 
khac=0
for i in range(len(sample)-1):
    f,p=stats. f_oneway(sample[i],sample[i+1])
    if p <0.05:
        khac=1
        break
print("have reject H0 : ",khac)
Y=df3.ArrDelay.copy()
X=df3.drop(columns=["ArrDelay"])
Y_cate=Y.copy()
Y_cate[Y_cate >=30 ] = 0
Y_cate[Y_cate !=0] = 1
Y_cate.value_counts()
X.head(5)
X_number=pd.get_dummies(X)
X_number.head(5)
X_number.shape
X.head(5)
X_logis=X.drop(columns=["Origin","Dest"])
X_logis_number=pd.get_dummies(X_logis)
X_logis_number.head(5)
scaler = StandardScaler()
X_logis_number = scaler.fit_transform(X_logis_number)
print(X_logis_number)

X_logis_train,X_logis_test,Y_logis_train,Y_logis_test=train_test_split(X_logis_number,Y_cate,test_size=0.3,random_state=0)
# Training
Logistic= LogisticRegression()
Logistic.fit(X_logis_train,Y_logis_train)
# Predict
Y_logis_pred=Logistic.predict(X_logis_test)
# Performance
Logistic_report=metrics.classification_report(Y_logis_test,Y_logis_pred)
print(Logistic_report)
# LogisticRegression tuning
# penalty = [‘l1’, ‘l2’, ‘elasticnet’, ‘none’] # default=’l2’
# dual =[False] # because N sample > N feature
# tol =[ 0.01 ,0.001,0.0001] # Tolerance for stopping criteria default =e^-4
# C=[0.1, 1 ,10] #
# intercept_scaling=[1]
parameters = {'tol':[0.01, 0.001],'C':[0.1,5,10]}
clf_logistics=GridSearchCV(LogisticRegression(),parameters)
clf_logistics.fit(X_logis_train,Y_logis_train)
df_clf=pd.DataFrame(clf_logistics.cv_results_)
print(df_clf)
df_clf[["param_tol","param_C","mean_test_score"]]
dir(clf_logistics)
clf_logistics.best_score_
clf_logistics.best_params_
clf_logistics.refit_time_
logistic=LogisticRegression(penalty='l2', dual=False, tol=0.01, 
                           C=5, fit_intercept=True, intercept_scaling=1.0, 
                           class_weight=None, random_state=None)
t_LG_train_start=time.time()
DecisionTree_tune.fit(X_logis_train,Y_logis_train)
t_LG_train=time.time()-t_LG_train_start
t_LG_predict_start=time.time()
Y_LG_pred=DecisionTree_tune.predict(X_logis_test)
t_LG_predict=time.time()-t_LG_predict_start
LG_report=metrics.classification_report(Y_logis_test,Y_logis_pred)
print(LG_report)
kfold=KFold(n_splits=10, shuffle=False, random_state=None)
scores_logistic= cross_val_score(logistic, X_logis_number,Y_cate,cv=kfold)
print( scores_logistic)
scores_logistic.max()
print(X_logis_number.shape)
cov_mat = np.cov(X_logis_number.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
tot = sum(eigen_vals)
# var_exp ratio is fraction of eigen_val to total sum
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
# calculate the cumulative sum of explained variances
cum_var_exp = np.cumsum(var_exp)
plt.bar(range(1, 42), var_exp, alpha=0.75, align='center',label='individual explained variance')
plt.step(range(1, 42), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylim(0, 1.1)
plt.xlabel('Principal components')
plt.ylabel('Explained variance ratio')
plt.legend(loc='best')
plt.show()
pca_33 = PCA(n_components=33) #PCA with 8 primary components

# fit and transform both PCA models
X_pca_33 = pca_33.fit_transform(X_logis_number)
print(X_pca_33.shape)
X_PCA_train,X_PCA_test,Y_PCA_train,Y_PCA_test=train_test_split(X_pca_33,Y_cate,test_size=0.3,random_state=0)
Logistic_PCA=LogisticRegression(penalty='l2', dual=False, tol=0.001, 
                           C=0.1, fit_intercept=True, intercept_scaling=1.0, 
                           class_weight=None, random_state=None)
Logistic_PCA.fit(X_PCA_train,Y_PCA_train)

Y_PCA_logis_pred=Logistic_PCA.predict(X_PCA_test)
# Performance
Logistics_PCA_report=metrics.classification_report(Y_PCA_test,Y_PCA_logis_pred)
print(Logistics_PCA_report)
metrics.confusion_matrix(Y_logis_test,Y_logis_pred)
X.head(5)
X_number.shape
# Label encoding
X_DT=X.copy()
Label_encoding=LabelEncoder()
X_DT["UniqueCarrier"] = Label_encoding.fit_transform(X["UniqueCarrier"])
X_DT["Origin"] = Label_encoding.fit_transform(X["Origin"])
X_DT["Dest"] = Label_encoding.fit_transform(X["Dest"])
X_DT["season"] = Label_encoding.fit_transform(X["season"])
X_DT["week"] = Label_encoding.fit_transform(X["week"])
X_DT["day"] = Label_encoding.fit_transform(X["day"])
X_DT.head(5)

X_DT_train,X_DT_test,Y_DT_train,Y_DT_test=train_test_split(X_DT,Y_cate,test_size=0.3,random_state=0)
DecisionTree=DecisionTreeClassifier()
DecisionTree.fit(X_DT_train,Y_DT_train)
Y_DT_pred=DecisionTree.predict(X_DT_test)
DT_report=metrics.classification_report(Y_DT_test,Y_DT_pred)
print(DT_report)
parameters = {"max_depth":[1,10,20,30],'min_samples_split':[0.1,1.0,10],"min_samples_leaf":[0.1,0.5,5]}
clf_DT=GridSearchCV(DecisionTreeClassifier(),parameters,cv=5,return_train_score=False)
clf_DT.fit(X_DT_train,Y_DT_train)
df_DT_clf=pd.DataFrame(clf_DT.cv_results_)
print(df_DT_clf)
clf_DT.best_score_
clf_DT.best_params_
DecisionTree_tune=DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, min_samples_split=10)
t_DT_train_start=time.time()
DecisionTree_tune.fit(X_DT_train,Y_DT_train)
t_DT_train=time.time()-t_DT_train_start
t_DT_predict_start=time.time()
Y_DT_pred=DecisionTree_tune.predict(X_DT_test)
t_DT_predict=time.time()-t_DT_predict_start
DT_report=metrics.classification_report(Y_DT_test,Y_DT_pred)
print(DT_report)
metrics.confusion_matrix(Y_DT_test,Y_DT_pred)
print("Time training :",t_DT_train)
print("Time predict  :",t_DT_predict)
kfold=KFold(n_splits=10, shuffle=False, random_state=None)
scores_DT= cross_val_score(DecisionTree_tune, X_DT, Y_cate,cv=kfold)
print( scores_DT)
scores_DT.max()
RF=RandomForestClassifier(n_estimators=10)
RF=RF.fit(X_DT_train, Y_DT_train)
Y_RF_pred=RF.predict(X_DT_test)
RF_report=metrics.classification_report(Y_DT_test,Y_RF_pred)
print(RF_report)
parameters = {"n_estimators":[1,10,100],"max_depth":[10],'min_samples_split':[1.0,10],"min_samples_leaf":[0.1,0.5,5]}
clf_RF=GridSearchCV(RandomForestClassifier(),parameters,cv=5,return_train_score=False)
clf_RF.fit(X_DT_train,Y_DT_train)
df_RF_clf=pd.DataFrame(clf_RF.cv_results_)
print(df_RF_clf)
clf_RF.best_score_
clf_RF.best_params_
RF_tune=RandomForestClassifier(max_depth=10,min_samples_leaf=5,min_samples_split=10,n_estimators=100)
t_RF_train_start=time.time()
RF_tune.fit(X_DT_train,Y_DT_train)
t_RF_train=time.time()-t_RF_train_start
t_RF_predict_start=time.time()
Y_RF_pred=RF_tune.predict(X_DT_test)
t_RF_predict=time.time()-t_RF_predict_start
RF_report=metrics.classification_report(Y_DT_test,Y_RF_pred)
print(RF_report)
print("Time train :",t_RF_train)
print("Time predict :",t_RF_predict)

kfold=KFold(n_splits=10, shuffle=False, random_state=None)
scores_RF= cross_val_score(RF_tune, X_DT, Y_cate,cv=kfold)
print( scores_RF)
scores_RF.max()
gnb = GaussianNB()
t_NB_train_start=time.time()
Naive_bayes = gnb.fit(X_DT_train, Y_DT_train)
t_NB_train=time.time()-t_NB_train_start
t_NB_predict_start=time.time()
Y_NB_pred=Naive_bayes.predict(X_DT_test)
t_NB_predict=time.time()-t_NB_predict_start
NB_report=metrics.classification_report(Y_DT_test,Y_NB_pred)
print(NB_report)
clf_NB=metrics.accuracy_score(Y_DT_test,Y_NB_pred)
print(clf_NB)
kfold=KFold(n_splits=10, shuffle=False, random_state=None)
scores_NB= cross_val_score(Naive_bayes, X_DT, Y_cate,cv=kfold)
print( scores_NB)
scores_NB.max()
knn = KNeighborsClassifier()
knn = knn.fit(X_DT_train, Y_DT_train)
Y_knn_pred=knn.predict(X_DT_test)
knn_report=metrics.classification_report(Y_DT_test,Y_knn_pred)
print(knn_report)
parameters = {'n_neighbors': [1,5,9,15,21,25]}
clf_knn=GridSearchCV(KNeighborsClassifier(),parameters,cv=5,return_train_score=False,scoring = 'accuracy')
clf_knn.fit(X_DT_train,Y_DT_train)
df_knn_clf=pd.DataFrame(clf_knn.cv_results_)
print(df_knn_clf)
clf_knn.best_score_
clf_knn.best_params_
knn_tune = KNeighborsClassifier(n_neighbors= 25)
t_knn_train_start=time.time()
knn_tune.fit(X_DT_train,Y_DT_train)
t_knn_train=time.time()-t_knn_train_start
t_knn_predict_start=time.time()
Y_knn_pred=knn_tune.predict(X_DT_test)
t_knn_predict=time.time()-t_knn_predict_start
knn_report=metrics.classification_report(Y_DT_test,Y_knn_pred)
print(knn_report)
kfold=KFold(n_splits=10, shuffle=False, random_state=None)
scores_knn= cross_val_score(knn_tune, X_DT, Y_cate,cv=kfold)
print( scores_knn)
print(scores_knn.max())
#GBC=GradientBoostingClassifier()
#GBC = GBC.fit(X_DT_train, Y_DT_train)
#Y_GBC_pred=GBC.predict(X_DT_test)

#GBC_report=metrics.classification_report(Y_DT_test,Y_GBC_pred)
#print(GBC_report)
#SVM = SVC()
#t_SVM_start_train=time.time()
#SVM = SVM.fit(X_DT_train, Y_DT_train)
#t_SVM_train=time.time()-t_SVM_start_train

#t_SVM_start_predict=time.time()
#Y_SVM_pred=SVM.predict(X_DT_test)
#T_SVM_predict=time.time()-t_SVM_start_predict
#SVM_report=metrics.classification_report(Y_DT_test,Y_SVM_pred)
#print(SVM_report)
#parameters = {'Cs':[0.001, 0.1, 10],'gammas':[0.001, 1]}
#clf_SVM=GridSearchCV(SVC(),parameters,cv=5,return_train_score=False,scoring = 'accuracy')
#clf_SVM.fit(X_DT_train,Y_DT_train)
#df_SVM_clf=pd.DataFrame(clf_SVM.cv_results_)
#print(df_SVM_clf)
LG_F1_score=metrics.f1_score(Y_logis_test, Y_logis_pred)
print(LG_F1_score)
DT_F1_score=metrics.f1_score(Y_DT_test, Y_DT_pred)
print(DT_F1_score)
RF_F1_score=metrics.f1_score(Y_DT_test, Y_RF_pred)
print(RF_F1_score)
NB_F1_score=metrics.f1_score(Y_DT_test, Y_NB_pred)
print(NB_F1_score)
knn_F1_score=metrics.f1_score(Y_DT_test, Y_knn_pred)
print(knn_F1_score)
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_logis_test, Y_logis_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
LG_gini=2*roc_auc-1
print(LG_gini)
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_DT_test, Y_DT_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
DT_gini=2*roc_auc-1
print(DT_gini)
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_DT_test, Y_RF_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
RF_gini=2*roc_auc-1
print(RF_gini)
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_DT_test, Y_NB_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
NB_gini=2*roc_auc-1
print(NB_gini)
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_DT_test, Y_knn_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
knn_gini=2*roc_auc-1
print(knn_gini)
scores={'Score':[clf_logistics.best_score_,clf_DT.best_score_,clf_RF.best_score_,clf_NB,clf_knn.best_score_],
        "Cross_Validitation":[scores_logistic.max(),scores_DT.max(),scores_RF.max(),scores_NB.max(),scores_knn.max()],
        "Time Train":[t_LG_train,t_DT_train,t_RF_train,t_NB_train,t_knn_train],
        "Time Predict":[t_LG_predict,t_DT_predict,t_RF_predict,t_NB_predict,t_knn_predict],
        "F1-score":[LG_F1_score,DT_F1_score,RF_F1_score,NB_F1_score,knn_F1_score],
         "Gini":[LG_gini,DT_gini,RF_gini,NB_gini,knn_gini]}
index=["Logistic","DecisionTree","RandomForest","NavieBayes","KNN"]
df_compare=pd.DataFrame(scores,index=index)
print(df_compare)
X_DT.shape
Y_cate.shape
X_DT.to_csv("X_DT.csv")
Y_cate.to_csv("Y_cate.csv")
df_compare.to_csv("df_compare.csv")