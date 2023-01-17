import pandas as pd
import numpy as np 

#Görselleştirme kütüphaneleri
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
import missingno as msno

print("pandas",pd.__version__)
print("seaborn",sns.__version__)
print("missingno",msno.__version__)
print("numpy",np.__version__)

Data = pd.read_csv('../input/INGDatathonData2018.csv',sep=(','))

print(Data.info(verbose=True))

Data.shape
Data.head()
print(Data.describe())
print(Data['DEFAULT_FLAG'].value_counts())
plt.figure(1, figsize=(4,4))
Data['DEFAULT_FLAG'].value_counts().plot.pie(autopct="%1.1f%%")
plt.show()
Data=Data.drop(['SHARE_OF_TL_FACTORING_RISK','DT_MAX_TL_NON_CASH_LMT_UTIL3','DT_MAX_TL_NON_CASH_LMT_UTIL12','MAX_TTL_TL_CSH_LIMIT12','STDDEV_TTL_TL_CASH_RSK12','TTL_INDM_TL_NONCASH_LOAN','MAX_TTL_TL_CASH_RSK6','MAX_TTL_TL_CASH_RSK3','MAX_TTL_TL_CASH_RSK12','STDDEV_FACTORING_TL_RSK12','AVG_TTL_TL_NON_CASH_RSK6','MAX_TTL_TL_CSH_LIMIT6','MAX_TTL_TL_CSH_LIMIT3','TOTAL_TL_RSK','TOTAL_TL_CASH_RSK','TOTAL_TL_NON_CASH_RSK','TOTAL_TL_LIMIT','TOTAL_TL_CASH_LIMIT','TOTAL_TL_NON_CASH_LIMIT','CASH_LIMIT_TL_UTILIZATION','AVG_FACTORING_TL_RSK12','AVG_FACTORING_TL_RSK3','AVG_FACTORING_TL_RSK6','AVG_TTL_TL_CASH_RSK12','AVG_TTL_TL_CASH_RSK3','AVG_TTL_TL_CASH_RSK6','AVG_TTL_TL_CSH_LIMIT12','AVG_TTL_TL_CSH_LIMIT3','AVG_TTL_TL_CSH_LIMIT6','MAX_TTL_TL_LIMIT12','MAX_TTL_TL_LIMIT6','MAX_TTL_TL_LIMIT3','MAX_TTL_TL_NON_CSH_LIMIT12','MAX_TTL_TL_NON_CSH_LIMIT6','TL_NON_CASH_LIMIT_UTILIZATION','AVG_TTL_TL_LIMIT12','AVG_TTL_TL_LIMIT3','AVG_TTL_TL_LIMIT6','AVG_TTL_TL_NON_CASH_RSK12','AVG_TTL_TL_NON_CASH_RSK3','AVG_TTL_TL_NON_CSH_LIMIT12','AVG_TTL_TL_NON_CSH_LIMIT3','AVG_TTL_TL_NON_CSH_LIMIT6','AVG_TTL_TL_RSK12','AVG_TTL_TL_RSK6','AVG_TTL_TL_RSK3','MAX_FACTORING_TL_RSK12','MAX_FACTORING_TL_RSK6','MAX_FACTORING_TL_RSK3','MAX_INDM_TL_NON_CASH_LOANS12','MAX_INDM_TL_NON_CASH_LOANS6','MAX_INDM_TL_NON_CASH_LOANS3','MAX_TTL_TL_NON_CSH_LIMIT3','MAX_TTL_TL_RSK12','MAX_TTL_TL_RSK6','MAX_TTL_TL_RSK3','MIN_FACTORING_TL_RSK12','MIN_FACTORING_TL_RSK6','MIN_FACTORING_TL_RSK3','MIN_TTL_TL_CASH_RSK12','MIN_TTL_TL_CASH_RSK6','MIN_TTL_TL_CASH_RSK3','MIN_TTL_TL_CSH_LIMIT12','MIN_TTL_TL_CSH_LIMIT6','MIN_TTL_TL_CSH_LIMIT3','MIN_TTL_TL_LIMIT12','MIN_TTL_TL_LIMIT6','MIN_TTL_TL_LIMIT3','MIN_TTL_TL_NON_CSH_LIMIT12','MIN_TTL_TL_NON_CSH_LIMIT6','MIN_TTL_TL_NON_CSH_LIMIT6','MIN_TTL_TL_NON_CSH_LIMIT3','MIN_TTL_TL_RSK12','MIN_TTL_TL_RSK6','MIN_TTL_TL_RSK3','AVG_CASH_LIMIT_TL_UTIL12','AVG_CASH_LIMIT_TL_UTIL6','AVG_CASH_LIMIT_TL_UTIL3','AVG_TL_NON_CASH_LIMIT_UTIL12','AVG_TL_NON_CASH_LIMIT_UTIL6','AVG_TL_NON_CASH_LIMIT_UTIL3','CASH_LIMIT_TL_UTIL_GR12','CASH_LIMIT_TL_UTIL_GR6','CASH_LIMIT_TL_UTILIZATION','CASH_LIMIT_TL_UTLZTN_DNM_FRK6','CASH_LMT_TL_UTLZTN_DNM_FRK12','DT_MAX_FACTORING_TL_RSK12','DT_MAX_FACTORING_TL_RSK6','DT_MAX_FACTORING_TL_RSK12','DT_MAX_FACTORING_TL_RSK6','DT_MAX_FACTORING_TL_RSK3','DT_MAX_INDM_TL_NON_CSH_LOANS12','DT_MAX_INDM_TL_NON_CSH_LOANS6','DT_MAX_INDM_TL_NON_CSH_LOANS3','DT_MAX_TTL_TL_CASH_RSK12','DT_MAX_TTL_TL_CASH_RSK6','DT_MAX_TTL_TL_CASH_RSK3','DT_MAX_TTL_TL_CSH_LIMIT12','DT_MAX_TTL_TL_CSH_LIMIT6','DT_MAX_TTL_TL_CSH_LIMIT3','DT_MAX_TTL_TL_LIMIT12','DT_MAX_TTL_TL_LIMIT6','DT_MAX_TTL_TL_LIMIT3','DT_MAX_TTL_TL_NON_CASH_RSK12','DT_MAX_TTL_TL_NON_CASH_RSK6','DT_MAX_TTL_TL_NON_CASH_RSK3','DT_MAX_TTL_TL_NON_CSH_LMT12','DT_MAX_TTL_TL_NON_CSH_LMT6','DT_MAX_TTL_TL_NON_CSH_LMT3','DT_MAX_TTL_TL_RSK12','DT_MAX_TTL_TL_RSK6','DT_MAX_TTL_TL_RSK3','DT_MIN_FACTORING_TL_RSK12','DT_MIN_FACTORING_TL_RSK6','DT_MIN_FACTORING_TL_RSK3','DT_MIN_INDM_TL_NON_CSH_LOANS12','DT_MIN_INDM_TL_NON_CSH_LOANS12','DT_MIN_INDM_TL_NON_CSH_LOANS6','DT_MIN_INDM_TL_NON_CSH_LOANS3','DT_MIN_TL_NON_CASH_LMT_UTIL12','DT_MIN_TL_NON_CASH_LMT_UTIL6','DT_MIN_TL_NON_CASH_LMT_UTIL3','DT_MIN_TTL_TL_CASH_RSK12','DT_MIN_TTL_TL_CASH_RSK6','DT_MIN_TTL_TL_CASH_RSK3','DT_MIN_TTL_TL_CSH_LIMIT12','DT_MIN_TTL_TL_CSH_LIMIT6','DT_MIN_TTL_TL_CSH_LIMIT3','DT_MIN_TTL_TL_LIMIT12','DT_MIN_TTL_TL_LIMIT6','DT_MIN_TTL_TL_LIMIT3','DT_MIN_TTL_TL_NON_CASH_RSK12','DT_MIN_TTL_TL_NON_CASH_RSK6','DT_MIN_TTL_TL_NON_CASH_RSK3','DT_MIN_TTL_TL_NON_CSH_LMT12','DT_MIN_TTL_TL_NON_CSH_LMT6','DT_MIN_TTL_TL_NON_CSH_LMT3','DT_MIN_TTL_TL_RSK12','DT_MIN_TTL_TL_RSK6','DT_MIN_TTL_TL_RSK3','FACTORING_TL_RSK_GR6','MAX_CASH_LIMIT_TL_UTIL12','MAX_CASH_LIMIT_TL_UTIL6','MAX_CASH_LIMIT_TL_UTIL3','MAX_TL_NON_CASH_LIMIT_UTIL12','MAX_TL_NON_CASH_LIMIT_UTIL6','MAX_TL_NON_CASH_LIMIT_UTIL3','MAX_TTL_TL_NON_CASH_RSK12','MAX_TTL_TL_NON_CASH_RSK6','MAX_TTL_TL_NON_CASH_RSK3','MIN_CASH_LIMIT_TL_UTIL12','MIN_CASH_LIMIT_TL_UTIL6','MIN_CASH_LIMIT_TL_UTIL3','MIN_TL_NON_CASH_LIMIT_UTIL12','MIN_TL_NON_CASH_LIMIT_UTIL6','MIN_TL_NON_CASH_LIMIT_UTIL3','MIN_TTL_TL_NON_CASH_RSK12','MIN_TTL_TL_NON_CASH_RSK6','MIN_TTL_TL_NON_CASH_RSK3','STD_TL_NON_CASH_LIMIT_UTIL12','STDDEV_TTL_TL_CSH_LIMIT12','STDDEV_TTL_TL_LIMIT12','STDDEV_TTL_TL_NON_CASH_RSK12','STDDEV_TTL_TL_NON_CSH_LIMIT12','STDDEV_TTL_TL_RSK12','TL_NN_CSH_LMT_UTZTN_DNM_FRK12','TL_NON_CASH_LIMIT_UTIL_GR12','TL_NON_CASH_LIMIT_UTIL_GR6','TL_NON_CASH_LIMIT_UTILIZATION','TL_TOTAL_CASH_RSK_DNM_FRK12','TOTAL_TL_CASH_LIMIT','TOTAL_TL_CASH_RSK','TOTAL_TL_CASH_RSK_GR12','TOTAL_TL_CASH_RSK_GR6','TOTAL_TL_CSH_LIMIT_GR12','TOTAL_TL_CSH_LIMIT_GR6','TOTAL_TL_LIMIT','TOTAL_TL_LIMIT_GR12','TOTAL_TL_LIMIT_GR6','TOTAL_TL_LIMIT_UTILIZATION','TOTAL_TL_NON_CASH_LIMIT','TOTAL_TL_NON_CASH_RSK','TOTAL_TL_NON_CASH_RSK_GR12','TOTAL_TL_NON_CASH_RSK_GR6','TOTAL_TL_NON_CSH_LIMIT_GR12','TOTAL_TL_NON_CSH_LIMIT_GR6','TOTAL_TL_RSK','TOTAL_TL_RSK_GR12','TOTAL_TL_RSK_GR6','TTL_TL_LMT_UTLZTN_DNM_FRK12'],axis=1)

corres = Data.corr()
print(corres)
msno.heatmap(Data, figsize=(20,20))#görsel olarak missing valuelar
plt.show()


msno.matrix(Data)
msno.bar(Data,fontsize=10,figsize=(50,50))

data_credict=Data.loc[Data['DEFAULT_FLAG'] ==0] #.iloc[:20000,:]
non_credict=Data.loc[Data['DEFAULT_FLAG'] ==1]#.iloc[:1000,:]

msno.matrix(data_credict)
msno.bar(data_credict,fontsize=10,figsize=(50,50))
# yakşlaşık 20000 den sonraki verilerde boş değerler oldukça fazla
msno.matrix(non_credict) #yaklaşık 1000 sonrası boş
msno.bar(non_credict,fontsize=10,figsize=(50,50))
plt.show()

data_credict=Data.loc[Data['DEFAULT_FLAG'] ==0].iloc[:20000,:]
non_credict=Data.loc[Data['DEFAULT_FLAG'] ==1].iloc[:1200,:]
msno.matrix(data_credict)
msno.bar(data_credict,fontsize=10,figsize=(50,50))
msno.matrix(non_credict) #yaklaşık 1000 sonrası boş
msno.bar(non_credict,fontsize=10,figsize=(50,50))
plt.show()

DataX=pd.concat([data_credict,non_credict],axis=0)
DataX.DEFAULT_FLAG.value_counts()

null_data_sum =DataX.isnull().sum()*100/(len(DataX))
print(null_data_sum)
df_miss_val_count_unsort = pd.DataFrame(data=null_data_sum, columns=['rate'])
df_miss_val_count_unsort.head()
n_most_missing =((df_miss_val_count_unsort.rate.values >50)==True).sum()
print('%50 si bos olan kolon sayisi',n_most_missing)
df_miss_val_count_unsort['rate'].value_counts()
df_miss_val_count = pd.DataFrame(data=null_data_sum, columns=['rate']).sort_values(by=['rate'],ascending=False)[:n_most_missing]
print(df_miss_val_count)
Data2=DataX.drop(columns=df_miss_val_count.index.values)
print(Data2.shape)

null_data_sum2 =Data2.isnull().sum()*100/(len(Data2))
print(null_data_sum2)
msno.matrix(Data2)
msno.bar(Data2,fontsize=10,figsize=(50,50))
plt.show()

from statsmodels.imputation import mice

imp = mice.MICEData(Data2,perturbation_method='hot deck')
Data2=imp.data
Data3=Data2
print(Data2.isnull().sum())

msno.bar(Data2,fontsize=10,figsize=(50,50))
msno.matrix(Data2)
plt.show()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
ids=Data2.PRIMARY_KEY
flag=Data2.DEFAULT_FLAG
Data2.drop(['PRIMARY_KEY','DEFAULT_FLAG'],inplace=True,axis=1)

columns=Data2.columns.values
data_scaler=scaler.fit_transform(Data2)

Data2=pd.DataFrame(data_scaler,columns=columns)
Data2.head()
Data2['PRIMARY_KEY']=ids
Data2['DEFAULT_FLAG']=flag
Data2['DEFAULT_FLAG'].value_counts()
Data2.head()

plt.figure(1, figsize=(4,4))
Data2['DEFAULT_FLAG'].value_counts().plot.pie(autopct="%1.1f%%")
plt.show()
X=Data2.iloc[:,:-1]
y=Data2.iloc[:,-1:]

y.head()
#X.head()
X.tail()
X.shape
X.head()
#y.head()
colors = ['#ff6200' if v == 0 else '#f7f7f7' for v in Data2.DEFAULT_FLAG]
kwarg_params = {'linewidth': 1, 'edgecolor': 'black'}
fig = plt.Figure(figsize=(30,30))
plt.scatter(Data2.AVG_TTL_ACCRD_INT_AMT12, Data2.TOTAL_RSK,c=colors, **kwarg_params)
plt.title('unblance data AVG_TTL_ACCRD_INT_AMT12-TOTAL_RSK')
plt.show()
plt.scatter(Data2.PRIMARY_KEY, Data2.TOTAL_RSK,c=colors, **kwarg_params)
plt.title('unbalance data TOTAL_RSK')
plt.show()
plt.scatter(Data2.AVG_TTL_CASH_RSK12, Data2.TOTAL_RSK,c=colors, **kwarg_params)
plt.title('unbalance AVG_TTL_CASH_RSK12-TOTAL_RSK')
plt.show()

#from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=10)
columns=X.columns.values
X_resampled, y_resampled = rus.fit_resample(X, y)
X_resampled=pd.DataFrame(X_resampled,columns=columns)
y_resampled=pd.DataFrame(y_resampled,columns=['DEFAULT_FLAG'])

X_resampled.head()
Data4=pd.concat([y_resampled,X_resampled],axis=1)
type(Data4)
Data4.shape
Data4.head()
Data4.tail()
Data4['DEFAULT_FLAG'].value_counts()
colors = ['#ff6200' if v == 0 else '#f7f7f7' for v in Data4.DEFAULT_FLAG]
kwarg_params = {'linewidth': 1, 'edgecolor': 'black'}
fig = plt.Figure(figsize=(30,30))
plt.scatter(Data4.AVG_TTL_ACCRD_INT_AMT12, Data4.TOTAL_RSK,c=colors, **kwarg_params)
plt.title('balance data AVG_TTL_ACCRD_INT_AMT12-TOTAL_RSK')
plt.show()
plt.scatter(Data4.PRIMARY_KEY, Data4.TOTAL_RSK,c=colors, **kwarg_params)
plt.title('balance data TOTAL_RSK')
plt.show()
plt.scatter(Data4.AVG_TTL_CASH_RSK12, Data4.TOTAL_RSK,c=colors, **kwarg_params)
plt.title('balance AVG_TTL_CASH_RSK12-TOTAL_RSK')
plt.show()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_resampled,y_resampled, test_size =0.3,random_state =10)

X_test
#y_test.tail()
X_train.tail()
finaly=X_test.iloc[:,-1:]
finaly
#idstest=X_test.PRIMARY_KEY.values
#idstrain=X_train.PRIMARY_KEY

X_train.drop(['PRIMARY_KEY'],inplace=True,axis=1)
X_test.drop(['PRIMARY_KEY'],inplace=True,axis=1)

X_test.head()
#type(idstest)

#### XGBRegressor
#### LogisticRegression
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)

print("LogisticRegression score",log_reg.score(X_test,y_test))
#from sklearn.model_selection import GridSearchCV
#grid = {"C":np.logspace(-3,3,7),"penalty" :["l1","l2"]}
#logreg_cv = GridSearchCV(log_reg,grid,cv=10)
#logreg_cv.fit(X_train,y_train)
#print("LogisticRegression en iyi hyperlar",logreg_cv.best_params_) #l1 10.0 print("LogisticRegression en iyi score",logreg_cv.best_score_) #0.67
#log_reg2 = LogisticRegression(C=10,penalty ="l1") log_reg2.fit(X_train,y_train) print("LogisticRegression score",log_reg2.score(X_test,y_test))

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = knn ,X=X_train, y=y_train, cv=10 )
print("dogruluk",np.mean(accuracies)) #0.63
print("dogruluk",np.std(accuracies))
y_pred_knn = knn.predict_proba(X_test)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred_knn)


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
scores = cross_val_score(clf, X_train, y_train, cv=5)
scores.mean()
from sklearn.ensemble import RandomForestClassifier
c= RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
scores = cross_val_score(c, X_train, y_train, cv=5)
scores.mean()  
from sklearn.svm import SVC
svc = SVC(kernel='poly')
svc.fit(X_train,y_train)

y_predq = svc.predict(X_test)
print("DecisionTreeClassifier accurc:",svc.score(X_test,y_test))


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=50, criterion = 'gini',random_state=40) #entropy
rfc.fit(X_train,y_train)

y_pred4 = rfc.predict(X_test)
print("DecisionTreeClassifier accurc:",rfc.score(X_test,y_test))
from sklearn.decomposition import PCA

pca =PCA(n_components=20,whiten=True)
pca_x=pca.fit_transform(X_resampled.values)
print("varyans" ,sum(pca.explained_variance_ratio_))
    



x_pca=pd.DataFrame(pca_x,columns=[
 'PC5','PC1','PC2','PC3','PC4','PC5','PC1','PC2','PC3','PC4','PC5'
'PC5','PC1','PC2','PC3','PC4','PC5','PC1','PC2','PC3','PC4'])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x_pca,y_resampled, test_size =0.3,random_state =42)
log_reg2 = LogisticRegression()
log_reg2.fit(X_train,y_train)
y_p = log_reg2.predict(X_test)
print("LogisticRegression score",log_reg2.score(X_test,y_test))

from sklearn.decomposition import PCA

pca =PCA(n_components=1,whiten=True)
xscore=pca.fit_transform(X_test.values)
print("varyans" ,sum(pca.explained_variance_ratio_))
    

xscore


from sklearn.metrics import confusion_matrix
cm1= confusion_matrix(y_test,y_p)
cm1
rfc = RandomForestClassifier(n_estimators=40, criterion = 'gini',random_state=40) #entropy
rfc.fit(X_train,y_train)
predrfc = log_reg2.predict(X_test)
print("random forestClassifier accurc:",rfc.score(X_test,y_test))
from sklearn.metrics import roc_curve
y_pred_rf = rfc.predict_proba(X_test)[:, 1]
fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf_lm, tpr_rf_lm, label='random-f')
plt.plot(fpr_knn, tpr_knn, label='knn')
#plt.plot(fpr_rf, tpr_rf, label='RF')
#plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
#plt.plot(fpr_grd, tpr_grd, label='GBT')
#lt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,predrfc)
cm
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = rfc ,X=X_train, y=y_train, cv=10 )
print("dogruluk",np.mean(accuracies)) 
print("std",np.std(accuracies))

finaly['T']=xscore
finaly['Score']=predrfc
finaly['ff']=finaly['T']+finaly['Score']
finaly['ff']=finaly['ff']*-100
finaly.drop(['Score','T'],inplace=True,axis=1)

finaly
dddd=finaly.values
import tableprint
#import numpy as np

data = dddd
headers = ['PRIMARY_KEY', 'SCORE']

tableprint.table(data, headers)
def credictscore( str ):
    import pandas as pd
    #k.append(str)
    str2 = ''.join(str)
    str2=str2 +'.csv'
    Data = pd.read_csv(str2,sep=(';'))
    Data=Data.drop(['SHARE_OF_TL_FACTORING_RISK','DT_MAX_TL_NON_CASH_LMT_UTIL3','DT_MAX_TL_NON_CASH_LMT_UTIL12','MAX_TTL_TL_CSH_LIMIT12','STDDEV_TTL_TL_CASH_RSK12','TTL_INDM_TL_NONCASH_LOAN','MAX_TTL_TL_CASH_RSK6','MAX_TTL_TL_CASH_RSK3','MAX_TTL_TL_CASH_RSK12','STDDEV_FACTORING_TL_RSK12','AVG_TTL_TL_NON_CASH_RSK6','MAX_TTL_TL_CSH_LIMIT6','MAX_TTL_TL_CSH_LIMIT3','TOTAL_TL_RSK','TOTAL_TL_CASH_RSK','TOTAL_TL_NON_CASH_RSK','TOTAL_TL_LIMIT','TOTAL_TL_CASH_LIMIT','TOTAL_TL_NON_CASH_LIMIT','CASH_LIMIT_TL_UTILIZATION','AVG_FACTORING_TL_RSK12','AVG_FACTORING_TL_RSK3','AVG_FACTORING_TL_RSK6','AVG_TTL_TL_CASH_RSK12','AVG_TTL_TL_CASH_RSK3','AVG_TTL_TL_CASH_RSK6','AVG_TTL_TL_CSH_LIMIT12','AVG_TTL_TL_CSH_LIMIT3','AVG_TTL_TL_CSH_LIMIT6','MAX_TTL_TL_LIMIT12','MAX_TTL_TL_LIMIT6','MAX_TTL_TL_LIMIT3','MAX_TTL_TL_NON_CSH_LIMIT12','MAX_TTL_TL_NON_CSH_LIMIT6','TL_NON_CASH_LIMIT_UTILIZATION','AVG_TTL_TL_LIMIT12','AVG_TTL_TL_LIMIT3','AVG_TTL_TL_LIMIT6','AVG_TTL_TL_NON_CASH_RSK12','AVG_TTL_TL_NON_CASH_RSK3','AVG_TTL_TL_NON_CSH_LIMIT12','AVG_TTL_TL_NON_CSH_LIMIT3','AVG_TTL_TL_NON_CSH_LIMIT6','AVG_TTL_TL_RSK12','AVG_TTL_TL_RSK6','AVG_TTL_TL_RSK3','MAX_FACTORING_TL_RSK12','MAX_FACTORING_TL_RSK6','MAX_FACTORING_TL_RSK3','MAX_INDM_TL_NON_CASH_LOANS12','MAX_INDM_TL_NON_CASH_LOANS6','MAX_INDM_TL_NON_CASH_LOANS3','MAX_TTL_TL_NON_CSH_LIMIT3','MAX_TTL_TL_RSK12','MAX_TTL_TL_RSK6','MAX_TTL_TL_RSK3','MIN_FACTORING_TL_RSK12','MIN_FACTORING_TL_RSK6','MIN_FACTORING_TL_RSK3','MIN_TTL_TL_CASH_RSK12','MIN_TTL_TL_CASH_RSK6','MIN_TTL_TL_CASH_RSK3','MIN_TTL_TL_CSH_LIMIT12','MIN_TTL_TL_CSH_LIMIT6','MIN_TTL_TL_CSH_LIMIT3','MIN_TTL_TL_LIMIT12','MIN_TTL_TL_LIMIT6','MIN_TTL_TL_LIMIT3','MIN_TTL_TL_NON_CSH_LIMIT12','MIN_TTL_TL_NON_CSH_LIMIT6','MIN_TTL_TL_NON_CSH_LIMIT6','MIN_TTL_TL_NON_CSH_LIMIT3','MIN_TTL_TL_RSK12','MIN_TTL_TL_RSK6','MIN_TTL_TL_RSK3','AVG_CASH_LIMIT_TL_UTIL12','AVG_CASH_LIMIT_TL_UTIL6','AVG_CASH_LIMIT_TL_UTIL3','AVG_TL_NON_CASH_LIMIT_UTIL12','AVG_TL_NON_CASH_LIMIT_UTIL6','AVG_TL_NON_CASH_LIMIT_UTIL3','CASH_LIMIT_TL_UTIL_GR12','CASH_LIMIT_TL_UTIL_GR6','CASH_LIMIT_TL_UTILIZATION','CASH_LIMIT_TL_UTLZTN_DNM_FRK6','CASH_LMT_TL_UTLZTN_DNM_FRK12','DT_MAX_FACTORING_TL_RSK12','DT_MAX_FACTORING_TL_RSK6','DT_MAX_FACTORING_TL_RSK12','DT_MAX_FACTORING_TL_RSK6','DT_MAX_FACTORING_TL_RSK3','DT_MAX_INDM_TL_NON_CSH_LOANS12','DT_MAX_INDM_TL_NON_CSH_LOANS6','DT_MAX_INDM_TL_NON_CSH_LOANS3','DT_MAX_TTL_TL_CASH_RSK12','DT_MAX_TTL_TL_CASH_RSK6','DT_MAX_TTL_TL_CASH_RSK3','DT_MAX_TTL_TL_CSH_LIMIT12','DT_MAX_TTL_TL_CSH_LIMIT6','DT_MAX_TTL_TL_CSH_LIMIT3','DT_MAX_TTL_TL_LIMIT12','DT_MAX_TTL_TL_LIMIT6','DT_MAX_TTL_TL_LIMIT3','DT_MAX_TTL_TL_NON_CASH_RSK12','DT_MAX_TTL_TL_NON_CASH_RSK6','DT_MAX_TTL_TL_NON_CASH_RSK3','DT_MAX_TTL_TL_NON_CSH_LMT12','DT_MAX_TTL_TL_NON_CSH_LMT6','DT_MAX_TTL_TL_NON_CSH_LMT3','DT_MAX_TTL_TL_RSK12','DT_MAX_TTL_TL_RSK6','DT_MAX_TTL_TL_RSK3','DT_MIN_FACTORING_TL_RSK12','DT_MIN_FACTORING_TL_RSK6','DT_MIN_FACTORING_TL_RSK3','DT_MIN_INDM_TL_NON_CSH_LOANS12','DT_MIN_INDM_TL_NON_CSH_LOANS12','DT_MIN_INDM_TL_NON_CSH_LOANS6','DT_MIN_INDM_TL_NON_CSH_LOANS3','DT_MIN_TL_NON_CASH_LMT_UTIL12','DT_MIN_TL_NON_CASH_LMT_UTIL6','DT_MIN_TL_NON_CASH_LMT_UTIL3','DT_MIN_TTL_TL_CASH_RSK12','DT_MIN_TTL_TL_CASH_RSK6','DT_MIN_TTL_TL_CASH_RSK3','DT_MIN_TTL_TL_CSH_LIMIT12','DT_MIN_TTL_TL_CSH_LIMIT6','DT_MIN_TTL_TL_CSH_LIMIT3','DT_MIN_TTL_TL_LIMIT12','DT_MIN_TTL_TL_LIMIT6','DT_MIN_TTL_TL_LIMIT3','DT_MIN_TTL_TL_NON_CASH_RSK12','DT_MIN_TTL_TL_NON_CASH_RSK6','DT_MIN_TTL_TL_NON_CASH_RSK3','DT_MIN_TTL_TL_NON_CSH_LMT12','DT_MIN_TTL_TL_NON_CSH_LMT6','DT_MIN_TTL_TL_NON_CSH_LMT3','DT_MIN_TTL_TL_RSK12','DT_MIN_TTL_TL_RSK6','DT_MIN_TTL_TL_RSK3','FACTORING_TL_RSK_GR6','MAX_CASH_LIMIT_TL_UTIL12','MAX_CASH_LIMIT_TL_UTIL6','MAX_CASH_LIMIT_TL_UTIL3','MAX_TL_NON_CASH_LIMIT_UTIL12','MAX_TL_NON_CASH_LIMIT_UTIL6','MAX_TL_NON_CASH_LIMIT_UTIL3','MAX_TTL_TL_NON_CASH_RSK12','MAX_TTL_TL_NON_CASH_RSK6','MAX_TTL_TL_NON_CASH_RSK3','MIN_CASH_LIMIT_TL_UTIL12','MIN_CASH_LIMIT_TL_UTIL6','MIN_CASH_LIMIT_TL_UTIL3','MIN_TL_NON_CASH_LIMIT_UTIL12','MIN_TL_NON_CASH_LIMIT_UTIL6','MIN_TL_NON_CASH_LIMIT_UTIL3','MIN_TTL_TL_NON_CASH_RSK12','MIN_TTL_TL_NON_CASH_RSK6','MIN_TTL_TL_NON_CASH_RSK3','STD_TL_NON_CASH_LIMIT_UTIL12','STDDEV_TTL_TL_CSH_LIMIT12','STDDEV_TTL_TL_LIMIT12','STDDEV_TTL_TL_NON_CASH_RSK12','STDDEV_TTL_TL_NON_CSH_LIMIT12','STDDEV_TTL_TL_RSK12','TL_NN_CSH_LMT_UTZTN_DNM_FRK12','TL_NON_CASH_LIMIT_UTIL_GR12','TL_NON_CASH_LIMIT_UTIL_GR6','TL_NON_CASH_LIMIT_UTILIZATION','TL_TOTAL_CASH_RSK_DNM_FRK12','TOTAL_TL_CASH_LIMIT','TOTAL_TL_CASH_RSK','TOTAL_TL_CASH_RSK_GR12','TOTAL_TL_CASH_RSK_GR6','TOTAL_TL_CSH_LIMIT_GR12','TOTAL_TL_CSH_LIMIT_GR6','TOTAL_TL_LIMIT','TOTAL_TL_LIMIT_GR12','TOTAL_TL_LIMIT_GR6','TOTAL_TL_LIMIT_UTILIZATION','TOTAL_TL_NON_CASH_LIMIT','TOTAL_TL_NON_CASH_RSK','TOTAL_TL_NON_CASH_RSK_GR12','TOTAL_TL_NON_CASH_RSK_GR6','TOTAL_TL_NON_CSH_LIMIT_GR12','TOTAL_TL_NON_CSH_LIMIT_GR6','TOTAL_TL_RSK','TOTAL_TL_RSK_GR12','TOTAL_TL_RSK_GR6','TTL_TL_LMT_UTLZTN_DNM_FRK12'],axis=1)
    data_credict=Data.loc[Data['DEFAULT_FLAG'] ==0].iloc[:20000,:]
    non_credict=Data.loc[Data['DEFAULT_FLAG'] ==1].iloc[:1200,:]
    DataX=pd.concat([data_credict,non_credict],axis=0)
    null_data_sum =DataX.isnull().sum()*100/(len(DataX))
    df_miss_val_count_unsort = pd.DataFrame(data=null_data_sum, columns=['rate'])
    n_most_missing =((df_miss_val_count_unsort.rate.values >50)==True).sum()
    df_miss_val_count_unsort['rate'].value_counts()
    df_miss_val_count = pd.DataFrame(data=null_data_sum, columns=['rate']).sort_values(by=['rate'],ascending=False)[:n_most_missing]
    Data2=DataX.drop(columns=df_miss_val_count.index.values)
    from statsmodels.imputation import mice

    imp = mice.MICEData(Data2,perturbation_method='hot deck')
    Data2=imp.data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    ids=Data2.PRIMARY_KEY
    flag=Data2.DEFAULT_FLAG
    Data2.drop(['PRIMARY_KEY','DEFAULT_FLAG'],inplace=True,axis=1)

    columns=Data2.columns.values
    data_scaler=scaler.fit_transform(Data2)
    Data2=pd.DataFrame(data_scaler,columns=columns)
    
    Data2['PRIMARY_KEY']=ids
    Data2['DEFAULT_FLAG']=flag
    
    X=Data2.iloc[:,:-1]
    y=Data2.iloc[:,-1:]
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=10)
    columns=X.columns.values
    X_resampled, y_resampled = rus.fit_resample(X, y)
    X_resampled=pd.DataFrame(X_resampled,columns=columns)
    y_resampled=pd.DataFrame(y_resampled,columns=['DEFAULT_FLAG'])
    Data4=pd.concat([y_resampled,X_resampled],axis=1)
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X_resampled,y_resampled, test_size =0.3,random_state =10)
    finaly=X_test.iloc[:,-1:]
    
    X_train.drop(['PRIMARY_KEY'],inplace=True,axis=1)
    X_test.drop(['PRIMARY_KEY'],inplace=True,axis=1)
   
    pca =PCA(n_components=1,whiten=True)
    xscore=pca.fit_transform(X_test.values)
    rfc = RandomForestClassifier(n_estimators=40, criterion = 'gini',random_state=40) #entropy
    rfc.fit(X_train,y_train)
    predrfc = rfc.predict(X_test)
    
    finaly['T']=xscore
    finaly['Score']=predrfc
    finaly['ff']=finaly['T']+finaly['Score']
    finaly['ff']=finaly['ff']*-100
    finaly.drop(['Score','T'],inplace=True,axis=1)
    data1=finaly.values
    import tableprint
    #import numpy as np

    data = data1
    headers = ['PRIMARY_KEY', 'SCORE']

    tableprint.table(data1, headers)
    return  tableprint.table(data1, headers);
credictscore("INGDatathonData2018")
#çapraz skorlama
