import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
df=pd.read_csv('../input/bank.csv')
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import datasets
from io import StringIO
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
%matplotlib inline
df.head(10)
df.info()
df.describe()
y=df['deposit']
X=df
y=y.to_frame()
y
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=101)
print(y.shape)
print(X.shape)
print(X_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_train.shape)
#X_train' deki kolonlarımızı görelim
X_train.columns
#X_train verilerimizin histogram grafiğine bakalım.
sns.pairplot(X_train,diag_kind='hist')
y.value_counts()
combine_=[y_train,y_test]
depositmapping={'yes':1,'no': 0}
for dt in combine_:
   dt['deposit']=dt['deposit'].map(depositmapping)
y_train
#yes olanlara 1, no olanlara 0 ataması yapalım
combine=[X_train,X_test]
depositmapping={'yes':1,'no': 0}
for dt in combine:
   dt['deposit']=dt['deposit'].map(depositmapping)
#x_train' den ilk 5 satırı görüntülüyelim
X_train.head()
#y_train den ilk 5 satırı görüntülüyelim
y_train.head()
#Kategorisel verileri analiz edelim

#Job kategorisel verisini ele alalım.

X_train[['job','deposit']].groupby('job',as_index=False).mean().sort_values('deposit',ascending=False)
for df in combine:
    df['job']=df['job'].replace(['management','technician','unknown','admin.','housemaid','self-employed','services',
                                'blue-collar','entrepreneur'],'rare',regex=True)
jobmapping={'student':1,'retired':2,'unemployed':3,'rare':0}
for df in combine:
    df['job']=df['job'].map(jobmapping)
    
#kişinin medeni durumunu analiz edelim

X_train[['marital','deposit']].groupby('marital',as_index=False).mean().sort_values('deposit',ascending=False)
#Eğitim durumunu analiz edelim
X_train[['education','deposit']].groupby('education',as_index=False).mean().sort_values('deposit',ascending=False)
educationmapping={'primary':1,'secondary':2,'tertiary':3,'unknown':0}
for df in combine:
    df['education']=df['education'].map(educationmapping)
#'default' değişken analizi

X_train[['default','deposit']].groupby('default',as_index=False).mean().sort_values('deposit',ascending=False)
defaultmapping={'no':1,'yes':0}
for df in combine:
    df['default']=df['default'].map(defaultmapping)
#Kredi değişkenini analiz edelim

X_train[['loan','deposit']].groupby('loan',as_index=False).mean().sort_values('deposit',ascending=False)
loanmapping={'no':1,'yes':0}
for df in combine:
    df['loan']=df['loan'].map(defaultmapping)
# 'contact' değişkenini analiz edelim

X_train[['contact','deposit']].groupby('contact',as_index=False).mean().sort_values('deposit',ascending=False)
#'month' değişkenini analiz edelim

X_train[['month','deposit']].groupby('month',as_index=False).mean().sort_values('deposit',ascending=False)
for df in combine:
    df['month']=df['month'].replace(['mar','dec','sep','oct'],2,regex=True)
    df['month']=df['month'].replace(['apr','feb','aug','jun'],1,regex=True)
    df['month']=df['month'].replace(['nov','jul','jan','may'],0,regex=True)
#'poutcome' değişkeni analizi

X_train[['poutcome','deposit']].groupby('poutcome',as_index=False).mean().sort_values('deposit',ascending=False)
poutcomemapping={'success':2,'other':1,'failure':0,'unknown':0}

for df in combine:
    df['poutcome']=df['poutcome'].map(poutcomemapping)
#'age' değişkeni analizi
sns.distplot(X_train['age'])
#skewness and kurtosis değerlerini hesaplayalım
from scipy.stats import kurtosis
from scipy.stats import skew

K=kurtosis(X_train['age'])
s=skew(X_train['age'])
print('k:',K)
print('s:',s)
for df in combine:
    df['age_trns']=df['age'].apply(np.log)
    
sns.distplot(X_train['age_trns'])
K=kurtosis(X_train['age_trns'])
s=skew(X_train['age_trns'])
print('k:',K)
print('s:',s)
plt.figure()
x=df[df['deposit']==1]['age_trns']
y=df[df['deposit']==0]['age_trns']
plt.hist(x,bins=20,alpha=0.5,label='deposited')
plt.hist(y,bins=20,alpha=0.5,label='did not deposit')
plt.legend()
plt.show()
for df in combine:
    df['age_bands']=pd.cut(df['age_trns'],bins=5,precision=1)
X_train[['age_bands','deposit']].groupby('age_bands',as_index=False).mean().sort_values('deposit',ascending=False)
#variablebalance değişkeni analizi
plt.figure(figsize=(12,8))
sns.distplot(X_train['balance'])
K_bal=kurtosis(X_train['balance'])
s_bal=skew(X_train['balance'])
print('k:',K_bal)
print('s:',s_bal)

#Bu değişkenin k ve s değerlerinin çok yüksek olduğunu görüyoruz. Log trans uygulayalım ve görelim.

for df in combine:
    df['bal_trns']=df['balance'].apply(np.cbrt)
    
sns.distplot(X_train['bal_trns'])

K_trns_bal=kurtosis(X_train['bal_trns'])
s_trns_bal=skew(X_train['bal_trns'])
print('k:',K_trns_bal)
print('s:',s_trns_bal)
#S ve k değerlerimizin önemli ölçüde azaldığını görüyoruz.

x=df[df['deposit']==1]['bal_trns']
y=df[df['deposit']==0]['bal_trns']
plt.figure(figsize=(12,8))
plt.hist(x,bins=10,alpha=0.5,label='deposited')
plt.hist(y,bins=10,alpha=0.5,label='did not deposit')
plt.legend()
X_train['bal_bands']=pd.cut(X_train['bal_trns'],bins=10)
X_train[['bal_bands','deposit']].groupby('bal_bands',as_index=False).mean().sort_values('bal_bands',ascending=False)
#'day' değişkeni analizi
sns.distplot(X_train['day'])
x=df[df['deposit']==1]['day']
y=df[df['deposit']==0]['day']
plt.figure(figsize=(12,8))
plt.hist(x,bins=31,alpha=0.5,label='deposited')
plt.hist(y,bins=31,alpha=0.5,label='did not deposit')
plt.legend()
#'duration' değişkeni analizi

sns.distplot(X_train['duration'])
K_dur=kurtosis(X_train['duration'])
s_dur=skew(X_train['duration'])
print('k:',K_dur)
print('s:',s_dur)
#Verilerin oldukça çarpık olduğunu görüyoruz. Cbrt dönüşümü uygulayalım.
for df in combine:
    df['dur_trns']=df['duration'].apply(np.cbrt)
    
sns.distplot(X_train['dur_trns'])

K_trns_dur=kurtosis(X_train['dur_trns'])
s_trns_dur=skew(X_train['dur_trns'])
print('k:',K_trns_dur)
print('s:',s_trns_dur)

#Müşterinin bir arama yapmadan önce bir deopsit yapıp yapmayacağını tahmin etmemiz gerektiğinden değişken duartion'u dahil etmeyeceğiz.
#'campaign' değişkeni analizi
sns.distplot(X_train['campaign'])
K_cam=kurtosis(X_train['campaign'])
s_cam=skew(X_train['campaign'])
print('k:',K_cam)
print('s:',s_cam)
for df in combine:
    df['cam_trns']=df['campaign'].apply(np.log)
    
sns.distplot(X_train['cam_trns'])

K_trns_cam=kurtosis(X_train['cam_trns'])
s_trns_cam=skew(X_train['cam_trns'])
print('k:',K_trns_cam)
print('s:',s_trns_cam)
x=df[df['deposit']==1]['cam_trns']
y=df[df['deposit']==0]['cam_trns']
plt.figure(figsize=(12,8))
plt.hist(x,bins=5,alpha=0.5,label='deposited')
plt.hist(y,bins=5,alpha=0.5,label='did not deposit')
plt.legend()
X_train['cam_bands']=pd.cut(X_train['cam_trns'],bins=10)
X_train[['cam_bands','deposit']].groupby('cam_bands',as_index=False).mean().sort_values('deposit',ascending=False)
#'pdays' değişken analizi
sns.distplot(X_train['pdays'])
K_pd=kurtosis(X_train['pdays'])
s_pd=skew(X_train['pdays'])
print('k:',K_pd)
print('s:',s_pd)
for df in combine:
    df['pd_trns']=df['pdays'].apply(np.cbrt)
    
sns.distplot(X_train['pd_trns'])

K_trns_pd=kurtosis(X_train['pd_trns'])
s_trns_pd=skew(X_train['pd_trns'])
print('k:',K_trns_pd)
print('s:',s_trns_pd)
x=df[df['deposit']==1]['pd_trns']
y=df[df['deposit']==0]['pd_trns']
plt.figure(figsize=(12,8))
plt.hist(x,bins=10,alpha=0.5,label='deposited')
plt.hist(y,bins=10,alpha=0.5,label='did not deposit')
plt.legend()
X_train['p_bands']=pd.cut(X_train['pd_trns'],bins=10)
X_train[['p_bands','deposit']].groupby('p_bands',as_index=False).mean().sort_values('deposit',ascending=False)
#Herhangi bir eğilim göremediğimiz için bu değişkeni şimdilik bırakıyoruz ve daha sonra onu çok değişkenli analizde kullanıp kullanamayacağımızı görüyoruz.
#'previous' değişken analizi
sns.distplot(X_train['previous'])
K_pre=kurtosis(X_train['previous'])
s_pre=skew(X_train['previous'])
print('k:',K_pre)
print('s:',s_pre)



for df in combine:
    df['pre_trns']=df['previous'].apply(np.cbrt)
    
sns.distplot(X_train['pre_trns'])

K_trns_pre=kurtosis(X_train['pre_trns'])
s_trns_pre=skew(X_train['pre_trns'])
print('k:',K_trns_pre)
print('s:',s_trns_pre)
X_train['pre_bands']=pd.cut(X_train['pre_trns'],bins=10)
X_train[['pre_bands','deposit']].groupby('pre_bands',as_index=False).mean().sort_values('deposit',ascending=False)
#önemli bir eğilim görülmedi, bu nedenle şimdilik bu parametreyi bırakalım
X_train.columns
X_train.head()
X_final_train=X_train[['job','loan','month','poutcome','age_trns','bal_trns','cam_trns','dur_trns']]
X_final_test=X_test[['job','loan','month','poutcome','age_trns','bal_trns','cam_trns','dur_trns']]
X_train.columns
from sklearn.tree import  DecisionTreeClassifier

dt=DecisionTreeClassifier(random_state=101)
dt.fit(X_final_train,y_train)
predict=dt.predict(X_final_test)
accuracy_test=round(dt.score(X_final_test,y_test)*100,2)
accuracy_train=round(dt.score(X_final_train,y_train)*100,2)

print('train accuracy of decision tree classifier',accuracy_train)
print('test accuracy of decision tree classifier',accuracy_test)
  
k_plot=[]
t_plot=[]

for k in range(1,10,1):
     dt=DecisionTreeClassifier(max_depth=k,random_state=101)
     dt.fit(X_final_train,y_train)
     predict=dt.predict(X_final_test)
     accuracy_test=round(dt.score(X_final_test,y_test)*100,2)
     accuracy_train=round(dt.score(X_final_train,y_train)*100,2)
     print(k)
     #print('train accuracy of decision tree classifier',accuracy_train)
     #print('test accuracy of decision tree classifier',accuracy_test)
     k_plot.append(accuracy_test)
     t_plot.append(accuracy_train)
fig,axes=plt.subplots(1,1,figsize=(12,8))

axes.set_xticks(range(1,10,1))
k=range(1,10,1)
plt.plot(k,k_plot)
plt.plot(k,t_plot,'r')
dt=DecisionTreeClassifier(max_depth=5,random_state=101)
dt.fit(X_final_train,y_train)
predict=dt.predict(X_final_test)
accuracy_test=round(dt.score(X_final_test,y_test)*100,2)
accuracy_train=round(dt.score(X_final_train,y_train)*100,2)

print('train accuracy of decision tree classifier',accuracy_train)
print('test accuracy of decision tree classifier',accuracy_test)