# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.cluster import KMeans

from sklearn.cluster import AgglomerativeClustering

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

df = pd.read_csv('../input/eval-lab-3-f464/train.csv')
print(df['gender'].value_counts())
print(df['SeniorCitizen'].value_counts())
print(df['Married'].value_counts())
print(df['Children'].value_counts())
print(df['TVConnection'].value_counts())
print(df['Channel1'].value_counts())
print(df['Channel2'].value_counts())
print(df['Internet'].value_counts())
print(df['HighSpeed'].value_counts())
print(df['AddedServices'].value_counts())
print(df['Subscription'].value_counts())
print(df['PaymentMethod'].value_counts())
Internet_count = df['Internet'].value_counts()

sns.set(style="darkgrid")

sns.barplot(Internet_count.index, Internet_count.values, alpha=0.9)

plt.title('Frequency Distribution of Internet')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Internet', fontsize=12)

plt.show()
Subscription_count = df['Subscription'].value_counts()

sns.set(style="darkgrid")

sns.barplot(Subscription_count.index, Subscription_count.values, alpha=0.9)

plt.title('Frequency Distribution of Subscription')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Subscription', fontsize=12)

plt.show()
Subscription_count = df['Subscription'].value_counts()

sns.set(style="darkgrid")

sns.barplot(Subscription_count.index, Subscription_count.values, alpha=0.9)

plt.title('Frequency Distribution of Subscription')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Subscription', fontsize=12)

plt.show()
Satisfied_count = df['Satisfied'].value_counts()

sns.set(style="darkgrid")

sns.barplot(Satisfied_count.index, Satisfied_count.values, alpha=0.9)

plt.title('Frequency Distribution of Satisfied')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Satisfied', fontsize=12)

plt.show()
df1 = df.copy()
df1=pd.get_dummies(df1, columns=['gender','SeniorCitizen','Married','Children','TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','Internet','HighSpeed','AddedServices','Subscription','PaymentMethod'])
df1.head()
df1.info()
df2= df1.copy()
df1.columns =  ['custId','tenure','MonthlyCharges','TotalCharges','Satisfied','gender_Female','gender_Male','SeniorCitizen_0','SeniorCitizen_1','Married_No','Married_Yes','Children_No','Children_Yes','TVConnection_Cable','TVConnection_DTH','TVConnection_No','Channel1_No','Channel1_No_tv_connection','Channel1_Yes','Channel2_No','Channel2_No_tv_connection','Channel2_Yes','Channel3_No','Channel3_No_tv_connection','Channel3_Yes','Channel4_No','Channel4_No_tv_connection','Channel4_Yes','Channel5_No','Channel5_No_tv_connection','Channel5_Yes','Channel6_No','Channel6_No_tv_connection','Channel6_Yes','Internet_No','Internet_Yes','HighSpeed_No','HighSpeed_Nointernet','HighSpeed_Yes','AddedServices_No','AddedServices_Yes','Subscription_Annually','Subscription_Biannually','Subscription_Monthly','PaymentMethod_Banktransfer','PaymentMethod_Cash','PaymentMethod_Creditcard','PaymentMethod_NetBanking']
df1.gender_Female.value_counts()
# df1.info()
f, ax = plt.subplots(figsize=(40,40))

corr =df1.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10,as_cmap=True),

            square=True, ax=ax, annot=True)
feature_col = ['tenure','MonthlyCharges','gender_Female','gender_Male','SeniorCitizen_0','SeniorCitizen_1','Married_No','Married_Yes','Children_No','Children_Yes','TVConnection_Cable','TVConnection_DTH','TVConnection_No','Channel1_No','Channel1_No_tv_connection','Channel1_Yes','Channel2_No','Channel2_No_tv_connection','Channel2_Yes','Channel3_No','Channel3_No_tv_connection','Channel3_Yes','Channel4_No','Channel4_No_tv_connection','Channel4_Yes','Channel5_No','Channel5_No_tv_connection','Channel5_Yes','Channel6_No','Channel6_No_tv_connection','Channel6_Yes','Internet_No','Internet_Yes','HighSpeed_No','HighSpeed_Nointernet','HighSpeed_Yes','AddedServices_No','AddedServices_Yes','Subscription_Annually','Subscription_Biannually','Subscription_Monthly','PaymentMethod_Banktransfer','PaymentMethod_Cash','PaymentMethod_Creditcard','PaymentMethod_NetBanking']
feature_cols = ['tenure','MonthlyCharges','SeniorCitizen_0','SeniorCitizen_1','Married_No','Married_Yes','Children_No','Children_Yes','TVConnection_Cable','TVConnection_DTH','TVConnection_No','Channel1_No','Channel1_No_tv_connection','Channel1_Yes','Channel2_No','Channel2_No_tv_connection','Channel2_Yes','Channel3_No','Channel3_No_tv_connection','Channel3_Yes','Channel4_No','Channel4_No_tv_connection','Channel4_Yes','Channel5_No','Channel5_No_tv_connection','Channel5_Yes','Channel6_No','Channel6_No_tv_connection','Channel6_Yes','Internet_No','Internet_Yes','HighSpeed_No','HighSpeed_Nointernet','HighSpeed_Yes','AddedServices_No','AddedServices_Yes','Subscription_Annually','Subscription_Biannually','Subscription_Monthly','PaymentMethod_Banktransfer','PaymentMethod_Cash','PaymentMethod_Creditcard','PaymentMethod_NetBanking']
feature_cols1 =  ['tenure','MonthlyCharges','SeniorCitizen_0','SeniorCitizen_1','Married_No','Married_Yes','Children_No','Children_Yes','TVConnection_Cable','TVConnection_DTH','TVConnection_No','Channel1_No','Channel1_No_tv_connection','Channel1_Yes','Channel2_No','Channel2_No_tv_connection','Channel2_Yes','Channel3_No','Channel3_No_tv_connection','Channel3_Yes','Channel4_No','Channel4_No_tv_connection','Channel4_Yes','Channel5_No','Channel5_No_tv_connection','Channel5_Yes','Channel6_No','Channel6_No_tv_connection','Channel6_Yes','AddedServices_No','AddedServices_Yes','Subscription_Annually','Subscription_Biannually','Subscription_Monthly','PaymentMethod_Banktransfer','PaymentMethod_Cash','PaymentMethod_Creditcard','PaymentMethod_NetBanking']
feature_cols2 =  ['tenure','MonthlyCharges','SeniorCitizen_0','SeniorCitizen_1','Married_No','Married_Yes','Children_No','Children_Yes','TVConnection_Cable','TVConnection_DTH','TVConnection_No','Channel1_No','Channel1_No_tv_connection','Channel1_Yes','Channel2_No','Channel2_No_tv_connection','Channel2_Yes','Channel3_No','Channel3_No_tv_connection','Channel3_Yes','Channel4_No','Channel4_No_tv_connection','Channel4_Yes','Channel5_No','Channel5_No_tv_connection','Channel5_Yes','Channel6_No','Channel6_No_tv_connection','Channel6_Yes','AddedServices_No','AddedServices_Yes','Subscription_Biannually','Subscription_Monthly','PaymentMethod_Banktransfer','PaymentMethod_Creditcard','PaymentMethod_NetBanking']
feature_cols3 = ['tenure','MonthlyCharges','SeniorCitizen_0','SeniorCitizen_1','Married_No','Married_Yes','Children_No','Children_Yes','TVConnection_Cable','TVConnection_DTH','TVConnection_No','Channel1_No','Channel1_No_tv_connection','Channel2_No','Channel2_No_tv_connection','Channel3_No','Channel3_No_tv_connection','Channel4_No','Channel4_No_tv_connection','Channel5_No','Channel5_No_tv_connection','Channel6_No','Channel6_No_tv_connection','AddedServices_No','AddedServices_Yes','Subscription_Biannually','Subscription_Monthly','PaymentMethod_Banktransfer','PaymentMethod_Creditcard','PaymentMethod_NetBanking']
feature_cols4 = ['tenure','TVConnection_Cable','Channel5_No','Channel6_No','Subscription_Biannually','Subscription_Monthly','PaymentMethod_NetBanking']
X = df1[feature_cols4]

y = df1.Satisfied
x = df1.tenure

y = df1.Satisfied 
plt.scatter(x, y, s=np.pi*3, c=(0,0,0), alpha=0.5)

plt.title('Scatter plot')

plt.xlabel('x')

plt.ylabel('y')

plt.show()
x1 = df1.PaymentMethod_NetBanking

y1 = df1.Satisfied 
plt.scatter(x1, y1, s=np.pi*3, c=(0,0,0), alpha=0.5)

plt.title('Scatter plot')

plt.xlabel('x')

plt.ylabel('y')

plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()

# X_train = sc.fit_transform(X_train)

# X_test = sc.transform(X_test)
# kmeans =  KMeans(n_clusters=2, n_init=10, max_iter=60, random_state=42)# You want cluster the customer records into 2: Satisfied or Not Satisfied

# kmeans.fit(X_train)
# predkm = kmeans.predict(X_test)

# fpr, tpr, thresholds = metrics.roc_curve(y_test, predkm)

# auc(fpr, tpr)
# from sklearn.decomposition import PCA 

# pca = PCA(n_components = 2) 

# X_principal = pca.fit_transform(X_train) 

# X_principal = pd.DataFrame(X_principal) 

# X_principal.columns = ['P1', 'P2'] 
# plt.figure(figsize =(6, 6)) 

# plt.scatter(X_principal['P1'], X_principal['P2'], cmap ='rainbow') 

# plt.show() 
# from imblearn.over_sampling import SMOTE

# smote = SMOTE(kind='regular',k_neighbors=1, sampling_strategy='minority')

# x_sm1,y_sm1= smote.fit_resample(X_train,y_train)
# from imblearn.over_sampling import ADASYN

# adasyn= ADASYN(n_neighbors=1, sampling_strategy='minority')

# x_sm2,y_sm2= adasyn.fit_resample(X_train,y_train)
# alg2 = AgglomerativeClustering(n_clusters = 2)

# alg2.fit(x_sm2)
# alg1 = AgglomerativeClustering(n_clusters = 2)

# alg1.fit(x_sm1)
alg = AgglomerativeClustering(n_clusters = 2,linkage = 'complete')

alg.fit(X_train)
from sklearn import metrics

from sklearn.metrics import auc
predalg = alg.fit_predict(X_test)

fpr, tpr, thresholds = metrics.roc_curve(y_test, predalg)

auc(fpr, tpr)
# plt.figure(figsize =(6, 6)) 

# plt.scatter(X_principal['P1'], X_principal['P2'],  

#            c = alg.fit_predict(X_principal), cmap ='rainbow') 

# plt.show() 
# birch = Birch(n_clusters = 2)

# birch.fit(X_train)
# predb = birch.predict(X_test)

# fpr, tpr, thresholds = metrics.roc_curve(y_test, predb)

# auc(fpr, tpr)
# ms = MeanShift(bandwidth=2)

# ms.fit(X_train)
# predms = ms.predict(X_test)

# fpr, tpr, thresholds = metrics.roc_curve(y_test, predms)

# auc(fpr, tpr)
df_test = pd.read_csv('../input/eval-lab-3-f464/test.csv')
df1_test = df_test.copy()
df1_test=pd.get_dummies(df1_test, columns=['gender','SeniorCitizen','Married','Children','TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','Internet','HighSpeed','AddedServices','Subscription','PaymentMethod'])
# df1_test.info()
df1_test.columns = ['custId','tenure','MonthlyCharges','TotalCharges','gender_Female','gender_Male','SeniorCitizen_0','SeniorCitizen_1','Married_No','Married_Yes','Children_No','Children_Yes','TVConnection_Cable','TVConnection_DTH','TVConnection_No','Channel1_No','Channel1_No_tv_connection','Channel1_Yes','Channel2_No','Channel2_No_tv_connection','Channel2_Yes','Channel3_No','Channel3_No_tv_connection','Channel3_Yes','Channel4_No','Channel4_No_tv_connection','Channel4_Yes','Channel5_No','Channel5_No_tv_connection','Channel5_Yes','Channel6_No','Channel6_No_tv_connection','Channel6_Yes','Internet_No','Internet_Yes','HighSpeed_No','HighSpeed_Nointernet','HighSpeed_Yes','AddedServices_No','AddedServices_Yes','Subscription_Annually','Subscription_Biannually','Subscription_Monthly','PaymentMethod_Banktransfer','PaymentMethod_Cash','PaymentMethod_Creditcard','PaymentMethod_NetBanking']
test = df1_test[feature_cols4]
# test = sc.transform(test)
predictions = alg.fit_predict(test)
df1_test['Satisfied'] = predictions
df1_test.Satisfied
header = ['custId','Satisfied']

df1_test.to_csv('test_csv_alg_11.csv',columns = header, index=False)
alg1 = AgglomerativeClustering(n_clusters = 2)

alg1.fit(X_train)
predalg1 = alg1.fit_predict(X_test)

fpr, tpr, thresholds = metrics.roc_curve(y_test, predalg1)

auc(fpr, tpr)
df2_test = df_test.copy()
df2_test=pd.get_dummies(df2_test, columns=['gender','SeniorCitizen','Married','Children','TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','Internet','HighSpeed','AddedServices','Subscription','PaymentMethod'])
df2_test.columns = ['custId','tenure','MonthlyCharges','TotalCharges','gender_Female','gender_Male','SeniorCitizen_0','SeniorCitizen_1','Married_No','Married_Yes','Children_No','Children_Yes','TVConnection_Cable','TVConnection_DTH','TVConnection_No','Channel1_No','Channel1_No_tv_connection','Channel1_Yes','Channel2_No','Channel2_No_tv_connection','Channel2_Yes','Channel3_No','Channel3_No_tv_connection','Channel3_Yes','Channel4_No','Channel4_No_tv_connection','Channel4_Yes','Channel5_No','Channel5_No_tv_connection','Channel5_Yes','Channel6_No','Channel6_No_tv_connection','Channel6_Yes','Internet_No','Internet_Yes','HighSpeed_No','HighSpeed_Nointernet','HighSpeed_Yes','AddedServices_No','AddedServices_Yes','Subscription_Annually','Subscription_Biannually','Subscription_Monthly','PaymentMethod_Banktransfer','PaymentMethod_Cash','PaymentMethod_Creditcard','PaymentMethod_NetBanking']
test1 = df2_test[feature_cols4]
predictions1 = alg1.fit_predict(test1)
df2_test['Satisfied'] = predictions1
header = ['custId','Satisfied']

df2_test.to_csv('test_csv_alg_10.csv',columns = header, index=False)