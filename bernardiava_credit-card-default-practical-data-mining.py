import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import itertools
cc = pd.read_csv("../input/UCI_Credit_Card.csv",skiprows=0)
cc.head(2)
cc.drop('ID',axis=1,inplace=True)
cc = cc.rename(columns={"default.payment.next.month": "dpnm"})
cc.head(2)
cc.info()
cc.describe()
cc['dpnm'].value_counts()
cc['dpnm'].value_counts(normalize=True)
plt.figure(figsize=(8,6)) 
corr_matrix = cc.corr()
sns.heatmap(corr_matrix);
plt.figure(figsize=(13,15))
plt.subplot(4,3,1)
sns.boxplot(x='dpnm',y='LIMIT_BAL',data=cc)
plt.subplot(4,3,2)
sns.boxplot(x='dpnm',y='SEX',data=cc)
plt.subplot(4,3,3)
sns.boxplot(x='dpnm',y='EDUCATION',data=cc)
plt.subplot(4,3,4)
sns.boxplot(x='dpnm',y='MARRIAGE',data=cc)
plt.subplot(4,3,5)
sns.boxplot(x='dpnm',y='AGE',data=cc)
plt.subplot(4,3,6)
sns.boxplot(x='dpnm',y='PAY_0',data=cc)
plt.subplot(4,3,7)
sns.boxplot(x='dpnm',y='PAY_2',data=cc)
plt.subplot(4,3,8)
sns.boxplot(x='dpnm',y='PAY_3',data=cc)
plt.subplot(4,3,9)
sns.boxplot(x='dpnm',y='PAY_4',data=cc)
plt.subplot(4,3,10)
sns.boxplot(x='dpnm',y='PAY_5',data=cc)
plt.subplot(4,3,11)
sns.boxplot(x='dpnm',y='PAY_6',data=cc)


plt.figure(figsize=(13,25))
plt.subplot(6,2,1)
sns.boxplot(x='dpnm',y='BILL_AMT1',data=cc)
plt.subplot(6,2,2)
sns.boxplot(x='dpnm',y='BILL_AMT2',data=cc)
plt.subplot(6,2,3)
sns.boxplot(x='dpnm',y='BILL_AMT3',data=cc)
plt.subplot(6,2,4)
sns.boxplot(x='dpnm',y='BILL_AMT4',data=cc)
plt.subplot(6,2,5)
sns.boxplot(x='dpnm',y='BILL_AMT5',data=cc)
plt.subplot(6,2,6)
sns.boxplot(x='dpnm',y='PAY_AMT1',data=cc)
plt.subplot(6,2,7)
sns.boxplot(x='dpnm',y='PAY_AMT2',data=cc)
plt.subplot(6,2,8)
sns.boxplot(x='dpnm',y='PAY_AMT3',data=cc)
plt.subplot(6,2,9)
sns.boxplot(x='dpnm',y='PAY_AMT4',data=cc)
plt.subplot(6,2,10)
sns.boxplot(x='dpnm',y='PAY_AMT5',data=cc)
plt.subplot(6,2,11)
sns.boxplot(x='dpnm',y='PAY_AMT6',data=cc)

plt.figure(figsize=(10,3))
sns.countplot(x='EDUCATION', hue='dpnm', data=cc)
# importing alll the necessary packages to use the various classification algorithms
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.model_selection import train_test_split #to split the dataset for training and testing
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn import svm  #for Support Vector Machine (SVM) Algorithm
from sklearn import metrics #for checking the model accuracy
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algorithm
import warnings
warnings.filterwarnings('ignore')
cc.shape
train_a,test_a=train_test_split(cc,test_size=0.3,random_state=0) 
train_b_a=train_a[['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0',
                   'PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1',
                   'BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1',
                   'PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']]
train_c_a=train_a.dpnm
test_b_a=test_a[['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0',
                 'PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1',
                 'BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1',
                 'PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']]
test_c_a=test_a.dpnm

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(train_a)
X_train_std = sc.transform(train_a)
X_test_std = sc.transform(test_a)

print('After standardizing our features, the first 5 rows of our data now look like this:\n')
print(pd.DataFrame(X_train_std, columns=cc.columns).head())
model=svm.SVC()
model.fit(train_b_a,train_c_a) 
prediction=model.predict(test_b_a) 
print('The accuracy of the SVM using all variables is:',metrics.accuracy_score(prediction,test_c_a))
model = LogisticRegression()
model.fit(train_b_a,train_c_a) 
prediction=model.predict(test_b_a) 
print('The accuracy of the Logistic Regression using all variables is:',metrics.accuracy_score(prediction,test_c_a))
model=DecisionTreeClassifier()
model.fit(train_b_a,train_c_a) 
prediction=model.predict(test_b_a) 
print('The accuracy of the Decision Tree using all variables is:',metrics.accuracy_score(prediction,test_c_a))
model=KNeighborsClassifier(n_neighbors=3) 
model.fit(train_b_a,train_c_a) 
prediction=model.predict(test_b_a) 
print('The accuracy of the KNN using all variables is:',metrics.accuracy_score(prediction,test_c_a))
model=svm.LinearSVC()
model.fit(train_b_a,train_c_a) 
prediction=model.predict(test_b_a) 
print('The accuracy of the Linear SVC using all variables is:',metrics.accuracy_score(prediction,test_c_a))
ccz=cc[['EDUCATION','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','dpnm']]
train_d,test_d=train_test_split(ccz,test_size=0.3,random_state=0) 
train_e_d=train_d[['EDUCATION','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']]
train_f_d=train_d.dpnm
test_e_d=test_d[['EDUCATION','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']]
test_f_d=test_d.dpnm
model=svm.SVC()
model.fit(train_e_d,train_f_d) 
prediction=model.predict(test_e_d) 
print('The accuracy of the SVM using Education, Age, and Payment History is:',metrics.accuracy_score(prediction,test_f_d))
from sklearn.decomposition import PCA
from pylab import plot,show
from numpy import genfromtxt, zeros
target = genfromtxt('../input/UCI_Credit_Card.csv',delimiter=',',usecols=(24),skip_header=1,dtype=int)
pca = PCA(n_components=2)
pcad = pca.fit_transform(cc[cc.columns[:23]])
plot(pcad[target==0,0],pcad[target==0,1],'bo')
plot(pcad[target==1,0],pcad[target==1,1],'ro')
show()
print(pca.explained_variance_ratio_)
cc.shape
print (1-sum(pca.explained_variance_ratio_))
for i in range(1,24):
    pca = PCA(n_components=i)
    pca.fit(cc)
    print (sum(pca.explained_variance_ratio_) * 100,'%')