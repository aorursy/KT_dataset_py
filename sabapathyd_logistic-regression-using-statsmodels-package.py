import numpy as np

import pandas as pd

import statsmodels.api as sm

import seaborn as sns

import matplotlib.pyplot as plt

import pylab

import scipy.stats as stats

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve, auc

from sklearn.preprocessing import binarize

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier



%matplotlib inline
data = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

data.head(10)
data['quality'].value_counts()    #Finding the count of the response variable
data['Quality'] = data['quality'].apply(lambda x:'1' if x>6 else '0')     #Converting the response variable to Binary
data.drop(['quality'],axis=1,inplace=True)    # drop the previous column
data.head(10)
data.isna().sum()       #Checking for null values
data.describe()
plt.boxplot(data['total sulfur dioxide'])

plt.show()
Q1 = data.quantile(0.25)

Q3 = data.quantile(0.75)

IQR = Q3 - Q1

print(IQR)            #Finding Inter Quartile range
data = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]

data.reset_index(inplace=True,drop=True)          # Removing the outliers
data.Quality.value_counts()
147/(1047+147)
index = list(range(0,12))

columns = list(data.columns[0:11])

dictc = dict(zip(index,columns))

print(dictc)          # creating column names
x = data.drop(['Quality'],axis=1)

y = data['Quality']

scaler = StandardScaler()

scaler.fit(x)

x= scaler.transform(x)

x = pd.DataFrame(x)

x=x.rename(columns=dictc)



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

#len(x_test),len(x_train),len(y_train),len(y_test)

y_train = np.ravel(y_train)
logit1 = sm.Logit(y.astype(float),sm.add_constant(x.astype(float))).fit()

print(logit1.summary())
logit = LogisticRegression(solver='lbfgs')

logit.fit(x_train,y_train)

predict = logit.predict(x_test)

predictt = logit.predict(x_train)
prob_test = logit.predict_proba(x_test)[:,1]

#print(prob_test.reshape(1,-1))
prob_train = logit.predict_proba(x_train)[:,1]

#print(prob_train.reshape(1,-1))
accuracy_score(y_test,predict)
confusion_matrix(y_test,predict)
print(classification_report(y_test,predict))
data.drop(['fixed acidity'],axis=1,inplace=True)
data.drop(['residual sugar'],axis=1,inplace=True)
data.drop(['density'],axis=1,inplace=True)
data.drop(['pH'],axis=1,inplace=True)
data.drop(['chlorides'],axis=1,inplace=True)
data.drop(['citric acid'],axis=1,inplace=True)
data.drop(['free sulfur dioxide'],axis=1,inplace=True)
x = data.drop(['Quality'],axis=1)

y = data['Quality']

logit1 = sm.Logit(y.astype(float),sm.add_constant(x.astype(float))).fit()

print(logit1.summary())
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

len(x_test),len(x_train),len(y_train),len(y_test)
logit = LogisticRegression(solver='lbfgs')

logit.fit(x_train,y_train)

predict = logit.predict(x_test)

predictt = logit.predict(x_train)
prob_test = logit.predict_proba(x_test)[:,1]

#print(prob_test.reshape(1,-1))
prob_train = logit.predict_proba(x_train)[:,1]

#print(prob_train.reshape(1,-1))
print("accuracy:",accuracy_score(y_test,predict))
print("confusion matrix \n",confusion_matrix(y_test,predict))
print(classification_report(y_test,predict))
roc_auc_train = roc_auc_score(y_train,predictt)

fpr, tpr, threshold = roc_curve(pd.to_numeric(y_train),prob_train)

roc_auc = auc(fpr,tpr)
plt.figure()

plt.plot(fpr,tpr,color ='blue',label='ROC curve(area= %0.2f)'%(roc_auc))

plt.plot([0,1],[0,1],'r--')

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title("Receiver Operating Characteristic : Train Data")

plt.legend(loc='lower right')

plt.show()
roc_auc_test = roc_auc_score(y_test,predict)

fpr1, tpr1, threshold1 = roc_curve(pd.to_numeric(y_test),prob_test)

roc_auc1 = auc(fpr1,tpr1) 
plt.figure()

plt.plot(fpr1,tpr1,color ='blue',label='ROC curve(area= %0.2f)'%(roc_auc1))

plt.plot([0,1],[0,1],'r--')

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title("Receiver Operating Characteristic : Test data")

plt.legend(loc='lower right')

plt.show()
y_predict = binarize(prob_test.reshape(1,-1),0.25)[0]

y_predict = y_predict.astype(int)
confusion_matrix(pd.to_numeric(y_test),y_predict)
print(classification_report(pd.to_numeric(y_test),y_predict))
y_predict = binarize(prob_test.reshape(1,-1),0.50)[0]

y_predict = y_predict.astype(int)
confusion_matrix(pd.to_numeric(y_test),y_predict)
print(classification_report(pd.to_numeric(y_test),y_predict))
y_predict = binarize(prob_test.reshape(1,-1),0.75)[0]

y_predict = y_predict.astype(int)
confusion_matrix(pd.to_numeric(y_test),y_predict)
print(classification_report(pd.to_numeric(y_test),y_predict))
i = np.arange(len(tpr1))

roc = pd.DataFrame({'fpr':pd.Series(fpr1,index =i),'tpr':pd.Series(tpr1,index=i),

                   '1-fpr':pd.Series(1-fpr1,index=i),'tf':pd.Series(tpr1 - (1-fpr1),index=i),

                   'thresholds':pd.Series(threshold1,index=i)})

roc.iloc[(roc.tf-0).abs().argsort()[:1]]
plt.figure

plt.plot(roc['tpr'],color='red')

plt.plot(roc['1-fpr'],color='blue')

plt.xlabel('1-Flase Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic')

plt.show()
y_predict = binarize(prob_test.reshape(1,-1),0.16)[0]

y_predict = y_predict.astype(int)
print("accuracy:",accuracy_score(pd.to_numeric(y_test),pd.to_numeric(y_predict)))
print("confusion matrix:\n",confusion_matrix(pd.to_numeric(y_test),y_predict))
print(classification_report(pd.to_numeric(y_test),y_predict))