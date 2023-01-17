import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn import svm



from sklearn.model_selection import train_test_split

import seaborn as sns



%matplotlib inline
#load data

data = pd.read_csv("../input/creditcard.csv")

data.head()
data.tail()
data.groupby(("Class")).mean()
#correlation matrix

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(data.drop(['Amount','Time'],1).corr(), vmax=.8, square=True);
#correlation matrix for only Fraud

f, (ax1, ax2) = plt.subplots(1,2,figsize=(13, 5))

sns.heatmap(data.query("Class==1").drop(['Class','Time'],1).corr(), vmax=.8, square=True, ax=ax1)

ax1.set_title('Fraud')

sns.heatmap(data.query("Class==0").drop(['Class','Time'],1).corr(), vmax=.8, square=True, ax=ax2);

ax2.set_title('Legit')

plt.show()
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,3))

data.query("Class==1").hist(column='V6',bins=np.linspace(-10,10,20),ax=ax1,label='Fraud')

ax1.legend()

data.query("Class==0").hist(column='V6',bins=np.linspace(-10,10,20),ax=ax2,label='Legit')

plt.legend()

plt.show()

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,3))

data.query("Class==1").hist(column='V2',bins=np.linspace(-10,10,20),ax=ax1,label='Fraud')

ax1.legend()

data.query("Class==0").hist(column='V2',bins=np.linspace(-10,10,20),ax=ax2,label='Legit')

plt.legend()

plt.show()
bins=np.linspace(-10,50,40)

data.query("Class==1").hist(column='Amount',bins=bins)

data.query("Class==0").hist(column="Amount",bins=bins)
data.query("Class==1").hist(column="Time")#,bins=np.linspace(-10,10,20))

data.query("Class==0").hist(column="Time")#,bins=np.linspace(-10,10,20))
X_Legit = data.query("Class==0").drop(["Amount","Class","Time"],1)

y_Legit = data.query("Class==0")["Class"]



X_Fraud = data.query("Class==1").drop(["Amount","Class","Time"],1)

y_Fraud = data.query("Class==1")["Class"]



#split data into training and cv set

X_train, X_test, y_train, y_test = train_test_split(X_Legit, y_Legit, test_size=0.33, random_state=42)

print(len(X_test))

X_test = X_test.append(X_Fraud)

print(len(X_Fraud),'   ', len(X_test))

y_test = y_test.append(y_Fraud)

X_test.head()
data.plot.scatter("V21","V22",c="Class")
# Write my own Multivariate Gaussian outlier detection

X = X_train

m = len(X)

mu = 1./m * X.mean()

Sigma=0

for i in range(m):

    Sigma += np.outer((X.iloc[i]-mu) , (X.iloc[i]-mu))

Sigma*=1./m

Sig_inv = np.linalg.inv(Sigma)

Sig_det = np.linalg.det(Sigma)
np.matrix(Sigma).shape
# This function calculates the probability for a Gaussian distribution

def prob(x_example):

    n=len(Sigma)

    xminusmu = x_example - mu

    return 1./((2*np.pi)**(n/2.) * Sig_det**0.5) * np.exp(-0.5* xminusmu.dot(Sig_inv).dot(xminusmu))
Sigma.diagonal()
# Check out some resulting probablilities for Fraud examples

for i in range(10):

    print(prob(X_Fraud.iloc[i]))
# Check out some resulting probablilities for NON-Fraud examples

for i in range(10):

    print(prob(X_train.iloc[i]))
# Picking out 100 training examples to test how many are misclassified as false positive

ptrain_result = np.apply_along_axis(prob, 1, X_train.head(100))
sum(ptrain_result < 1e-13)
# Copying this to a variable with a new name because i am using the same 

# variable below with 'Amount' included as feature

pTest_result = np.apply_along_axis(prob, 1, X_test)

pTest_result_prev = np.copy(pTest_result)
epsilon = 1e-13

yTest_result_prev = (pTest_result_prev < epsilon)
tp = sum(yTest_result_prev  & y_test)

tn = sum((~ yTest_result_prev)  & (~ y_test))

fp = sum((yTest_result_prev)  & (~ y_test))

fn = sum((~ yTest_result_prev)  & ( y_test))



print("true_pos ",tp)

print("true_neg ",tn)

print("false_pos ",fp)

print("false_neg ",fn)



recall = tp / (tp + fn)

precision = tp / (tp + fp)

F1 = 2*recall*precision/(recall+precision)

print("recall=",recall,"\nprecision=",precision)

print("F1=",F1)
data["Amountresc"] = (data["Amount"])/data["Amount"].var()



X_Legit = data.query("Class==0").drop(["Amount","Class","Time"],1)

y_Legit = data.query("Class==0")["Class"]



X_Fraud = data.query("Class==1").drop(["Amount","Class","Time"],1)

y_Fraud = data.query("Class==1")["Class"]



#split data into training and cv set

X_train, X_test, y_train, y_test = train_test_split(X_Legit, y_Legit, test_size=0.33, random_state=42)

X_test = X_test.append(X_Fraud)

y_test = y_test.append(y_Fraud)

X_test.head()
# Use my outlier detection

X = X_train

m = len(X)

mu = 1./m * X.mean()

Sigma=0

for i in range(m):

    Sigma += np.outer((X.iloc[i]-mu) , (X.iloc[i]-mu))

Sigma*=1./m

Sig_inv = np.linalg.inv(Sigma)

Sig_det = np.linalg.det(Sigma)
pTest_result = np.apply_along_axis(prob, 1, X_test)
epsilon = 2e-11

yTest_result = (pTest_result < epsilon)



tp = sum(yTest_result  & y_test)

tn = sum((~ yTest_result)  & (~ y_test))

fp = sum((yTest_result)  & (~ y_test))

fn = sum((~ yTest_result)  & ( y_test))



print("true_pos ",tp)

print("true_neg ",tn)

print("false_pos ",fp)

print("false_neg ",fn)



recall = tp / (tp + fn)

precision = tp / (tp + fp)

F1 = 2*recall*precision/(recall+precision)

print("recall=",recall,"\nprecision=",precision)

print("F1=",F1)
len(X_train)
## Use only part of training set, otherwise it takes very long

Xsmall = X_train.head(20000)
# fit the model

clf = svm.OneClassSVM(kernel="rbf", nu=0.01, gamma=0.3)

clf.fit(Xsmall)
y_pred_test = clf.predict(X_test)

y_pred_test = np.array([y==-1 for y in y_pred_test])



tp = sum(y_pred_test  & y_test)

tn = sum((~ y_pred_test)  & (~ y_test))

fp = sum((y_pred_test)  & (~ y_test))

fn = sum((~ y_pred_test)  & ( y_test))





print("true_pos ",tp)

print("true_neg ",tn)

print("false_pos ",fp)

print("false_neg ",fn)



recall = tp / (tp + fn)

precision = tp / (tp + fp)

F1 = 2*recall*precision/(recall+precision)

print("recall=",recall,"\nprecision=",precision)

print("F1=",F1)
# ## Some results from different test runs

# # nu=0.01, gamma=0.3

# true_pos  478

# true_neg  55794

# false_pos  38030

# false_neg  14

# recall= 0.971544715447 

# precision= 0.0124130050899

# F1= 0.0245128205128



# # nu=0.05, gamma=0.3

# true_pos  478

# true_neg  55774

# false_pos  38050

# false_neg  14

# recall= 0.971544715447 

# precision= 0.0124065614618

# F1= 0.0245002562788



# # nu=0.05, gamma=0.2

# true_pos  463

# true_neg  68954

# false_pos  24870

# false_neg  29

# recall= 0.941056910569 

# precision= 0.0182765562705

# F1= 0.0358567279768



# # nu=0.5, gamma=0.5

# true_pos  487

# true_neg  36001

# false_pos  57823

# false_neg  5

# recall= 0.989837398374 

# precision= 0.00835191219345

# F1= 0.0165640624469



# # nu=0.5, gamma=0.1

# true_pos  478

# true_neg  47316

# false_pos  46508

# false_neg  14

# recall= 0.971544715447 

# precision= 0.0101732430937

# F1= 0.0201356417709
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)





clf = IsolationForest(max_samples=10, random_state=rng)

clf.fit(X_train.head(100000))

y_pred_train = clf.predict(X_train)

y_pred_test = clf.predict(X_test)

#y_pred_outliers = clf.predict(X_outliers)
y_pred_test = clf.predict(X_test)

y_pred_test = np.array([y==-1 for y in y_pred_test])



tp = sum(y_pred_test  & y_test)

tn = sum((~ y_pred_test)  & (~ y_test))

fp = sum((y_pred_test)  & (~ y_test))

fn = sum((~ y_pred_test)  & ( y_test))





print("true_pos ",tp)

print("true_neg ",tn)

print("false_pos ",fp)

print("false_neg ",fn)



recall = tp / (tp + fn)

precision = tp / (tp + fp)

F1 = 2*recall*precision/(recall+precision)

print("recall=",recall,"\nprecision=",precision)

print("F1=",F1)