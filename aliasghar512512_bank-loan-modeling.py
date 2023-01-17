#Campaign for selling personal loans.



#This case is about a bank (Thera Bank) which has a growing customer base. Majority of these customers are liability customers (depositors) with varying size of deposits. The number of customers who are also borrowers (asset customers) is quite small, and the bank is interested in expanding this base rapidly to bring in more loan business and in the process, earn more through the interest on loans. In particular, the management wants to explore ways of converting its liability customers to personal loan customers (while retaining them as depositors). A campaign that the bank ran last year for liability customers showed a healthy conversion rate of over 9% success. This has encouraged the retail marketing department to devise campaigns with better target marketing to increase the success ratio with minimal budget.



#The department wants to build a model that will help them identify the potential customers who have higher probability of purchasing the loan. This will increase the success ratio while at the same time reduce the cost of the campaign.



# The file Bank.xls contains data on 5000 customers. The data include customer demographic information (age, income, etc.), the customer's relationship with the bank (mortgage, securities account, etc.), and the customer response to the last personal loan campaign (Personal Loan). Among these 5000 customers, only 480 (= 9.6%) accepted the personal loan that was offered to them in the earlier campaign.



# Column descriptions



##	Data Description:

##

#### ID --->Customer ID

#### Age--->Customer's age in completed years

#### Experience	#years of professional experience

#### Income--->Annual income of the customer ($000)

#### ZIPCode--->Home Address ZIP code.

#### Family--->Family size of the customer

#### CCAvg--->Avg. spending on credit cards per month ($000)

#### Education--->Education Level. 1: Undergrad; 2: Graduate; 3: Advanced/Professional

#### Mortgage--->Value of house mortgage if any. ($000)

#### personal Loan--->Did this customer accept the personal loan offered in the last campaign?

#### securities Account--->Does the customer have a securities account with the bank?

#### CD Account--->Does the customer have a certificate of deposit (CD) account with the bank?

#### Online--->Does the customer use internet banking facilities?

#### CreditCard--->Does the customer use a credit card issued by UniversalBank?
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline 
df=pd.read_csv(r'../input/../input/Bank_Loan.csv')

df=df.drop(['ID','ZIP Code'],axis=1)

df.isnull().sum()

#df
df.describe().transpose()
sns.pairplot(df)#,diag_kind='kde',hue='Personal Loan')
df['Experience'] = df['Experience'].abs()
corr = df.corr()

sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})

plt.figure(figsize=(13,7))

# create a mask so we only see the correlation values once

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask, 1)] = True

a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')

rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)

roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
fig, ax = plt.subplots()

colors = {1:'red',2:'yellow',3:'green'}

ax.scatter(df['Experience'],df['Age'],c=df['Education'].apply(lambda x:colors[x]))

plt.xlabel('Experience')

plt.ylabel('Age')
sns.boxplot(x='Education',y='Income',hue='Personal Loan',data=df)
sns.boxplot(x="Education", y='Mortgage', hue="Personal Loan", data=df,color='yellow')
sns.countplot(x="Securities Account", data=df,hue="Personal Loan")
sns.countplot(x='Family',data=df,hue='Personal Loan')
sns.countplot(x='CD Account',data=df,hue='Personal Loan')
sns.distplot( df[df['Personal Loan'] == 0]['CCAvg'], color = 'r')

sns.distplot( df[df['Personal Loan'] == 1]['CCAvg'], color = 'b')

print('Credit card spending of Non-Loan customers: ',df[df['Personal Loan'] == 0]['CCAvg'].median()*1000)

print('Credit card spending of Loan customers    : ',df[df['Personal Loan']== 1]['CCAvg'].median()*1000)
# Separate the independent attributes store them in X array

# Store the target column i.e personal loan into Y array

X=df.drop(['Personal Loan','Experience'] ,axis=1) # since experience doesn't have so much impact on the personal loan

y=df[['Personal Loan']]



#X

#y



X = X.values   #logistic modeling algorithm requires feature array not dataframe

y = y.values
# Create the training and test data set in the ratio of 70:30 respectively

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=123)

y_train = np.ravel(y_train)   # to convert 1 d vector into 1 d array

#X_train.shape

#y_train.shape

#y_test.shape
lr=LinearRegression()

lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)

print(lr.intercept_)

print(lr.coef_)

lr.score(X_train ,y_train)
from sklearn.linear_model import LogisticRegression

X_train2, X_test2, y_train2, y_test2 = train_test_split(X,y, test_size=0.3,random_state=123)

#y_train2 = np.ravel(y_train2)

lg=LogisticRegression()

lg.fit(X_train2,y_train2)

y_pred2=lg.predict(X_test2)
print(classification_report(y_test2,y_pred2))

print(confusion_matrix(y_test2,y_pred2))

accuracy_score(y_test2,y_pred2)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X,y, test_size=0.3,random_state=123)

model = GaussianNB()

model.fit(X_train3, y_train3)



predictions=model.predict(X_test3)



#Assess the accuracy of the model on test data

print(metrics.confusion_matrix(y_test3,predictions))

expected = y_test3

predicted = model.predict(X_test3)

# summarize the fit of the model

print(metrics.classification_report(expected, predicted))

print(accuracy_score(expected, predicted))
X_train4, X_test4, y_train4, y_test4 = train_test_split(X,y, test_size=0.3,random_state=123)
KNN = KNeighborsClassifier(n_neighbors= 10 , weights = 'distance',metric='euclidean')

KNN.fit(X_train4, y_train4)
predicted_labels = KNN.predict(X_test4)

print(metrics.confusion_matrix(y_test4, predicted_labels))

print(metrics.classification_report(y_test4, predicted_labels))

print(accuracy_score(y_test4, predicted_labels))
print('Accuracy for linear model is :', 100*lr.score(X_train , y_train))

print('Accuracy for logistic model is :', 100*accuracy_score(y_test2 , y_pred2))

print('Accuracy for GaussianNB model is :', 100*accuracy_score(expected, predicted))

print('Accuracy for KNN model is :', 100*accuracy_score(y_test4, predicted_labels))