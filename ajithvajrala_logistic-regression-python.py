import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

import matplotlib.pyplot as plt 

plt.rc("font", size=14)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

import seaborn as sns

sns.set(style="white")

sns.set(style="whitegrid", color_codes=True)

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv("../input/bank-marketing-dataset/bank.csv")

#data = data.dropna()

print(data.shape)

print(list(data.columns))
data.head()
data['job'].value_counts()
data['education'].value_counts()
sns.countplot(x = 'deposit',data=data, palette='hls')

plt.show()
count_no_sub = len(data[data['deposit']=='no'])

count_sub = len(data[data['deposit']=='yes'])

pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)

print("percentage of no subscription is", pct_of_no_sub*100)

pct_of_sub = count_sub/(count_no_sub+count_sub)

print("percentage of subscription", pct_of_sub*100)
data.groupby('deposit').mean()
pd.crosstab(data.job,data.deposit).plot(kind='bar')

plt.title('Purchase Frequency for Job Title')

plt.xlabel('Job')

plt.ylabel('Frequency of Purchase')
table=pd.crosstab(data.marital,data.deposit)

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

plt.title('Stacked Bar Chart of Marital Status vs Purchase')

plt.xlabel('Marital Status')

plt.ylabel('Proportion of Customers')
table=pd.crosstab(data.education,data.deposit)

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

plt.title('Stacked Bar Chart of Education vs Purchase')

plt.xlabel('Education')

plt.ylabel('Proportion of Customers')
pd.crosstab(data.month,data.deposit).plot(kind='bar')

plt.title('Purchase Frequency for Job Title')

plt.xlabel('Job')

plt.ylabel('Frequency of Purchase')
data.age.hist()

plt.title('Histogram of Age')

plt.xlabel('Age')

plt.ylabel('Frequency')
pd.crosstab(data.poutcome,data.deposit).plot(kind='bar')

plt.title('Purchase Frequency for Poutcome')

plt.xlabel('Poutcome')

plt.ylabel('Frequency of Purchase')

plt.savefig('pur_fre_pout_bar')
data.head()
cat_vars=['job','marital','education','default','housing','loan','contact','month','poutcome']

for var in cat_vars:

    cat_list='var'+'_'+var

    cat_list = pd.get_dummies(data[var], prefix=var)

    data1=data.join(cat_list)

    data=data1

cat_vars=['job','marital','education','default','housing','loan','contact','month','poutcome']

data_vars=data.columns.values.tolist()

to_keep=[i for i in data_vars if i not in cat_vars]
data_final=data[to_keep]

data_final.columns.values
data_final_vars=data_final.columns.values.tolist()

y=['deposit']

X=[i for i in data_final_vars if i not in y]
y = data['deposit']

X = data_final[X]
y.value_counts()
X.shape, y.shape
y = y.apply(lambda x: 1 if x=="yes" else 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

rfe = RFE(logreg, 20)

rfe = rfe.fit(X_train, y_train)

print(rfe.support_)

print(rfe.ranking_)
#columns afetr elimination are

cols = X.columns[rfe.support_]
X=data_final[cols]
import statsmodels.api as sm

logit_model=sm.Logit(y,X)

result=logit_model.fit()

print(result.summary2())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import confusion_matrix

confusion_matrix = confusion_matrix(y_test, y_pred)

print(confusion_matrix)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()