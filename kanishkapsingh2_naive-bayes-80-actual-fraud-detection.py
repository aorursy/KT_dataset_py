import numpy as np # linear algebra

from math import log

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score
credit_data=pd.read_csv('../input/creditcard.csv')
credit_data.head()
credit_data['Time_in_hours']=credit_data['Time']/3600

credit_data['Log_Amount']=np.log(credit_data['Amount']+1)
sns.plt.hist(data=credit_data,x='Time_in_hours')
sns.FacetGrid(data=credit_data[credit_data['Class']==1],col='Class').map(sns.plt.hist,'Time_in_hours')
plt.figure(1)

plt.subplot(211)

sns.plt.hist(data=credit_data[credit_data['Class']==0],x='Amount',label='Normal Transactions')

plt.legend(loc='best')

plt.subplot(212)

sns.plt.hist(data=credit_data[credit_data['Class']==1],x='Amount',label='Fraud Transactions')

plt.legend(loc='best')
#Function to find in the correlation matrix the values which are of significant use for us.

def is_high(x):

    for i in range(len(x)):

        if (x.iloc[i]<0.5 and x.iloc[i]>-0.5):

            x.iloc[i]=0

    return x
a=credit_data.corr(method='spearman')

a=a.apply(is_high)

#To display the whole correlation matrix

pd.options.display.max_columns = 50

a
#To split the data into test and train dataset.

def split_data(dataset,ratio):

    sample=np.random.rand(len(dataset))<ratio

    return(dataset[sample],dataset[~sample])
col=list(credit_data.columns.values)
#Function to classify based on Naive Bayes. The algorithm runs 10 times and gives the mean of 

#predicted accuracy for each time.And it also tell which variable I removed from the total variable

#list so that I come to know which ones have to be removed.

def NB_Classify(ratio,drop_var):

    print('You dropped:',drop_var)

    #print (train.groupby('Class').count()['V1'])

    #print (test.groupby('Class').count()['V1'])

    pred_acc=[]

    for i in range(10):

        train,test=split_data(credit_data,ratio)

        clf=GaussianNB()

        clf.fit(train.drop(drop_var,axis=1),train['Class'])

        pred=clf.predict(test.drop(drop_var,axis=1))

        #print(pd.crosstab(test['Class'],pred))

        #print('You dropped:',drop_var)

        #print(accuracy_score(test['Class'],pred))

        pred_acc.append([pd.crosstab(test['Class'],pred).iloc[1,1]/(pd.crosstab(test['Class'],pred).iloc[1,0]+pd.crosstab(test['Class'],pred).iloc[1,1])])

    #' and got an accuracy of: ',np.mean(pred_acc)) 

    print(np.mean(pred_acc))
for var in col:

    NB_Classify(0.6,['Class','Log_Amount',var])
NB_Classify(0.6,['Class','Time','Log_Amount'])