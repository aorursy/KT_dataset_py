import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

%matplotlib inline
#download the data set from  
#https://www.kaggle.com/dalpozz/creditcardfraud/data
# load the data set
url = "../input/creditcard.csv"
data=pd.read_csv(url)
data.shape
data.head()
#how many transactions are fraud 
data["Class"].value_counts()
# statistics
data.describe()
# lets plot plain scatter plot considering Amount and Class
data.plot(kind='scatter', x='Amount', y='Class',title ='Amount verus Transactions type');
plt.show()
g=sns.FacetGrid(data, hue='Class', size=8)
plot=g.map(sns.distplot,"Amount").add_legend()
g = g.set(xlim=(0,3000))

#Divide the dataset according to the label FraudTransactions and Normal Transactions
# Fraud means Class=1 and Normal means status =0
fraud=data.loc[data["Class"]==1]
normal=data.loc[data["Class"]==0]
plt.figure(figsize=(10,5))
plt.subplot(121)
fraud.Amount.plot.hist(title="Histogram of Fraud transactions")
plt.subplot(122)
normal.Amount.plot.hist(title="Histogram of Normal transactions")
print("Summary Statistics of fraud transactions:")
fraud.describe().Amount
print("Summary Statistics of Normal transactions:")
normal.describe().Amount
# DataSet contains two days transactions. 
# Feature 'Time' contains the seconds elapsed between each transaction and the first 
# transaction in the dataset.let us convert time in seconds to hours of a day
dataSubset = data[['Time', 'Amount', 'Class']].copy()

# Get rid of $ and , in the SAL-RATE, then convert it to a float
def seconds_Hour_Coversion(seconds):
      hours = seconds/(60*60) ## for conversion of seconds to hours.
      if hours>24: 
    ## if it is more than 24 hours then divide it by 2 as max number of hours is 48.
        hours= hours/2 
        return int(hours)
      else:
        return int(hours)
# Save the result in a new column
dataSubset['Hours'] = dataSubset['Time'].apply(seconds_Hour_Coversion)
g=sns.FacetGrid(dataSubset, hue='Class', size=10)
plot=g.map(sns.distplot,"Hours").add_legend()
g = g.set(xlim=(0,24))
#Divide the data set according to the label FraudTransactions and Normal Transactions
# Fraud means Class=1 and Normal means status =0
frauddata=dataSubset.loc[data["Class"]==1]
normaldata=dataSubset.loc[data["Class"]==0]
frauddata.describe()
#let us plot a heat map for correlation of the features
sns.heatmap(data.corr())
## let us take 100 samples from the dataset using train_test_split without missing 
# class distrubution in the original dataset.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data.loc[:, data.columns != 'Class'],\
                data['Class'], test_size=0.00035, random_state=42)
sample = pd.concat([X_test, y_test], axis=1) 
sample.shape
# Computing the Similarity
similarity=cosine_similarity(sample)

# rename the index and name it as TransactionId
sample.index.name = 'TransactionId'
sample.head()
def printResult(transaction1,first10pairs,sample,dict):
     x=transaction1[1][30]
     y=transaction1[0]
     s1='For the transaction id = '+ '{:d}'.format(y) + ', and Class = ' + \
        '{0:.5g}'.format(x)
     print (s1 +"\n")   
     print ('Similar transactions are :'+ '\n')
     for k in first10pairs:
        printSimilarity(k,dict[k],sample)
     print ('--------------------------------------------------------'+"\n")   
            
def printSimilarity(transactionId,similarity,sample):
   
     for transaction in sample.iterrows(): 
            if transaction[0] == transactionId:
              x=transaction[1][30]
              s=similarity
              y=transactionId  
              print ("Class = " + '{0:.5g}'.format(x) + ", Similarity = "+\
                 '{:f}'.format(s)+ ", transactionId = "+'{:d}'.format(y)+"\n")
              
import operator
import itertools
i=-1;
dict={}
for transaction1 in sample.iterrows() :
        i=i+1
        j=0
        for transaction2 in sample.iterrows():
            if i is not j:
              dict[transaction2[0]] = similarity[i][j]
            j=j+1
                   
        if dict : 
            sorted_dict = sorted(dict, key=dict.__getitem__)[:10]
            printResult(transaction1,sorted_dict,sample,dict)
            dict.clear()    
        
