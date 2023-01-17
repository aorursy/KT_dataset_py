import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec



#for data preprocessing

from sklearn.decomposition import PCA



#for modeling

from sklearn.neighbors import LocalOutlierFactor

from sklearn.ensemble import IsolationForest



#filter warnings

import warnings

warnings.filterwarnings("ignore")



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/creditcard.csv")

df.head()
sns.countplot(df.Class)

plt.show()

print(df.Class.value_counts())
timedelta = pd.to_timedelta(df['Time'], unit='s')

df['Time_hour'] = (timedelta.dt.components.hours).astype(int)



plt.figure(figsize=(12,5))

sns.distplot(df[df['Class'] == 0]["Time_hour"], color='g')

sns.distplot(df[df['Class'] == 1]["Time_hour"], color='r')

plt.title('Fraud and Normal Transactions by Hours', fontsize=17)

plt.xlim([-1,25])

plt.show()
cols= df[['Time', 'Amount']]



pca = PCA()

pca.fit(cols)

X_PCA = pca.transform(cols)



df['V29']=X_PCA[:,0]

df['V30']=X_PCA[:,1]



df.drop(['Time','Time_hour', 'Amount'], axis=1, inplace=True)



df.columns
columns = df.drop('Class', axis=1).columns

grid = gridspec.GridSpec(6, 5)



plt.figure(figsize=(20,10*2))



for n, col in enumerate(df[columns]):

    ax = plt.subplot(grid[n])

    sns.distplot(df[df.Class==1][col], bins = 50, color='g')

    sns.distplot(df[df.Class==0][col], bins = 50, color='r') 

    ax.set_ylabel('Density')

    ax.set_title(str(col))

    ax.set_xlabel('')

    

plt.show()
def ztest(feature):

    

    mean = normal[feature].mean()

    std = fraud[feature].std()

    zScore = (fraud[feature].mean() - mean) / (std/np.sqrt(sample_size))

    

    return zScore
columns= df.drop('Class', axis=1).columns

normal= df[df.Class==0]

fraud= df[df.Class==1]

sample_size=len(fraud)

significant_features=[]

critical_value=2.58



for i in columns:

    

    z_vavlue=ztest(i)

    

    if( abs(z_vavlue) >= critical_value):    

        print(i," is statistically significant") #Reject Null hypothesis. i.e. H0

        significant_features.append(i)
significant_features.append('Class')

df= df[significant_features]



inliers = df[df.Class==0]

ins = inliers.drop(['Class'], axis=1)



outliers = df[df.Class==1]

outs = outliers.drop(['Class'], axis=1)



ins.shape, outs.shape
def normal_accuracy(values):

    

    tp=list(values).count(1)

    total=values.shape[0]

    accuracy=np.round(tp/total,4)

    

    return accuracy



def fraud_accuracy(values):

    

    tn=list(values).count(-1)

    total=values.shape[0]

    accuracy=np.round(tn/total,4)

    

    return accuracy
state= 42



ISF = IsolationForest(random_state=state)

ISF.fit(ins)



normal_isf = ISF.predict(ins)

fraud_isf = ISF.predict(outs)



in_accuracy_isf=normal_accuracy(normal_isf)

out_accuracy_isf=fraud_accuracy(fraud_isf)

print("Accuracy in Detecting Normal Cases:", in_accuracy_isf)

print("Accuracy in Detecting Fraud Cases:", out_accuracy_isf)
LOF = LocalOutlierFactor(novelty=True)

LOF.fit(ins)



normal_lof = LOF.predict(ins)

fraud_lof = LOF.predict(outs)



in_accuracy_lof=normal_accuracy(normal_lof)

out_accuracy_lof=fraud_accuracy(fraud_lof)

print("Accuracy in Detecting Normal Cases:", in_accuracy_lof)

print("Accuracy in Detecting Fraud Cases:", out_accuracy_lof)
fig, (ax1,ax2)= plt.subplots(1,2, figsize=[15,2])



ax1.set_title("Accuracy of Isolation Forest",fontsize=20)

sns.barplot(x=[in_accuracy_isf,out_accuracy_isf], 

            y=['normal', 'fraud'],

            label="classifiers", 

            color="b", 

            ax=ax1)

ax1.set(xlim=(0,1))



ax2.set_title("Accuracy of Local Outlier Factor",fontsize=20)

sns.barplot(x=[in_accuracy_lof,out_accuracy_lof], 

            y=['normal', 'fraud'], 

            label="classifiers", 

            color="r", 

            ax=ax2)

ax2.set(xlim=(0,1))

plt.show()