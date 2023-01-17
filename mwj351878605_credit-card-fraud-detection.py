import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import StratifiedShuffleSplit



%matplotlib inline
df = pd.read_csv('../input/creditcard.csv')
df.shape
df.info()
df.head()
df['Class'].value_counts()
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))



bins = 50



ax1.hist(df.Time[df['Class']==1],bins = bins)

ax1.set_title('Fraud')



ax2.hist(df.Time[df['Class']==0],bins = bins)

ax2.set_title('Normal')



plt.xlabel('Time')

plt.ylabel('Value Counts')

plt.show()
print('Fraud')

print(df['Amount'].loc[df['Class']==1].describe())

print('\n')

print('Normal')

print(df['Amount'].loc[df['Class']==0].describe())
#为了方便图表显示把Amount>=2200的Amount值设为2200

df.loc[df['Amount']>=2200,'Amount_change'] = 2200

df.loc[df['Amount']<2200,'Amount_change'] = df['Amount']



f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12,4))



bins = 30



ax1.hist(df.Amount_change[df.Class == 1], bins = bins)

ax1.set_title('Fraud')



ax2.hist(df.Amount_change[df.Class == 0], bins = bins)

ax2.set_title('Normal')



plt.xlabel('Amount')

plt.ylabel('Value Counts')

plt.show()
sess = StratifiedShuffleSplit(df['Class'].values,test_size=0.3)

for train_index,test_index in sess:

    trainData = df.iloc[train_index]

    testData = df.iloc[test_index]

X_train,y_train = trainData.loc[:,'V1':'V28'],trainData['Class']

X_test,y_test = testData.loc[:,'V1':'V28'],testData['Class']



ss = StandardScaler()

ss.fit(X_train)

X_train_ss = ss.transform(X_train)

X_test_ss = ss.transform(X_test)
number_records_fraud = len(trainData.loc[trainData['Class']==1])

fraud_index = np.array(trainData.loc[trainData['Class']==1].index)

normal_index = np.array(trainData.loc[trainData['Class']==0].index)



random_normal_index = np.random.choice(normal_index,number_records_fraud,replace=False)

random_normal_index = np.array(random_normal_index)



under_sample_index = np.concatenate([fraud_index,random_normal_index])

trainData_undersample = trainData.loc[under_sample_index,:]



X_train_undersample,y_train_undersample = trainData_undersample.loc[:,'V1':'V28'],trainData_undersample['Class']



#标准化

X_train_ss_undersample = ss.transform(X_train_undersample)

y_train_undersample.value_counts()
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import recall_score,confusion_matrix,roc_auc_score,precision_recall_curve



def choice_C(C):

    '''

    输入不同参数C，根据reecall值选出最优C值

    '''

    lr = LogisticRegression(C = C)

    lr.fit(X_train_ss_undersample,y_train_undersample)

    

    pred_lr = lr.predict(X_test_ss)

    

    cnf_matrix = confusion_matrix(y_test,pred_lr)

    #print(cnf_matrix)

    print("When C = {0},recall metric in the testing dataset: {1}".format(C,cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1])))

    

    return None
C = [0.001,0.01,0.1,1]

for i in C:

    choice_C(i)