import numpy as np, pandas as pd, os

np.random.seed(300)



# GENERATE USELESS VARIABLES AND RANDOM TARGETS

train = pd.DataFrame(np.zeros((250,300)))

for i in range(300): train.iloc[:,i] = np.random.normal(0,1,250)

train['target'] = np.random.uniform(0,1,250)

train.loc[ train['target']>0.34, 'target'] = 1.0

train.loc[ train['target']<=0.34, 'target'] = 0.0
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt



oof = np.zeros(len(train))

rskf = RepeatedStratifiedKFold(n_splits=25, n_repeats=5)

for train_index, test_index in rskf.split(train.iloc[:,:-1], train['target']):

    clf = LogisticRegression(solver='liblinear',penalty='l1',C=0.1,class_weight='balanced')

    clf.fit(train.loc[train_index].iloc[:,:-1],train.loc[train_index]['target'])

    oof[test_index] += clf.predict_proba(train.loc[test_index].iloc[:,:-1])[:,1]

aucTR = round(roc_auc_score(train['target'],oof),5)

print('CV =',aucTR)
dfTR = pd.DataFrame({'var':np.arange(300),'CV':np.zeros(300)})

for i in range(300):

    logr = LogisticRegression(solver='liblinear').fit(train[[i]],train['target'])

    dfTR.loc[i,'CV'] = roc_auc_score(train['target'],logr.predict_proba(train[[i]])[:,1])

dfTR.sort_values('CV',inplace=True,ascending=False)

dfTR.head()
plt.hist(dfTR['CV'],bins=25)

plt.title('Histogram of CV of 300 useless variables')

plt.show()
plt.figure(figsize=(5,5))

plt.scatter(train[133],train[162],c=train['target'])

plt.plot([-2,2],[-2,2],':k')

plt.title('Among 300 simulated useless variables, we find these two!')

plt.xlabel('synthetic variable 133')

plt.ylabel('synthetic variable 162')

plt.show()
# GENERATE USELESS VARIABLES AND RANDOM TARGETS

public = pd.DataFrame(np.zeros((1975,300)))

for i in range(300): public.iloc[:,i] = np.random.normal(0,1,1975)

public['target'] = np.random.uniform(0,1,1975)

public.loc[ public['target']>0.34, 'target'] = 1.0

public.loc[ public['target']<=0.34, 'target'] = 0.0
oof = np.zeros(len(public))

rskf = RepeatedStratifiedKFold(n_splits=25, n_repeats=5)

for train_index, test_index in rskf.split(public.iloc[:,:-1], public['target']):

    clf = LogisticRegression(solver='liblinear',penalty='l1',C=0.1,class_weight='balanced')

    clf.fit(public.loc[train_index].iloc[:,:-1],public.loc[train_index]['target'])

    oof[test_index] += clf.predict_proba(public.loc[test_index].iloc[:,:-1])[:,1]

aucPU = round(roc_auc_score(public['target'],oof),5)

print('LB =',aucPU)
dfPU = pd.DataFrame({'var':np.arange(300),'LB':np.zeros(300)})

for i in range(300):

    logr = LogisticRegression(solver='liblinear').fit(public[[i]],public['target'])

    dfPU.loc[i,'LB'] = roc_auc_score(public['target'],logr.predict_proba(public[[i]])[:,1])

dfPU.sort_values('LB',inplace=True,ascending=False)

dfPU.head()
plt.hist(dfPU['LB'],bins=25)

plt.title('Histogram of LB of 300 useless variables')

plt.show()
auc = []

target = np.ones(250); target[:90] = 0.0

for i in range(10000):

    useless = np.random.normal(0,1,250)

    auc.append( roc_auc_score(target,useless) )

    #if i%1000==0: print(i)

z = 1.28 # 80% CE, 1.645 is 90% CE

low = round( 0.500 - z * np.std(auc),3)

high = round( 0.500 + z * np.std(auc),3)

print('80% of useless AUC are between',low,'and',high)

plt.hist(auc,bins=100); plt.show()
outliers = []

target = np.ones(250); target[:90] = 0.0

for i in range(1000):

    ct = 0

    for j in range(300):

        useless = np.random.normal(0,1,250)

        auc = roc_auc_score(target,useless)

        if (auc<low)|(auc>high): ct += 1

    outliers.append(ct)

    #if i%100==0: print(i)

plt.hist(outliers,bins=100); plt.show()

mn = np.mean(outliers); st = np.std(outliers)

lw = round(mn-z*st,1); hg = round(mn+z*st,1)

print('We are 80% confident that between',lw,'and',hg,

      'useless variables have AUC less than',low,'or greater than',high)
train = pd.read_csv('../input/train.csv')

ct = 0

for i in range(300):

    auc = roc_auc_score(train['target'],train[str(i)])

    if (auc<low)|(auc>high): ct += 1   

print('There are',ct,'real variables with AUC less than',low,'or greater than',high)

a = round(ct-hg,1); b = round(ct-lw,1)

print('Therefore we are 80% confident that between',a,'and',b,

      'real variables are useful with AUC less than',low,'or greater than',high)

print('Additionally there are possible useful real variables with weak AUC between',low,'and',high)