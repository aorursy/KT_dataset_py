%reset -f

import numpy as np
import pandas as pd
import os, sys, time
import matplotlib as mtp  
import matplotlib.pyplot as plt  
import seaborn as sns

train = pd.read_csv("../input/train.csv");

test = pd.read_csv("../input/test.csv");

def cleanData(x) :
    mapper = {'yes':'1', 'no':'0'}
    x = x.fillna(0);
    x = x.replace({'dependency':mapper,'edjefa':mapper,'edjefe':mapper});
    x['dependency'] = x['dependency'].astype('float')
    x['edjefa'] = x['edjefa'].astype('float')
    x['edjefe'] = x['edjefe'].astype('float')
    return x

def dropColumns(x,y) :
    x = x.drop(y, axis=1);
    return x

train = cleanData(train)
test = cleanData(test)

column_list = pd.DataFrame({"col_name":train.columns})

cl = column_list[~column_list['col_name'].isin (['Id','idhogar','Target']) ]

trainHousehold = train.groupby(['idhogar'])[cl.col_name].sum()
trainHousehold['Target'] = train.groupby(['idhogar'])['Target'].max()
testHousehold = test.groupby(['idhogar'])[cl.col_name].sum()

trainHousehold = trainHousehold.reset_index()
testHousehold = testHousehold.reset_index()

testIds = test[['Id','idhogar']]
testHIds = testHousehold[['idhogar']]

train = dropColumns(train,['Id','idhogar'])
test = dropColumns(test,['Id','idhogar'])

trainHousehold = dropColumns(trainHousehold,['idhogar'])
testHousehold = dropColumns(testHousehold,['idhogar'])

trainFactors = train.drop(['Target'], axis=1)
trainResponse = train['Target']

trainHFactors = trainHousehold.drop(['Target'], axis=1)
trainHResponse = trainHousehold['Target']


from sklearn.cluster import KMeans
k_rng = range(2,15)
k_est = [KMeans(n_clusters = k).fit(trainFactors) for k in k_rng]
hk_est = [KMeans(n_clusters = k).fit(trainHFactors) for k in k_rng]

from sklearn import metrics

silhouette_score = [metrics.silhouette_score(trainFactors, e.labels_, metric='euclidean') for e in k_est]
hsilhouette_score = [metrics.silhouette_score(trainHFactors, e.labels_, metric='euclidean') for e in hk_est]

%matplotlib inline

import matplotlib.gridspec as gridspec
import matplotlib.gridspec as SubplotSpec
from matplotlib.figure import Figure

plt.figure()
plt.title('Silhouette coefficient for various values of k')
plt.plot(k_rng, silhouette_score, 'b*-')
plt.xlim([1,15])
plt.grid(True)
plt.ylabel('Silhouette Coefficient')
plt.show()

plt.figure()
plt.title('Silhouette coefficient for various values of k')
plt.plot(k_rng, hsilhouette_score, 'b*-')
plt.xlim([1,15])
plt.grid(True)
plt.ylabel('Silhouette Coefficient')
plt.show()


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(max_depth=4,random_state=42)
hgbc = GradientBoostingClassifier(max_depth=6,random_state=42)

gbc.fit(trainFactors,trainResponse)

hgbc.fit(trainHFactors,trainHResponse)

trainPredResponse = gbc.predict(trainFactors)
trainPredHResponse = hgbc.predict(trainHFactors)

pd.unique(trainPredResponse)
pd.unique(trainPredHResponse)

from sklearn.metrics import f1_score


for i in ['micro','macro','weighted'] :
    score = f1_score(trainResponse,trainPredResponse,average=i)
    print("f1 score {} is {} ".format(i, score))

for i in ['micro','macro','weighted'] :
    score = f1_score(trainHResponse,trainPredHResponse,average=i)
    print("f1 score {} is {} ".format(i, score))


testResponse = gbc.predict(test)

testHResponse = hgbc.predict(testHousehold)

idx_csv = pd.DataFrame({"Id":testIds['Id'],"idhogar":testIds['idhogar'],"Target_x":testResponse})

idxh_csv = pd.DataFrame({"idhogar":testHIds['idhogar'],"Target":testHResponse})

idx_csv_merged = pd.merge(idx_csv,idxh_csv,on='idhogar')

idx_csv = idx_csv_merged[['Id','Target']]
idx_csv.to_csv("predicted_test_3.csv",index=False)