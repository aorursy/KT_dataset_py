%reset -f

import numpy as np
import pandas as pd
import os, sys, time
import matplotlib.pyplot as plt 
from sklearn.metrics import f1_score
from xgboost.sklearn import XGBClassifier

train = pd.read_csv("../input/train.csv");
test = pd.read_csv("../input/test.csv");
idx = test[['Id']]
def cleanData(x) :
    mapper = {'yes':'1', 'no':'0'}
    x = x.drop(['Id','idhogar'], axis=1);
    x = x.fillna(0);
    x = x.replace({'dependency':mapper,'edjefa':mapper,'edjefe':mapper});
    x['dependency'] = x['dependency'].astype('float')
    x['edjefa'] = x['edjefa'].astype('float')
    x['edjefe'] = x['edjefe'].astype('float')
    return x

train = cleanData(train)
test = cleanData(test)
trainFactors = train.drop(['Target'], axis=1)
trainResponse = train['Target']
DISPLAY_MAX_ROWS = 100  # number of max rows to print for a DataFrame
pd.set_option('display.max_rows', DISPLAY_MAX_ROWS)
from sklearn.cluster import KMeans
k_rng = range(2,15)
k_est = [KMeans(n_clusters = k).fit(trainFactors) for k in k_rng]
from sklearn import metrics
silhouette_score = [metrics.silhouette_score(trainFactors, e.labels_, metric='euclidean') for e in k_est]
plt.figure()
plt.title('Silhouette coefficient for various values of k')
plt.plot(k_rng, silhouette_score, 'b*-')
plt.xlim([1,15])
plt.grid(True)
plt.ylabel('Silhouette Coefficient')
plt.show()
feature_cols = trainFactors.columns
dtc = XGBClassifier(max_depth=4,random_state=42 )
dtc.fit(trainFactors, trainResponse)
feature_importance = pd.DataFrame({'feature':feature_cols, 'importance':dtc.feature_importances_})
feature_importance[feature_importance.importance>0]
trainPredResponse = dtc.predict(trainFactors)
pd.unique(trainPredResponse)
for i in ['micro','macro','weighted'] :
    score = f1_score(trainResponse,trainPredResponse,average=i)
    print("f1 score {} is {} ".format(i, score))
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(max_depth=4,random_state=42,n_estimators=100)
gbc.fit(trainFactors,trainResponse)
trainPredResponse = gbc.predict(trainFactors)
pd.unique(trainPredResponse)
from sklearn.metrics import f1_score

for i in ['micro','macro','weighted'] :
    score = f1_score(trainResponse,trainPredResponse,average=i)
    print("f1 score {} is {} ".format(i, score))   
testResponse = gbc.predict(test)
pd.unique(testResponse)
idx_csv = pd.DataFrame({"Id":idx['Id'],"Target":testResponse})
idx_csv.head()
idx_csv.to_csv("predicted_test_2.csv",index=False)