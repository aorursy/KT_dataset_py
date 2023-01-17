import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
from sklearn import preprocessing
from datetime import datetime
import os
import time
from datetime import datetime
import pandas as pd
import random
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, average_precision_score, precision_recall_curve
from imblearn.under_sampling import TomekLinks, NearMiss, RandomUnderSampler
from sklearn import tree
import sklearn.metrics as metric
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from tabulate import tabulate
from sklearn.model_selection import KFold
from sklearn import metrics
from warnings import simplefilter
from sklearn.metrics import roc_curve, roc_auc_score

df = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
df.head()
fraud = df[df["isFraud"] == 1]
nonfraud = df[df["isFraud"] == 0]
print("Average of frauds: \n",fraud.TransactionAmt.mean())
print("Average of nonfrauds: \n", nonfraud.TransactionAmt.mean())
print("Maximum frauds: \n",fraud.TransactionAmt.max())
print("Minimum frauds: \n",fraud.TransactionAmt.min())
print("Maximum nonfrauds: \n", nonfraud.TransactionAmt.max())
print("Minimum nonfrauds: \n", nonfraud.TransactionAmt.min())
catFeatures=['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain', 'M1',	'M2',	'M3',	'M4',	'M5',	'M6',	'M7',	'M8',	'M9', 'C1','V12','V13','V14','V15','V16']

numFeatures=list(df)
for x in catFeatures:
  numFeatures.remove(x)

numFeatures.remove('isFraud')
df['isFraud'] = df['isFraud'].astype(int)
for catFeatures_val in catFeatures:
    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    colors=['red', 'orange', 'yellow', 'green', 'blue', 'pink']
    (df[catFeatures_val].value_counts().head(6)).plot(kind='bar', title=catFeatures_val,  ax=ax[0], color=colors)
    ((fraud[catFeatures_val].value_counts()*100/df[catFeatures_val].count()).head(6)).plot(kind='barh', title='Percent of frauds', ax=ax[1],color=colors)

plt.show()
missings = df.isnull().sum()
all_data = np.product(df.shape)
all_missings = missings.sum()
print ("Percent of missings ",(all_missings/all_data) * 100)
('Percent of frauds:', (len(fraud)/len(df))*100)
legend = df['isFraud'].replace({0: "Nonfraud", 1: "Fraud"})
colors=['green', 'red']
(legend.value_counts().head(6)).plot(kind='barh', title=('Number of transactions'), color=colors,figsize=(20, 5), )
del legend
hours = df['TransactionDT'] / (3600) #Preparation of a variable showing hours based on variable TransactionDT
hours_ = np.floor(hours) % 24
df['hours'] = hours_

missings = [df[col].isnull().sum() / df.shape[0] for col in df.columns] #Delete columns with 50% of missings values
cols_to_out = [df.columns[i] for i in range(df.shape[1]) if missings[i] > 0.5]
df = df.drop(cols_to_out, axis=1)
identical = [c for c in df.columns if df[c].value_counts(normalize=True).values[0] > 0.9] #Delete columns with 90% the same values
throw_away = identical
throw_away.remove('isFraud')
df = df.drop(throw_away, axis=1)
df = df.drop(columns=['TransactionID', 'TransactionDT'])

pd.set_option('display.max_columns', None)
df.head(10)


print('Before transformations: ') #Reduction number of categories in P_emaildomain
print('Number of lvls P_emaildomain : ',len(df['P_emaildomain'].value_counts()))
df['P_emaildomain'].replace({'aim.com' :  'other' ,
 'anonymous.com' :  'anonymous' ,
 'aol.com' :  'aol' ,
 'att.net' :  'other' ,
 'bellsouth.net' :  'other' ,
 'cableone.net' :  'other' ,
 'centurylink.net' :  'other' ,
 'cfl.rr.com' :  'other' ,
 'charter.net' :  'other' ,
 'comcast.net' :  'other' ,
 'cox.net' :  'other' ,
 'earthlink.net' :  'other' ,
 'embarqmail.com' :  'other' ,
 'frontier.com' :  'frontier' ,
 'frontiernet.net' :  'frontier' ,
 'gmail' :  'gmail' ,
 'gmail.com' :  'gmail' ,
 'gmx.de' :  'other' ,
 'hotmail.co.uk' :  'hotmail' ,
 'hotmail.com' :  'hotmail' ,
 'hotmail.de' :  'hotmail' ,
 'hotmail.es' :  'hotmail' ,
 'hotmail.fr' :  'hotmail' ,
 'icloud.com' :  'other' ,
 'juno.com' :  'other' ,
 'live.com' :  'live' ,
 'live.com.mx' :  'live' ,
 'live.fr' :  'live' ,
 'mac.com' :  'other' ,
 'mail.com' :  'other' ,
 'me.com' :  'other' ,
 'msn.com' :  'msn' ,
 'netzero.com' :  'netzero' ,
 'netzero.net' :  'netzero' ,
 'optonline.net' :  'other' ,
 'outlook.com' :  'outlook' ,
 'outlook.es' :  'outlook' ,
 'prodigy.net.mx' :  'other' ,
 'protonmail.com' :  'other' ,
 'ptd.net' :  'other' ,
 'q.com' :  'other' ,
 'roadrunner.com' :  'other' ,
 'rocketmail.com' :  'other' ,
 'sbcglobal.net' :  'other' ,
 'sc.rr.com' :  'other' ,
 'servicios-ta.com' :  'other' ,
 'suddenlink.net' :  'other' ,
 'twc.com' :  'other' ,
 'verizon.net' :  'other' ,
 'web.de' :  'other' ,
 'windstream.net' :  'other' ,
 'yahoo.co.jp' :  'yahoo' ,
 'yahoo.co.uk' :  'yahoo' ,
 'yahoo.com' :  'yahoo' ,
 'yahoo.com.mx' :  'yahoo' ,
 'yahoo.de' :  'yahoo' ,
 'yahoo.es' :  'yahoo' ,
 'yahoo.fr' :  'yahoo' ,
 'ymail.com' :  'other' ,},inplace=True)
print('After transformations: ')
print('Number of lvls P_emaildomain : ',len(df['P_emaildomain'].value_counts()))

catFeatures=['ProductCD', 'card4', 'card6', 'P_emaildomain',	'M2',	'M3',	'M4','M6', 'C1','V12','V13','V15','V16']

numFeatures=list(df)
try:
    
    for x in catFeatures:
      numFeatures.remove(x)
except:
    pass

numFeatures.remove('isFraud')
df.fillna(-999, inplace=True)
for x in catFeatures:
    df[x] = df[x].astype(str)
mapy={}
for feature in catFeatures:
    le = preprocessing.LabelEncoder()
    df[feature] = le.fit_transform(df[feature])
    mapy[feature] = le
features = list(df)
features.remove('isFraud')
target='isFraud'
smote = SMOTE(n_jobs=-1, random_state=2020)
adas = ADASYN(random_state=2020)
ros = RandomOverSampler( random_state=2020)
tom = TomekLinks()
rus = RandomUnderSampler(random_state=2020)
nm=NearMiss()
def wrapper(nFolds = 5, randomState=2020, debug=False, features=features, df=df, sampling=False, sampler=False, *args, **kwargs):
    kf = KFold(n_splits=nFolds, shuffle=True, random_state=randomState)

    TestResults=[]
    TrainResults=[]
    predictions=[]
    indices = []
    for train, test in kf.split(df.index.values):
        clf = DecisionTreeClassifier(*args, **kwargs, random_state=randomState)
        if debug:
            print(clf)
        
        X_train, y_train = df.iloc[train][features], df.iloc[train][target] 
        X_test, y_test = df.iloc[test][features], df.iloc[test][target]

        if sampling:
          X_train, y_train = sampler.fit_sample(X_train, y_train)

        clf.fit(X_train, y_train)
        predsTrain = clf.predict_proba(X_train)[:,1]
        preds = clf.predict_proba(X_test)[:,1]
                              
        predictions.append(preds.tolist().copy())
        
        indices.append(df.iloc[test].index.tolist().copy())
        
        TrainScore = average_precision_score((y_train==1).astype(int), predsTrain)
        TestScore = average_precision_score((y_test==1).astype(int), preds)
        TrainResults.append(TrainScore)
        TestResults.append(TestScore)


    return  predictions, indices, TestResults, TrainResults

def plotAUPR(results):

	fig, ax = plt.subplots(figsize=(10,9))

	for true, pred, label in results:
		precision, recall, thresholds = precision_recall_curve(true, pred)
		average_precision = average_precision_score(true, pred)
		average_precision = round(average_precision, 4)
		lw=2
		ax.plot(recall, precision, lw=lw, label=f'{label}: {average_precision}')
  

	ax.set_xlim([0, 1])
	ax.set_ylim([0.0, 1.01])
	ax.set_xlabel('Recall')
	ax.set_ylabel('Precision')
	ax.set_title(f'Precision Recall Curve ')
	ax.legend(loc="lower right")
	plt.show()
rec2plot=[]

std=[]
valid_train=[]
valid_test=[]

predictions, indices, TestResults, TrainResults = wrapper(df=df,max_depth=21,
                                                                    min_samples_split=60, min_samples_leaf=50, max_features=37)
print( "std: ", np.std(predictions), 'train AUPR:', np.mean(TrainResults), 'test AUPR: ',np.mean(TestResults) )
std.append(np.std(predictions))
valid_train.append(np.mean(TrainResults))
valid_test.append(np.mean(TestResults))


true = (df[target]==1).astype(int).sort_index()
pred = pd.Series(sum(predictions, []), index=sum(indices, [])).sort_index()
rec2plot.append((true, pred, "DT"))



print('-----------------DT END')
n=''

for k in [rus,nm,tom,ros,smote,adas]:

  if k==rus:
    n='RandomUnderSampler'
  elif k==ros :
    n='RandomOverSampler'
  elif k==smote:
    n='SMOTE'
  elif k==nm:
    n='NearMiss'
  elif k==tom:
    n='TomekLinks'
  else:
    n='ADASYN'

  predictions, indices, TestResults, TrainResults = wrapper(df=df, max_depth=21,
                                                                    min_samples_split=60, min_samples_leaf=50, max_features=37, sampling=True, sampler=k)
  print( "std: ", np.std(predictions), 'train AUPR:', np.mean(TrainResults), 'test AUPR: ',np.mean(TestResults))
  valid_train.append(np.mean(TrainResults))
  valid_test.append(np.mean(TestResults))
  std.append(np.std(predictions))
  
  name = 'modelDT+'+  n


  true = (df[target]==1).astype(int).sort_index()
  pred = pd.Series(sum(predictions, []), index=sum(indices, [])).sort_index()
  rec2plot.append((true, pred, name))


  print('------------------------')

plotAUPR(rec2plot)

plotAUPR(rec2plot)