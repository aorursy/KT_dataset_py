# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve

from sklearn.metrics import average_precision_score, f1_score,confusion_matrix

from sklearn import manifold

from xgboost.sklearn import XGBClassifier

from xgboost import plot_importance, to_graphviz

import lightgbm as lgb 

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import DBSCAN

from sklearn.linear_model import LogisticRegression

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

warnings.filterwarnings("ignore", category=RuntimeWarning)

warnings.filterwarnings("ignore", category=FutureWarning)

warnings.filterwarnings("ignore", category=UserWarning)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/PS_20174392719_1491204439457_log.csv')

#df = pd.read_csv('./input/PS_20174392719_1491204439457_log.csv.zip',compression='zip')



df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', \

                        'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})

df[:1]
df.describe()
df.groupby('type')['isFraud','isFlaggedFraud'].sum()
df.groupby('type')['amount'].mean()
%%time

# Start Feature extraction

# Merchant flag for source and dist

# Trans tye for fraud trans type

data = df.copy()

# add source and target type

data['OrigC']=data['nameOrig'].apply(lambda x: 1 if str(x).find('C')==0 else 0)

data['DestC']=data['nameDest'].apply(lambda x: 1 if str(x).find('C')==0 else 0)

data['TRANSFER']=data['type'].apply(lambda x: 1 if x=='TRANSFER' else 0)

data['CASH_OUT']=data['type'].apply(lambda x: 1 if x=='CASH_OUT' else 0)



#
# Trans Amount from Loan ? killed features from  Arjun Joshua

data['OrigAmntErr']=(abs(data.oldBalanceOrig-data.newBalanceOrig)-data.amount)*data['OrigC']

data['DestAmntErr']=(abs(data.oldBalanceDest-data.oldBalanceDest)-data.amount)*data['DestC']
def Fplot(DF,catlist):

    nf = DF.shape[1]

    frdf=DF[catlist][DF.isFraud == 1]

    cldf=DF[catlist][DF.isFraud == 0]

    sns.set(font_scale=1)

    for i, cn in enumerate(catlist):

        if cn != 'isFraud':

            plt.figure(figsize=(12,nf*8))

            gs = gridspec.GridSpec(nf*2, 1)

            ax = plt.subplot(gs[i])

            ax.set_title(str(cn)+'(Fraud-Red/Normal-Green)')

            sns.distplot(frdf[cn], bins=50, color='Red', hist=True)

            ax = plt.subplot(gs[i+1])

            sns.distplot(cldf[cn], bins=50, color='Green', hist=True)

            DF[[cn,'isFraud']].boxplot(by='isFraud', vert=False,figsize=(12,5))

            plt.show()

    
dt=data.sample(n=500000)

flist=['step', u'amount','oldBalanceOrig',

       'newBalanceOrig', 'oldBalanceDest', 'newBalanceDest',

       'isFlaggedFraud', 'OrigC', 'DestC', 'TRANSFER',

       'CASH_OUT', 'OrigAmntErr', 'DestAmntErr']

Fplot(dt,flist)
# drop list 

droplist=['isFlaggedFraud','OrigC','type','nameDest','nameOrig']
#print XGBoost result

def presult(clf,x_test,y_test, plotimp=1):

    y_prob=clf.predict_proba(x_test)

    y_pred=clf.predict(x_test)

    print ('AUPRC :', (average_precision_score(y_test, y_prob[:, 1])))

    print ('F1 - macro :',(f1_score(y_test,y_pred,average='macro')))

    print  ('Confusion_matrix : ')

    print (confusion_matrix(y_test,y_pred))

    sns.set(font_scale=1.5)

    #sns.heatmap(confusion_matrix(testY,y_pred), annot=True,annot_kws={"size": 15},fmt='10g')

    #plt.show()

    if plotimp==1:

        plot_importance(clf)

        plt.show()
#Train/Test Split
MLData=data.drop(labels=droplist,axis=1)

X=MLData.drop('isFraud',axis=1)

Y=MLData.isFraud

trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.3,random_state=42)
print ('trainX.step.min,max,count', trainX.step.min(),trainX.step.max(),trainX.step.count())

print ('testX.step.min,max,count',testX.step.min(),testX.step.max(),testX.step.count())

print ('y=1 total, train, test',sum(Y),sum(trainY),sum(testY))
trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.3,random_state=42, shuffle=False)
print ('trainX.step.min,max,count', trainX.step.min(),trainX.step.max(),trainX.step.count())

print ('testX.step.min,max,count',testX.step.min(),testX.step.max(),testX.step.count())

print ('y=1 total, train, test',sum(Y),sum(trainY),sum(testY))
%%time

#Base metrics 

weights = (Y == 0).sum() / (1.0 * (Y == 1).sum())

clf = XGBClassifier( scale_pos_weight = weights, n_jobs = 4, random_state=42)

clf.fit(trainX, trainY)

print ('Test')

presult(clf,testX,testY,1)
#add DestC to drop list

droplist=['isFlaggedFraud','OrigC','type','nameDest','nameOrig','DestC']
%%time

# Check series transaction for same client

dt=data.copy()

dt = dt.sort_values(['step'])

print (sum(dt.groupby('nameOrig')['step'].count()>1)) #9298 ±0.2% 

# Only ±10 000 transaction for same client. Impossible create client profile features :-( 

#=> We can drop Trans type WO Fraud case





# drop Trans type WO Fraud

MLData=data.copy()

#only TRANSFER and CASH_OUT

MLData=MLData.loc[(MLData.TRANSFER+MLData.CASH_OUT)>0]

MLData.drop(droplist,axis=1,inplace=True)

#add CASH_OUT to drop list

MLData=MLData.drop(['CASH_OUT'],axis=1)

X=MLData.drop('isFraud',axis=1)

Y=MLData.isFraud

trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.3, shuffle=False)
%%time

# metrics only for CASH OUT and transfer in train

weights = (Y == 0).sum() / (1.0 * (Y == 1).sum())

clf = XGBClassifier( scale_pos_weight = weights, n_jobs = 4, random_state=42)

clf.fit(trainX, trainY)

presult(clf,testX,testY,1)
# modify step to hours in day. 

X['step24']=X.step%24
X.step24.describe()
trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.3, shuffle=False)
%%time

# metrics step in hour(s)

weights = (Y == 0).sum() / (1.0 * (Y == 1).sum())

clf = XGBClassifier( scale_pos_weight = weights, n_jobs = 4, random_state=42)

clf.fit(trainX, trainY)

presult(clf,testX,testY,1)
#  drop DestAmntErr and update newBalanceDest

X.loc[(X.DestAmntErr != 0) & (X.newBalanceDest == 0),'newBalanceDest'] = -1

X=X.drop('DestAmntErr',axis=1)

MLData.loc[(MLData.DestAmntErr != 0) & (MLData.newBalanceDest == 0),'newBalanceDest'] = -1

MLData=MLData.drop('DestAmntErr',axis=1)
X.newBalanceDest.describe()
trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.3, shuffle=False)
%%time

# metrics step in hour(s)

weights = (Y == 0).sum() / (1.0 * (Y == 1).sum())

clf = XGBClassifier( scale_pos_weight = weights, n_jobs = 4, random_state=42)

clf.fit(trainX, trainY)

presult(clf,testX,testY,1)
%%time

#LGBMClassifier

# metrics for 0.3 test size

clf = lgb.LGBMClassifier(n_estimators=100,max_depth=3, n_jobs = 4, random_state=42)

clf.fit(trainX, trainY)

presult(clf,testX,testY,0)
trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.5, shuffle=False)
%%time

# metrics for 0.5 test size

weights = (Y == 0).sum() / (1.0 * (Y == 1).sum())

clf = XGBClassifier( scale_pos_weight = weights, n_jobs = 4, random_state=42)

clf.fit(trainX, trainY)

presult(clf,testX,testY,0)
#ok try next step
#in first scale data. Theory - result do not change for random forest (XGB)
%%time

scaler = StandardScaler()

scaler.fit(trainX)

# metrics for 0.5 test size

weights = (Y == 0).sum() / (1.0 * (Y == 1).sum())

clf = XGBClassifier( scale_pos_weight = weights, n_jobs = 4, random_state=42)

clf.fit(scaler.transform(trainX), trainY)

presult(clf,scaler.transform(testX),testY,0)
%%time

#LGBMClassifier

scaler = StandardScaler()

scaler.fit(trainX)

# metrics for 0.3 test size

clf = lgb.LGBMClassifier(n_estimators=100,max_depth=3, n_jobs = 4, random_state=42)

clf.fit(scaler.transform(trainX), trainY)

presult(clf,scaler.transform(testX),testY,0)
# in practice we give best result for same time
# nex step - add synt feature  
MLData.corr().isFraud
Ldrop=['oldBalanceDest','newBalanceDest']
#try add linear probalitics to feature

scaler = StandardScaler()

scaler.fit(X.drop(Ldrop,axis=1))

lr = LogisticRegression(class_weight='balanced', C=0.5, random_state=33)

lr.fit(scaler.transform(trainX.drop(Ldrop,axis=1)),trainY)

trainX1=pd.DataFrame(scaler.transform(trainX.drop(Ldrop,axis=1)))

trainX1['lsynt']=lr.predict_log_proba(trainX1).T[1].reshape(-1,1)

testX1=pd.DataFrame(scaler.transform(testX.drop(Ldrop,axis=1)))

testX1['lsynt']=lr.predict_log_proba(testX1).T[1].reshape(-1,1)

y_pred=lr.predict(scaler.transform(testX.drop(Ldrop,axis=1)))

print  ('Confusion_matrix : ')

print (confusion_matrix(testY,y_pred))
%%time

# metrics for lsynt feature and  0.5 test size

# Data already scaled

weights = (Y == 0).sum() / (1.0 * (Y == 1).sum())

clf = XGBClassifier( scale_pos_weight = weights, n_jobs = 4, random_state=42)

clf.fit(trainX1, trainY)

presult(clf,testX1,testY,1)
%%time

#LGBMClassifier

# metrics for 0.3 test size

clf = lgb.LGBMClassifier(n_estimators=100,max_depth=3, n_jobs = 4, random_state=42)

clf.fit(trainX1, trainY)

presult(clf,testX1,testY,0)
trainX, testX, trainY, testY = train_test_split(X, Y, test_size = 0.3, shuffle=False)
#try add linear probalitics to feature

scaler = StandardScaler()

scaler.fit(X.drop(Ldrop,axis=1))

lr = LogisticRegression(class_weight='balanced', C=0.5, random_state=33)

lr.fit(scaler.transform(trainX.drop(Ldrop,axis=1)),trainY)

trainX1=pd.DataFrame(scaler.transform(trainX.drop(Ldrop,axis=1)))

trainX1['lsynt']=lr.predict_log_proba(trainX1).T[1].reshape(-1,1)

testX1=pd.DataFrame(scaler.transform(testX.drop(Ldrop,axis=1)))

testX1['lsynt']=lr.predict_log_proba(testX1).T[1].reshape(-1,1)

y_pred=lr.predict(scaler.transform(testX.drop(Ldrop,axis=1)))

print  ('Confusion_matrix : ')

print (confusion_matrix(testY,y_pred))
%%time

# metrics for lsynt feature and  0.3 test size

# Data already scaled

weights = (Y == 0).sum() / (1.0 * (Y == 1).sum())

clf = XGBClassifier( scale_pos_weight = weights, n_jobs = 4, random_state=42)

clf.fit(trainX1, trainY)

presult(clf,testX1,testY,1)
# not so good. Try only Scaler
%%time

scaler = StandardScaler()

scaler.fit(trainX)

# metrics for 0.3 test size

weights = (Y == 0).sum() / (1.0 * (Y == 1).sum())

clf = XGBClassifier( scale_pos_weight = weights, n_jobs = 4, random_state=42)

clf.fit(scaler.transform(trainX), trainY)

presult(clf,scaler.transform(testX),testY,0)
%%time

#LGBMClassifier

scaler = StandardScaler()

scaler.fit(trainX)

# metrics for 0.3 test size

clf = lgb.LGBMClassifier(n_estimators=100,max_depth=3, n_jobs = 4, random_state=42)

clf.fit(scaler.transform(trainX), trainY)

presult(clf,scaler.transform(testX),testY,0)