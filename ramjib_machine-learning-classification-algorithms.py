import pandas as pd

import numpy as np



from scipy.stats import kurtosis,skew

import time



import seaborn as sns

import matplotlib.pyplot as plt

sns.set()



from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn import svm,tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score



import xgboost as xgb

import lightgbm as lgb
data = pd.read_csv('../input/gld-resnet-50-features/resnet_50_features.csv')

data.head()
plt.figure(figsize=(10,5))

ax = sns.barplot(x = data['label'].describe().index, y = data['label'].describe().values.astype(str))

for p in ax.patches:

    ht = p.get_height()

    ax.text(p.get_x()+p.get_width()//2,ht,'{:.1f}'.format(ht))

plt.title('Label Summary')

plt.show()
data['mapped_label'] = data['label'].astype('category')

data['mapped_label'] = data['mapped_label'].cat.codes

plt.figure(figsize=(20,5))

plt.subplot(121)

sns.distplot(data.label.astype(int), color='#2CDAE5')

plt.title('Original Label')



plt.subplot(122)

sns.distplot(data.mapped_label.astype(int),color='#F3322C')

plt.title('Mapped Label')

plt.show()
X = data.iloc[:,:-4]

Y = data['mapped_label']



print('X shape: {} \nY shape: {}'.format(X.shape, Y.shape))

x_train,x_valid,y_train,y_valid = train_test_split(X,Y, stratify=Y, test_size=0.2, random_state=2020)

print('x_train shape: {}\ny_train shape: {}\nx_valid shape: {}\ny_valid shape: {}'.format(x_train.shape,y_train.shape,x_valid.shape,y_valid.shape))

del data,X,Y
st = time.time()

nb = GaussianNB()

nb.fit(x_train,y_train)



y_pred = nb.predict(x_valid)



nb_as = round(accuracy_score(y_valid,y_pred),4)

nb_ps = round(precision_score(y_valid, y_pred,average='weighted'),4)

nb_rs = round(recall_score(y_valid,y_pred,average='weighted'),4)

nb_f1 = round(f1_score(y_valid,y_pred,average='weighted'),4)



print('Time Taken in seconds: {}\nAccuracy Score: {}\nPrecision Score: {}\nRecall Score: {}\nF1-Score: {}'.format(time.time()-st,nb_as,nb_ps,nb_rs,nb_f1))

st = time.time()

svc = svm.SVC()

svc.fit(x_train,y_train)



y_pred = svc.predict(x_valid)



svc_as = round(accuracy_score(y_valid,y_pred),4)

svc_ps = round(precision_score(y_valid, y_pred,average='weighted'),4)

svc_rs = round(recall_score(y_valid,y_pred,average='weighted'),4)

svc_f1 = round(f1_score(y_valid,y_pred,average='weighted'),4)



print('Time Taken in seconds: {}\nAccuracy Score: {}\nPrecision Score: {}\nRecall Score: {}\nF1-Score: {}'.format(time.time()-st,svc_as,svc_ps,svc_rs,svc_f1))
st = time.time()

dt = tree.DecisionTreeClassifier()

dt.fit(x_train,y_train)



y_pred = dt.predict(x_valid)



dt_as = round(accuracy_score(y_valid,y_pred),4)

dt_ps = round(precision_score(y_valid, y_pred,average='weighted'),4)

dt_rs = round(recall_score(y_valid,y_pred,average='weighted'),4)

dt_f1 = round(f1_score(y_valid,y_pred,average='weighted'),4)



print('Time Taken in seconds: {}\nAccuracy Score: {}\nPrecision Score: {}\nRecall Score: {}\nF1-Score: {}'.format(time.time()-st,dt_as,dt_ps,dt_rs,dt_f1))
st = time.time()

knn = KNeighborsClassifier()

knn.fit(x_train,y_train)



y_pred = knn.predict(x_valid)



knn_as = round(accuracy_score(y_valid,y_pred),4)

knn_ps = round(precision_score(y_valid, y_pred,average='weighted'),4)

knn_rs = round(recall_score(y_valid,y_pred,average='weighted'),4)

knn_f1 = round(f1_score(y_valid,y_pred,average='weighted'),4)



print('Time Taken in seconds: {}\nAccuracy Score: {}\nPrecision Score: {}\nRecall Score: {}\nF1-Score: {}'.format(time.time()-st,knn_as,knn_ps,knn_rs,knn_f1))
st = time.time()

rf = RandomForestClassifier()

rf.fit(x_train,y_train)



y_pred = rf.predict(x_valid)



rf_as = round(accuracy_score(y_valid,y_pred),4)

rf_ps = round(precision_score(y_valid, y_pred,average='weighted'),4)

rf_rs = round(recall_score(y_valid,y_pred,average='weighted'),4)

rf_f1 = round(f1_score(y_valid,y_pred,average='weighted'),4)



print('Time Taken in seconds: {}\nAccuracy Score: {}\nPrecision Score: {}\nRecall Score: {}\nF1-Score: {}'.format(time.time()-st,rf_as,rf_ps,rf_rs,rf_f1))
st = time.time()

ab = AdaBoostClassifier()

ab.fit(x_train,y_train)



y_pred = ab.predict(x_valid)



ab_as = round(accuracy_score(y_valid,y_pred),4)

ab_ps = round(precision_score(y_valid, y_pred,average='weighted'),4)

ab_rs = round(recall_score(y_valid,y_pred,average='weighted'),4)

ab_f1 = round(f1_score(y_valid,y_pred,average='weighted'),4)



print('Time Taken in seconds: {}\nAccuracy Score: {}\nPrecision Score: {}\nRecall Score: {}\nF1-Score: {}'.format(time.time()-st,ab_as,ab_ps,ab_rs,ab_f1))
st = time.time()

xg = xgb.XGBClassifier()

xg.fit(x_train,y_train)



y_pred = xg.predict(x_valid)



xg_as = round(accuracy_score(y_valid,y_pred),4)

xg_ps = round(precision_score(y_valid, y_pred,average='weighted'),4)

xg_rs = round(recall_score(y_valid,y_pred,average='weighted'),4)

xg_f1 = round(f1_score(y_valid,y_pred,average='weighted'),4)



print('Time Taken in seconds: {}\nAccuracy Score: {}\nPrecision Score: {}\nRecall Score: {}\nF1-Score: {}'.format(time.time()-st,xg_as,xg_ps,xg_rs,xg_f1))
st = time.time()

lg = lgb.LGBMClassifier()

lg.fit(x_train,y_train)



y_pred = lg.predict(x_valid)



lg_as = round(accuracy_score(y_valid,y_pred),4)

lg_ps = round(precision_score(y_valid, y_pred,average='weighted'),4)

lg_rs = round(recall_score(y_valid,y_pred,average='weighted'),4)

lg_f1 = round(f1_score(y_valid,y_pred,average='weighted'),4)



print('Time Taken in seconds: {}\nAccuracy Score: {}\nPrecision Score: {}\nRecall Score: {}\nF1-Score: {}'.format(time.time()-st,lg_as,lg_ps,lg_rs,lg_f1))
plt.figure(figsize=(20,8))

plt.subplot(221)

sns.lineplot(['Naive_Bayes', 'SVC', 'DT', 'KNN', 'RF', 'AB', 'XGB', 'LGBM'], [nb_as,svc_as,dt_as,knn_as,rf_as,ab_as,xg_as,lg_as],

        marker='*',label='Accuracy')

sns.lineplot(['Naive_Bayes', 'SVC', 'DT', 'KNN', 'RF', 'AB', 'XGB', 'LGBM'], [nb_ps,svc_ps,dt_ps,knn_ps,rf_ps,ab_ps,xg_ps,lg_ps],

        marker='*',label='Precision')

sns.lineplot(['Naive_Bayes', 'SVC', 'DT', 'KNN', 'RF', 'AB', 'XGB', 'LGBM'], [nb_rs,svc_rs,dt_rs,knn_rs,rf_rs,ab_rs,xg_rs,lg_rs],

        marker='*',label='Recall')

sns.lineplot(['Naive_Bayes', 'SVC', 'DT', 'KNN', 'RF', 'AB', 'XGB', 'LGBM'], [nb_f1,svc_f1,dt_f1,knn_f1,rf_f1,ab_f1,xg_f1,lg_f1],

        marker='*',label='F1-Score')

plt.legend()

plt.show()