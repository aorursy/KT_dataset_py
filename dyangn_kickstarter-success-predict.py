# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import os

import seaborn as sns

from sklearn import metrics

import matplotlib.pyplot as plt
# load one dataset and preview

df = pd.read_csv("../input/ks-projects-201801.csv",parse_dates=True)

df.head(3)
# use seaborn to plot the pivot of states' counts

state_pv = pd.pivot_table(df,index=['state'],values=['ID'],aggfunc=len).sort_values(by='ID',ascending=False)

sns.barplot(x=state_pv.index,y='ID',data=state_pv)

plt.title("The total of each state categories")

plt.ylabel("counts")
# plot two columns: state and goal

sns.scatterplot(x='state',y='goal',data=df)

plt.title("The relationship between state and total goal")

plt.ticklabel_format(style='plain', axis='y')
# plot two columns: state and backers

sns.scatterplot(x='state',y='backers',data=df)

plt.title("The relationship between state and backers")

plt.ticklabel_format(style='plain', axis='y')
# plot two columns: state and usd pledged

sns.scatterplot(x='state',y='usd pledged',data=df)

plt.title("The relationship between state and usd pledged")

plt.ticklabel_format(style='plain', axis='y')
features = df[['deadline','goal','launched','state','backers','usd pledged']]
# in order to simplify the problem, we rule out 

# the state of undefined, suspended and live. 

# We also assume canceled as fail.

features = (

    features.dropna()

            .query('(state == "failed") or (state == "canceled") or (state == "successful")')

)

features.head()
features.info()
features['deadline'] = pd.to_datetime(features['deadline']) 

features['launched'] = pd.to_datetime(features['launched']) 

features['diff'] = features['deadline'] - features['launched']



# convert the time delta to numeric days

def tointerval(r):

    return pd.Timedelta(r).days





features['time'] = features['diff'].apply(tointerval)
# convert the state to numeric

def gen_result(r):

    if r == 'successful':

        return 1

    else:

        return 0



# make new column based on state condition

features['result'] = features['state'].apply(gen_result)
# plot two columns: state and time span

sns.scatterplot(x='state',y='time',data=features)

plt.title("The relationship between state and duration of projects")

plt.ticklabel_format(style='plain', axis='y')
model_val = features[['result','time','goal','backers','usd pledged']]

model_val.describe()
import sklearn as sk

from sklearn import preprocessing



# shuffle the sample

model_val = sk.utils.shuffle(model_val)



# assign X exculde the result, nparray

X = model_val.drop("result",axis=1).values

# assign y as column 'result'

y = model_val["result"].values



# normalize features

# y is 0,1s, not need to normalize

X = preprocessing.normalize(X)



# we take around 20% of samples as test dataset

test = 74000

X_trian = X[:-test]

y_trian = y[:-test]



X_test = X[-test:]

y_test = y[-test:]
from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=10)

neigh.fit(X_trian, y_trian) 
neigh.score(X_test,y_test)
# cross valide the model

from sklearn.model_selection import cross_val_score



score = cross_val_score(neigh,X_test,y_test,cv=5)

print(score.mean(),score.std() *2)

print(score)
y_predict = neigh.predict(X_test)
tn, fp, fn, tp = metrics.confusion_matrix(y_test,y_predict).ravel()

(tn, fp, fn, tp)
kneigbor_metrix = metrics.confusion_matrix(y_test,y_predict)

df_kn_metrix = pd.DataFrame(kneigbor_metrix,range(2),range(2))

sns.heatmap(df_kn_metrix,annot=True,fmt='g')


proba = neigh.predict_proba(X_test)

preds = proba[:,1]

fpr, tpr, _ = metrics.roc_curve(y_test,preds)

roc_auc = metrics.auc(fpr,tpr)



plt.title("Receiver Operating Characteristic for kneigbor model")

plt.plot(fpr,tpr,'b', label=f'AUC = {roc_auc:.2f}')

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
# TODO use the precision for ROC plot to consider if the result influenced by imbalance data

# metrics.precision_recall_curve(y_test,preds)

# TODO complete all evaluation

# TODO Optimation

# TODO predict 2016 data，evalutation

# TODO Save models