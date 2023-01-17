import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import scipy
import sklearn
data = pd.read_csv('../input/creditcard.csv')
print(data.columns)
print(data.shape)
print(data.describe())
data.hist(figsize=(20,20))
plt.show()
data=data.sample(frac=0.7, random_state=1)

fraud=data[data['Class']==1]
valid=data[data['Class']==0]

outlier_frac=len(fraud)/float(len(valid))
print(outlier_frac)

print('Fraud cases: {}'.format(len(fraud)))
print('Valid cases: {}'.format(len(valid)))
columns=data.columns.tolist()
columns=[c for c in columns if c not in ['Class']]
target=['Class']

X=data[columns]
Y=data[target]

print(X.shape,Y.shape)
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


state=1

classifiers={
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                       contamination=outlier_frac,
                                       random_state=state,n_jobs=-1),
}
n_outliers=len(fraud)


for i,(clf_name,clf) in enumerate(classifiers.items()):
    if clf_name=='Isolation Forest':
        clf.fit(X)
        scores_pred=clf.decision_function(X)
        y_pred=clf.predict(X)

    y_pred[y_pred==1]=0
    y_pred[y_pred==-1]=1
#    n_errors=(y_pred != Y).sum()
#    print('{}:{}'.format(clf_name,n_errors))
    print(accuracy_score(Y,y_pred))
    print(classification_report(Y,y_pred))
