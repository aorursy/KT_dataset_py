# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

data=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
print(data.columns)
print(data.describe())
data.hist(figsize=(20,20))

plt.show()
fraud=data[data['Class']==1]

valid=data[data['Class']==0]



fraud

valid
outlier=len(fraud)/float(len(valid))

outlier
corrleation_mat=data.corr()

fig=plt.figure(figsize=(12,9))

sns.heatmap(corrleation_mat,vmax=.8,square=True)

plt.show()
columns=data.columns.tolist()

columns=[c for c in columns if c not in ["Class"]]

target="Class"

x=data[columns]

y=data[target]

print(x.shape)

print(y.shape)

from sklearn.metrics import classification_report,accuracy_score

from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor



state=1

classifiers={

    "Isolation Forest":IsolationForest(max_samples=len(x),contamination=outlier,random_state=state),

    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20,contamination=outlier)

}
plt.figure(figsize=(9,7))

n_outliers=len(fraud)

for i,(clf_name,clf) in enumerate(classifiers.items()):

    if clf_name=="Local Outlier Factor":

        yhat=clf.fit_predict(x)

        scores_pred=clf.negative_outlier_factor_

    else:

        clf.fit(x)

        scores_pred=clf.decision_function(x)

        yhat=clf.predict(x)

    yhat[yhat==1]=0

    yhat[yhat==-1]=1

    n_errrors=(yhat!=y).sum()

    print(accuracy_score(y,yhat))

    print(classification_report(y,yhat))