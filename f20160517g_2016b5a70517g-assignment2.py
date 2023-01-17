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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.metrics import f1_score
def process_input(train, test):

    X=train.drop('Class', axis=1)

    X=X.drop('ID', axis=1)

    y = train[['Class']]



    X_test=test.copy()

    X_test=X_test.drop('ID', axis=1)



    X=pd.get_dummies(X)

    X_test=pd.get_dummies(X_test)

    return X, y, X_test
train=pd.read_csv('/kaggle/input/data-mining-assignment-2/train.csv')

test=pd.read_csv('/kaggle/input/data-mining-assignment-2/test.csv')
X, y, X_test = process_input(train, test)
objcols=['col'+str(x) for x in [2, 11, 37, 44, 56]]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)
rfc = RandomForestClassifier(n_estimators = 2000)

rfc.fit(X_train, np.asarray(y_train).ravel())
def cols(features, tresh):

    cols_keep = []

    for i in range(len(X_train.columns)):

        if features[i]>=tresh:

            cols_keep.append(X_train.columns[i])

    return cols_keep
cols_keep=cols(rfc.feature_importances_, 0.018)

X_train_imp = X_train[cols_keep]

X_val_imp = X_train[cols_keep]

X_test_imp = X_test[cols_keep]

X_imp = X[cols_keep]
rfc = RandomForestClassifier(n_estimators=2000)

cv=cross_validate(rfc, X_imp, np.asarray(y).ravel(), cv=10, scoring='f1_micro', return_train_score=True)

print("Validation f1-score = ",np.mean(cv['test_score']))
rfc = RandomForestClassifier(n_estimators=2000)

rfc.fit(X_imp, np.asarray(y).ravel())

preds=rfc.predict(X_test_imp)
sub = pd.DataFrame({'ID':pd.read_csv('/kaggle/input/data-mining-assignment-2/test.csv')['ID'].values, 'Class':preds})

pd.DataFrame.to_csv(sub, 'final_submission1.csv')

print(np.unique(sub['Class'].values, return_counts=True))
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "results.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(sub)