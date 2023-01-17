import sklearn as sk

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

np.random.seed(42)
# data_orig = pd.read_csv("train.csv")

data_orig = pd.read_csv("../input/train.csv")



data = data_orig.copy()
X = data.copy().drop(columns=['Class'])

y = data.copy()['Class']
for i in X.columns:

    X[i] = X[i].replace(to_replace='?', value=np.nan)

to_remove = []

for i in X.columns:

    if(X[i].isna().sum() > 70000):

        to_remove.append(i)

X = X.drop(columns=to_remove)

X.drop(columns=['ID'], inplace=True)
numerical = []

for i in X.columns:

    if(X[i].dtype == 'object'):

        X[i].fillna(method='bfill', inplace=True)

        X[i].fillna(method='ffill', inplace=True)

    else:

        if(i in X.columns):

            numerical.append(i)

            X[i].fillna(value=data[i].mean(), inplace=True)
X = pd.get_dummies(X)

X.drop(columns=['Detailed_D32', 'Detailed_D38'], inplace=True)
from sklearn.ensemble import RandomForestClassifier
# for i in numerical:

#     X[i] = (X[i] - X[i].mean())/X[i].std()
rf = RandomForestClassifier(n_estimators=100, class_weight={0:1, 1:100000000}, warm_start=True)

rf.fit(X, y)

X_test = pd.read_csv('../input/test.csv')

X_test.drop(columns=to_remove, inplace=True)
for i in X_test.columns:

    X_test[i] = X_test[i].replace(to_replace='?', value=np.nan)



for i in X_test.columns:

    if(X_test[i].dtype == 'object'):

        X_test[i].fillna(method='bfill', inplace=True)

        X_test[i].fillna(method='ffill', inplace=True)

    else:

        if(i in X_test.columns):

            numerical.append(i)

            X_test[i].fillna(value=data[i].mean(), inplace=True)

X_test = pd.get_dummies(X_test)
[i for i in X_test.columns if i not in X.columns]
id = X_test['ID']

X_test.drop(columns=['Detailed_D36', 'ID'], inplace=True)

[i for i in X.columns if i not in X_test.columns]

# for i in numerical:

#     X_test[i] = (X_test[i] - X[i].mean())/X[i].std()
y_pred = rf.predict(X_test)

[i for i in X_test.columns if i not in X.columns]
y_pred.mean()
sub = pd.DataFrame(columns=['ID', 'Class'])
sub['ID'] = id

sub['Class'] = y_pred                   
sub.to_csv('submission.csv', index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html='<adownload="{filename}"href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(sub)