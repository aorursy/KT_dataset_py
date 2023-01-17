import pandas as pd

import numpy as np
df_train = pd.read_csv('../input/Train.csv')

df_test = pd.read_csv('../input/Test.csv')

df_train =  df_train.drop("Currency",axis=1)
df_train_dropna = df_train.dropna()

df_test_dropna = df_test.dropna()
for name in df_train_dropna.columns:

    if 'Id' in name:

        k = name.find('Id')

        df_train_dropna[name[:4]+'_id'] = df_train_dropna[name].map(lambda s: s[k+3:]).astype(int)
for name in df_test_dropna.columns:

    if 'Id' in name:

        k = name.find('Id')

        df_test_dropna[name[:4]+'_id'] = df_test_dropna[name].map(lambda s: s[k+3:]).astype(int)
df_train_dropna1 = df_train_dropna.drop([name for name in df_train_dropna.columns if 'Id' in name], axis=1)
df_test_dropna1 = df_test_dropna.drop([name for name in df_test_dropna.columns if 'Id' in name], axis=1)
df_train_dropna2 = pd.get_dummies(df_train_dropna1, columns=['CurrencyCode'])

#df_train_dropna4 = pd.get_dummies(df_train_dropna2, columns=['ProductCategory'])

df_train_dropna4 = df_train_dropna2.drop('ProductCategory',axis=1)
df_test_dropna2 = pd.get_dummies(df_test_dropna1, columns=['CurrencyCode'])

# df_test_dropna4 = pd.get_dummies(df_test_dropna2, columns=['ProductCategory'])\

df_test_dropna4 = df_test_dropna2.drop('ProductCategory',axis=1)
for name in df_train_dropna4.columns:

    if 'Date' in name:

        df_train_dropna4[name + 'Year'] = df_train_dropna4[name].map(lambda s: s[:4]).astype(int)

        df_train_dropna4[name + 'Month'] = df_train_dropna4[name].map(lambda s: s[5:7]).astype(int)

        df_train_dropna4[name + 'Day'] = df_train_dropna4[name].map(lambda s: s[8:10]).astype(int)

        df_train_dropna4[name + 'Hour'] = df_train_dropna4[name].map(lambda s: s[11:13]).astype(int)

        df_train_dropna4[name + 'Minute'] = df_train_dropna4[name].map(lambda s: s[14:16]).astype(int)

        df_train_dropna4[name + 'Second'] = df_train_dropna4[name].map(lambda s: s[17:19]).astype(int)

        df_train_dropna4 = df_train_dropna4.drop(name,axis=1)

    elif 'Time' in name:

        df_train_dropna4[name + 'Year'] = df_train_dropna4[name].map(lambda s: s[:4]).astype(int)

        df_train_dropna4[name + 'Month'] = df_train_dropna4[name].map(lambda s: s[5:7]).astype(int)

        df_train_dropna4[name + 'Day'] = df_train_dropna4[name].map(lambda s: s[8:10]).astype(int)

        df_train_dropna4[name + 'Hour'] = df_train_dropna4[name].map(lambda s: s[11:13]).astype(int)

        df_train_dropna4[name + 'Minute'] = df_train_dropna4[name].map(lambda s: s[14:16]).astype(int)

        df_train_dropna4[name + 'Second'] = df_train_dropna4[name].map(lambda s: s[17:19]).astype(int)

        df_train_dropna4 = df_train_dropna4.drop(name,axis=1)
for name in df_test_dropna4.columns:

    if 'Date' in name:

        df_test_dropna4[name + 'Year'] = df_test_dropna4[name].map(lambda s: s[:4]).astype(int)

        df_test_dropna4[name + 'Month'] = df_test_dropna4[name].map(lambda s: s[5:7]).astype(int)

        df_test_dropna4[name + 'Day'] = df_test_dropna4[name].map(lambda s: s[8:10]).astype(int)

        df_test_dropna4[name + 'Hour'] = df_test_dropna4[name].map(lambda s: s[11:13]).astype(int)

        df_test_dropna4[name + 'Minute'] = df_test_dropna4[name].map(lambda s: s[14:16]).astype(int)

        df_test_dropna4[name + 'Second'] = df_test_dropna4[name].map(lambda s: s[17:19]).astype(int)

        df_test_dropna4 = df_test_dropna4.drop(name,axis=1)

    elif 'Time' in name:

        df_test_dropna4[name + 'Year'] = df_test_dropna4[name].map(lambda s: s[:4]).astype(int)

        df_test_dropna4[name + 'Month'] = df_test_dropna4[name].map(lambda s: s[5:7]).astype(int)

        df_test_dropna4[name + 'Day'] = df_test_dropna4[name].map(lambda s: s[8:10]).astype(int)

        df_test_dropna4[name + 'Hour'] = df_test_dropna4[name].map(lambda s: s[11:13]).astype(int)

        df_test_dropna4[name + 'Minute'] = df_test_dropna4[name].map(lambda s: s[14:16]).astype(int)

        df_test_dropna4[name + 'Second'] = df_test_dropna4[name].map(lambda s: s[17:19]).astype(int)

        df_test_dropna4 = df_test_dropna4.drop(name,axis=1)
y = df_train_dropna4['IsDefaulted']

X = df_train_dropna4.drop('IsDefaulted',axis=1)
X1 = X

for name in X1.columns:

    if name in df_test_dropna4.columns:

        1

    else:

        X1 = X1.drop(name,axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test1,y_train,y_test1 = train_test_split(X1,y ,test_size = 0.3,random_state= 0 )
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_jobs=2,random_state=0)

clf.fit(X_train,y_train)
y_pred_clf = clf.predict(X_test1)
from sklearn import metrics

metrics.roc_auc_score(y_test1,y_pred_clf)
clf1 = RandomForestClassifier(n_jobs=2,random_state=0)

clf1.fit(X1,y)

y_pred_clf1 = clf1.predict(df_test_dropna4)
transIds = df_test_dropna4["Tran_id"]

transId1s = []

for i in transIds:

    transId1s.append("TransactionId_" + str(i))
df12 = pd.DataFrame({"TransactionId" : transId1s , "IsDefaulted" : y_pred_clf1.astype(int)})
# df12.to_csv('Submission.csv',index=False)