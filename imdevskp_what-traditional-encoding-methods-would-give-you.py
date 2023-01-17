import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix



from sklearn import svm

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb



import eli5

from eli5.sklearn import PermutationImportance
train = pd.read_csv('../input/cat-in-the-dat/train.csv')

test = pd.read_csv('../input/cat-in-the-dat/test.csv')

train.head()
# train = train[200000:]
train.info()
# train.describe(include='all')
train[['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']].describe(include='all')
# label binerize binerize values to one or the other value



lb = LabelBinarizer()



for i in ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']:

    train[i] = lb.fit_transform(train[i])

    test[i] = lb.fit_transform(test[i])

# df.head()
train[['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']].describe()
nom_cols = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']



for i in nom_cols:

    dm_cols = pd.get_dummies(train[i])

    train = pd.concat([train, dm_cols], axis = 1) 

    train.drop(i, axis=1, inplace=True)

    

    dm_cols = pd.get_dummies(test[i])

    test = pd.concat([test, dm_cols], axis = 1) 

    test.drop(i, axis=1, inplace=True)

    

train.head()
hex_cols = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']



for i in hex_cols:

    train[i] = train[i].apply(lambda x: int(x, 16))

    train[i] = pd.to_datetime(train[i], unit='ms')

    test[i] = test[i].apply(lambda x: int(x, 16))

    test[i] = pd.to_datetime(test[i], unit='ms')
# df = train[['target', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']]

# df['nom_5_year'] = df['nom_5'].dt.year

# df['nom_5_month'] = df['nom_5'].dt.month

# df['nom_5_day'] = df['nom_5'].dt.day

# df['nom_5_hour'] = df['nom_5'].dt.hour

# df['nom_5_minute'] = df['nom_5'].dt.minute

# df['nom_5_second'] = df['nom_5'].dt.second



# df['nom_6_year'] = df['nom_6'].dt.year

# df['nom_6_month'] = df['nom_6'].dt.month

# df['nom_6_day'] = df['nom_6'].dt.day

# df['nom_6_hour'] = df['nom_6'].dt.hour

# df['nom_6_minute'] = df['nom_6'].dt.minute

# df['nom_6_second'] = df['nom_6'].dt.second



# df['diff'] = df['nom_6_year']-df['nom_5_year']





# plt.figure(figsize=(25, 25))

# sns.heatmap(df.corr(), annot=True, fmt='.1f', cmap='RdBu', vmax=0.8, vmin=-0.8)

# plt.show()
train[['ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']].describe(include='all')
ord_cols = ['ord_0', 'ord_1', 'ord_2', 'ord_3']



for i in ord_cols:

    print(i)

    print(train[i].value_counts())
en = LabelEncoder()



en = en.fit([1, 2, 3])

train['ord_0'] = en.transform(train['ord_0'])

test['ord_0'] = en.transform(test['ord_0'])



en = en.fit(['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster'])

train['ord_1'] = en.transform(train['ord_1'])

test['ord_1'] = en.transform(test['ord_1'])



en = en.fit(['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot'])

train['ord_2'] = en.transform(train['ord_2'])

test['ord_2'] = en.transform(test['ord_2'])



en = en.fit(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'])

train['ord_3'] = en.transform(train['ord_3'])

test['ord_3'] = en.transform(test['ord_3'])



en = en.fit(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 

             'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])

train['ord_4'] = en.transform(train['ord_4'])

test['ord_4'] = en.transform(test['ord_4'])
day = ['day', 'month']



for i in day:

    dm_cols = pd.get_dummies(train[i], prefix=i, prefix_sep='_')

    train = pd.concat([train, dm_cols], axis = 1) 

    train.drop(i, axis=1, inplace=True)

    

    dm_cols = pd.get_dummies(test[i], prefix=i, prefix_sep='_')

    test = pd.concat([test, dm_cols], axis = 1) 

    test.drop(i, axis=1, inplace=True)
# def cyc_enc(df, col, max_vals):

#     df[col] = np.sin(2 * np.pi * df[col]/max_vals)

#     df[col] = np.cos(2 * np.pi * df[col]/max_vals)

#     return df



# train = cyc_enc(train, 'day', 7)

# train = cyc_enc(train, 'month', 12)



# test = cyc_enc(test, 'day', 7) 

# test = cyc_enc(test, 'month', 12)
# train["ord_5a"]=train["ord_5"].str[0]

# train["ord_5b"]=train["ord_5"].str[1]



# test["ord_5a"]=test["ord_5"].str[0]

# test["ord_5b"]=test["ord_5"].str[1]
train.columns
# plt.figure(figsize=(40, 40))

# sns.heatmap(train.drop('id', axis=1).corr(), annot=True, fmt='.1f', cmap='RdBu', vmax=0.8, vmin=-0.8)

# plt.show()
imp_cols = ['bin_1', 'ord_0', 'ord_1', 'ord_3', 'ord_4', 'Blue', 'Red', 'Finland', 'Bassoon', 'month_2']
X = train[imp_cols]

X.info()
y = train['target']

# y.dtype
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# # SVM



# sv = svm.SVC()

# sv.fit(X_train, y_train)

# y_pred = sv.predict(X_test)

# accuracy_score(y_pred, y_test)
# # K Neares Neighbours



# knn = KNeighborsClassifier()

# knn.fit(X_train, y_train)

# y_pred = knn.predict(X_test)

# accuracy_score(y_pred, y_test)
# Logistic Regression



lr = LogisticRegression(C=0.123456789, solver="lbfgs", max_iter=5000)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

accuracy_score(y_pred, y_test)
# Naive Bayes



nb = GaussianNB()

nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)

accuracy_score(y_pred, y_test)
# Decision Tree



dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

accuracy_score(y_pred, y_test)
# Random Forest



rf = RandomForestClassifier()

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy_score(y_pred, y_test)
# valid_fraction = 0.1

# valid_size = int(len(train) * valid_fraction)



# train = train[['bin_1', 'ord_0', 'ord_1', 'ord_3', 'ord_4', 'Blue', 'Red', 'Finland', 'Bassoon', 'month_2', 'target']]



# train = train[:-2*valid_size]

# valid = train[-2*valid_size:-valid_size]

# test = train[-valid_size:]



# import lightgbm as lgb



# feature_cols = train.columns.drop('target')



# dtrain = lgb.Dataset(train[feature_cols], label=train['target'])

# dvalid = lgb.Dataset(valid[feature_cols], label=valid['target'])



# param = {'num_leaves':64, 'objective':'binary', 'metric':'auc'}

# num_round=1000



# bst = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)



# # from sklearn.metrics import roc_auc_score



# # ypred = bst.predict(test['bin_1', 'ord_0', 'ord_1', 'ord_3', 'ord_4', 'Blue', 'Red', 'Finland', 'Bassoon', 'month_2'])

# # score = roc_auc_score(test['target'], ypred)
perm = PermutationImportance(lr, random_state=1).fit(X_train, y_train)

eli5.show_weights(perm, feature_names = X_train.columns.tolist())
test.head()
ids = test["id"]

f = test[imp_cols]

p = bst.predict(f)



submission = pd.DataFrame({"id": ids, "target": p})

submission.to_csv("submission.csv",index=False)