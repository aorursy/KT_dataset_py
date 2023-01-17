import warnings

warnings.filterwarnings('ignore')

SEED = 0

import matplotlib.pyplot as plt 

import pandas as pd

import numpy as np
train = pd.read_csv('Data/input/TechJam2019/train.csv', index_col='id')

test = pd.read_csv('Data/input/TechJam2019/test.csv', index_col='id')

demo = pd.read_csv('Data/input/TechJam2019/demo.csv', index_col='id')

txn_file = pd.read_csv('Data/input/TechJam2019/txn.csv', index_col='id')
demo = demo.drop(columns=['n1'])

demo['c1'] = demo['c1'].fillna(demo['c1'].mode()[0])

demo['n2'] = demo['n2'].fillna(demo['n2'].median())
txn = txn_file.drop(columns=['old_cc_no', 't0', 'n3', 'n4'])

cat_cols_txn = ['old_cc_label','c5', 'c6', 'c7']

cat_cols_demo = ['c0', 'c1', 'c2', 'c3', 'c4']

num_cols_txn = ['n5', 'n6', 'n7']

num_cols_demo = ['n0', 'n2']
num_cols = num_cols_demo + num_cols_txn

cat_cols = cat_cols_demo + cat_cols_txn
## IMPLEMENTS

txn[num_cols_txn] = txn[num_cols_txn].groupby('id').sum()

txn[cat_cols_txn] = txn[cat_cols_txn].groupby('id').agg(lambda x: x.value_counts().index[0])
txn = txn.loc[~txn.index.duplicated(keep='first')]
train_data = train.join(demo, how='left').join(txn, how='left')
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

import category_encoders as ce
k0_encoder = ce.BinaryEncoder(cols=cat_cols_demo)

k1_encoder = ce.BinaryEncoder(cols=cat_cols_txn)

k0_encoder.fit(train_data[cat_cols_demo])

k1_encoder.fit(train_data[cat_cols_txn])

train0_data_encoded = k0_encoder.transform(train_data[cat_cols_demo])

train1_data_encoded = k1_encoder.transform(train_data[cat_cols_txn])



k0_scaler = MinMaxScaler().fit(train_data[num_cols_demo])

k1_scaler = MinMaxScaler().fit(train_data[num_cols_txn])

train0_data_encoded_scaled = train0_data_encoded.copy()

train1_data_encoded_scaled = train1_data_encoded.copy()

train0_data_encoded_scaled = train0_data_encoded_scaled.join(

    pd.DataFrame(k0_scaler.transform(train_data[num_cols_demo]), columns=num_cols_demo, index=train_data.index)

)

train1_data_encoded_scaled = train1_data_encoded_scaled.join(

    pd.DataFrame(k1_scaler.transform(train_data[num_cols_txn]), columns=num_cols_txn, index=train_data.index)

)
from sklearn.cluster import KMeans, MiniBatchKMeans

from sklearn.metrics import silhouette_score
kmeans0 = {}

for n in range(2,5):

    print('{} Clusters ---'.format(n))

    kmeans0[n] = MiniBatchKMeans(n_clusters=n, random_state=SEED)

    kmeans0[n].fit(train0_data_encoded_scaled)

    train_data['k0'+str(n)] = kmeans0[n].predict(train0_data_encoded_scaled)

    plt.hist(kmeans0[n].predict(train0_data_encoded_scaled))

    plt.show()
kmeans1 = {}

for n in range(2,5):

    print('{} Clusters ---'.format(n))

    kmeans1[n] = MiniBatchKMeans(n_clusters=n, random_state=SEED)

    kmeans1[n].fit(train1_data_encoded_scaled)

    train_data['k1'+str(n)] = kmeans1[n].predict(train1_data_encoded_scaled)

    plt.hist(kmeans1[n].predict(train1_data_encoded_scaled))

    plt.show()
import pickle

pickle.dump(kmeans0, open('kmeans0.pickle', 'wb'))

pickle.dump(kmeans1, open('kmeans1.pickle', 'wb'))

pickle.dump(k0_encoder, open('k0_encoder.pickle', 'wb'))

pickle.dump(k1_encoder, open('k1_encoder.pickle', 'wb'))

pickle.dump(k0_scaler, open('k0_scaler.pickle', 'wb'))

pickle.dump(k1_scaler, open('k1_scaler.pickle', 'wb'))
drop_cols = ['k0'+str(x) for x in range(2,5)] + ['k1'+str(x) for x in range(2,5)]
import lightgbm as lgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE, ADASYN

import category_encoders as ce
params = {

    'objective': 'multiclass',

    'metrics': 'multi_logloss',

    'num_class': 13,

    'learning_rate': 0.01,

    'num_leaves': 1000

}
log_cols = ['n2', 'n5', 'n7']

low_imp_cols = ['c4', 'c3', 'c2', 'c0']
for c in log_cols:

    train_data[c] = train_data[c].apply(lambda x: np.log(x))
from sklearn.decomposition import PCA
model= {}

encoder = {}

scaler = {}

pca = {}

for k0 in range(2,3):

    for k1 in range(2,2):

        print('-'*100)

        print('K0: {}, K1: {}'.format(k0,k1))

        train_f1_avg = 0

        test_f1_avg = 0

        n = 0

        for i in range(0, k0):

            model[i] = {}

            encoder[i] = {}

            scaler[i] = {}

            pca[i] = {}

            for j in range(0, k1):

                n += 1

                print('class: {}-{} '.format(i,j))

                train_data_k = train_data[(train_data['k0{}'.format(k0)] == i) & (train_data['k1{}'.format(k1)] == j)].drop(columns=drop_cols)

                ###DROP IMP

                #train_data_k = train_data_k.drop(columns=low_imp_cols)

                X = train_data_k.drop(columns='label')

                y = train_data_k['label']

                print('data size: {}'.format(len(X)))

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

                #UPSAMPLING

                encoder[i][j] = ce.BinaryEncoder(cols=cat_cols).fit(X_train)

                X_train = encoder[i][j].transform(X_train)

                X_test = encoder[i][j].transform(X_test)

                scaler[i][j] = MinMaxScaler().fit(X_train)

                X_train = pd.DataFrame(scaler[i][j].transform(X_train), columns=X_train.columns, index=X_train.index)

                X_test = pd.DataFrame(scaler[i][j].transform(X_test), columns=X_test.columns, index=X_test.index)

                

                pca[i][j] = PCA(n_components=30).fit(X_train)

                X_train = X_train.join(

                    pd.DataFrame(pca[i][j].transform(X_train), columns=['pca'+str(x) for x in range(30)], index=X_train.index)

                )

                X_test = X_test.join(

                    pd.DataFrame(pca[i][j].transform(X_test), columns=['pca'+str(x) for x in range(30)], index=X_test.index)

                )

                sm = SMOTE(random_state=SEED, n_jobs=-1, k_neighbors=1, sampling_strategy='minority')

                X_res, y_res = sm.fit_resample(X_train, y_train)

                X_res = pd.DataFrame(X_res, columns=X_train.columns)

                y_res = pd.DataFrame(y_res)

                X_train = X_res

                y_train = y_res

                #####

                lgb_train = lgb.Dataset(

                    X_train,

                    y_train

                )

                lgb_test = lgb.Dataset(

                    X_test,

                    y_test,

                    reference=lgb_train)



                model[i][j] = lgb.train(

                    params,

                    lgb_train,

                    #categorical_feature=cat_cols,

                    valid_sets=[lgb_train, lgb_test],

                    verbose_eval=1000,

                    num_boost_round=10000,

                    early_stopping_rounds=100,

                

                )

                train_pred_prob = model[i][j].predict(

                    X_train

                )

                test_pred_prob = model[i][j].predict(

                    X_test

                )

                train_pred = []

                for x in train_pred_prob:

                    train_pred.append(np.argmax(x))

                test_pred = []

                for x in test_pred_prob:

                    test_pred.append(np.argmax(x))



                from sklearn.metrics import f1_score, classification_report

                f1_train = f1_score(y_train, train_pred, average='weighted')

                f1_test = f1_score(y_test, test_pred, average='weighted')

                train_f1_avg += f1_train

                test_f1_avg += f1_test

                print('f1 train k0={} k1={} : {}'.format(i, j, f1_train))

                print('f1 test  k0={} k1={} : {}'.format(i, j, f1_test))

                print(classification_report(y_test, test_pred))

        print('AVERAGE TRAIN F1: ', train_f1_avg/n)

        print('AVERAGE TEST F1: ', test_f1_avg/n)

        print('-'*100)
for i in range(2):

    for j in range(2):

        print('model {}-{}'.format(i,j))

        lgb.plot_importance(model[i][j], figsize=(15,6))

        plt.show()
test_data = test.join(demo, how='left')

test_data = test_data.join(txn, how='left')





test0_data_encoded = k0_encoder.transform(test_data[cat_cols_demo])

test1_data_encoded = k1_encoder.transform(test_data[cat_cols_txn])



test0_data_encoded_scaled = test0_data_encoded.copy()

test1_data_encoded_scaled = test1_data_encoded.copy()



test0_data_encoded_scaled = test0_data_encoded_scaled.join(

    pd.DataFrame(k0_scaler.transform(test_data[num_cols_demo]), columns=num_cols_demo, index=test_data.index)

)

test1_data_encoded_scaled = test1_data_encoded_scaled.join(

    pd.DataFrame(k1_scaler.transform(test_data[num_cols_txn]), columns=num_cols_txn, index=test_data.index)

)
test_data['k0'+str(2)] = kmeans0[2].predict(test0_data_encoded_scaled)

test_data['k1'+str(2)] = kmeans1[2].predict(test1_data_encoded_scaled)
submission = pd.DataFrame(index=test_data.index)

for i in range(0,2):

    for j in range(0,2):

        X = test_data[(test_data['k02'] == i) & (test_data['k12'] == j)].copy()

        if len(X) > 0:

            for c in log_cols:

                X[c] = X[c].apply(lambda x: np.log(x))

            X = X.drop(columns=['k02', 'k12'])

            X = encoder[i][j].transform(X)

            pred = model[i][j].predict(X)

            submission = pd.concat([submission, pd.DataFrame(pred, index=X.index, columns=['class'+str(x) for x in range(0,13)])])
submission.dropna().to_csv('submission.csv')
y0_res.hist()
#X0_train, X0_test, y0_train, y0_test = train_test_split(X0_res, y0_res, test_size=0.2, random_state=SEED)

X0_train, X0_test, y0_train, y0_test = train_test_split(X0, y0, test_size=0.2, random_state=SEED)
import category_encoders as ce

from sklearn.preprocessing import RobustScaler
encoder0 = ce.BinaryEncoder(cols=cat_cols)

encoder0.fit(X0_train)

X0_train_encoded = encoder0.transform(X0_train)

X0_test_encoded = encoder0.transform(X0_test)

pickle.dump(encoder0, open('encoder0.pickle', 'wb'))
scaler0 = RobustScaler()

scaler0.fit(X0_train_encoded[num_cols])

pickle.dump(scaler0, open('scaler0.pickle', 'wb'))

X0_train_encoded_scaled = X0_train_encoded.copy()

X0_test_encoded_scaled = X0_test_encoded.copy()

X0_train_encoded_scaled[num_cols] = scaler0.transform(X0_train_encoded[num_cols])

X0_test_encoded_scaled[num_cols] = scaler0.transform(X0_test_encoded[num_cols])
params0 = {

    'objective': 'multiclass',

    'metrics': 'multi_logloss',

    'num_class': 13,

    'verbose_eval': 10,

    'learning_rate': 0.001,

    #'is_unbalance': True

}
lgb_train0 = lgb.Dataset(

    #X0_train,

    #X0_train_encoded,

    X0_train_encoded_scaled,

    y0_train

)

lgb_test0 = lgb.Dataset(

    #X0_test,

    #X0_test_encoded,

    X0_test_encoded_scaled,

    y0_test,

    reference=lgb_train0)



evals_result0 = {}  

base_model0 = lgb.train(

    params0,

    lgb_train0,

    valid_sets=[lgb_train0, lgb_test0],

    evals_result=evals_result0,

    verbose_eval=50,

    num_boost_round=20000,

    early_stopping_rounds=100,

)
ax = lgb.plot_metric(evals_result0, metric='multi_logloss')

plt.show()
import numpy as np

train_pred_prob0 = base_model0.predict(

    #X0_train

    #X0_train_encoded,

    X0_train_encoded_scaled

)

test_pred_prob0 = base_model0.predict(

    #X0_test

    #X0_test_encoded,

    X0_test_encoded_scaled

)

train_pred0 = []

for x in train_pred_prob0:

    train_pred0.append(np.argmax(x))

test_pred0 = []

for x in test_pred_prob0:

    test_pred0.append(np.argmax(x))



from sklearn.metrics import f1_score, classification_report

print('f1 train k_class=0 : {}'.format(f1_score(y0_train, train_pred0, average='weighted')))

print('f1 test  k_class=0 : {}'.format(f1_score(y0_test, test_pred0, average='weighted')))
print(classification_report(y0_test, test_pred0))
X1 = train_data_k1.drop(columns=['label', 'k_class'])

y1 = train_data_k1['label']
y1.hist()
sm = SMOTE(random_state=SEED, n_jobs=-1, k_neighbors=1)

X1_res, y1_res = sm.fit_resample(X1, y1)

X1_res = pd.DataFrame(X1_res, columns=X1.columns)

y1_res = pd.DataFrame(y1_res)
y1_res.hist()
#X1_train, X1_test, y1_train, y1_test = train_test_split(X1_res, y1_res, test_size=0.2, random_state=SEED)

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=SEED)
encoder1 = ce.BinaryEncoder(cols=cat_cols)

encoder1.fit(X1_train)

X1_train_encoded = encoder1.transform(X1_train)

X1_test_encoded = encoder1.transform(X1_test)

pickle.dump(encoder1, open('encoder1.pickle', 'wb'))
scaler1 = RobustScaler()

scaler1.fit(X1_train_encoded[num_cols])

pickle.dump(scaler1, open('scaler1.pickle', 'wb'))

X1_train_encoded_scaled = X1_train_encoded.copy()

X1_test_encoded_scaled = X1_test_encoded.copy()

X1_train_encoded_scaled[num_cols] = scaler1.transform(X1_train_encoded[num_cols])

X1_test_encoded_scaled[num_cols] = scaler1.transform(X1_test_encoded[num_cols])
params1 = {

    'objective': 'multiclass',

    'metrics': 'multi_logloss',

    'num_class': 13,

    'verbose_eval': 10,

    'learning_rate': 0.001

}
lgb_train1 = lgb.Dataset(

    #X1_train,

    #X1_train_encoded,

    X1_train_encoded_scaled,

    y1_train

)

lgb_test1 = lgb.Dataset(

    #X1_test

    #X1_test_encoded,

    X1_test_encoded_scaled,

    y1_test,

    reference=lgb_train1)



evals_result1 = {}  

base_model1 = lgb.train(

    params1,

    lgb_train1,

    valid_sets=[lgb_train1, lgb_test1],

    evals_result=evals_result1,

    verbose_eval=50,

    num_boost_round=20000,

    early_stopping_rounds=100

)
ax = lgb.plot_metric(evals_result1, metric='multi_logloss')

plt.show()
train_pred_prob1 = base_model1.predict(

    #X1_train

    #X1_train_encoded,

    X1_train_encoded_scaled

)

test_pred_prob1 = base_model1.predict(

    #X1_test

    #X1_test_encoded,

    X1_test_encoded_scaled

)

train_pred1 = []

for x in train_pred_prob1:

    train_pred1.append(np.argmax(x))

test_pred1 = []

for x in test_pred_prob1:

    test_pred1.append(np.argmax(x))



from sklearn.metrics import f1_score, classification_report

print('f1 train k_class=1 : {}'.format(f1_score(y1_train, train_pred1, average='weighted')))

print('f1 test  k_class=1 : {}'.format(f1_score(y1_test, test_pred1, average='weighted')))
print(classification_report(y1_test, test_pred1))
## TESTX0_test_encoded_scaled
test_data = test.join(demo, how='left')

test_scaled = k_scaler.transform(test_data)

test_scaled = pd.DataFrame(test_scaled, columns=test_data.columns, index=test_data.index)

test_data['k_class'] = kmeans.predict(test_scaled)

test_data = test_data.join(txn, how='left')

test_data0 = test_data[test_data['k_class'] == 0]

test_data1 = test_data[test_data['k_class'] == 1]

pred0 = base_model0.predict(test_data0.drop(columns=['k_class']))

pred1 = base_model1.predict(test_data1.drop(columns=['k_class']))
class_labels = ['class'+str(x) for x in range(0,13)]
pred0df = pd.DataFrame(pred0, index=test_data0.index, columns=class_labels)

pred1df = pd.DataFrame(pred1, index=test_data1.index, columns=class_labels)
submission = pd.concat([pred0df, pred1df])
submission.to_csv('submission.csv')
lgb.plot_importance(base_model0, figsize=(20,10))
lgb.plot_importance(base_model1, figsize=(20,10))
base_model0.save_model('base_model0.txt')

base_model1.save_model('base_model1.txt')