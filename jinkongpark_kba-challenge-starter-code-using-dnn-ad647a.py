import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

pd.set_option('max_columns', 10, 'max_rows', 20)
tr_train = pd.read_csv('../input/X_train.csv', encoding='cp949')
tr_test = pd.read_csv('../input/X_test.csv', encoding='cp949')
tr = pd.concat([tr_train, tr_test])
tr
features = []
f = tr.groupby('cust_id')['amount'].agg([('총구매액', 'sum')]).reset_index()
features.append(f); f
f = tr.groupby('cust_id')['amount'].agg([('구매건수', 'size')]).reset_index()
features.append(f); f
f = tr.groupby('cust_id')['amount'].agg([('평균구매가격', 'mean')]).reset_index()
features.append(f); f
n = tr.gds_grp_nm.nunique()
f = tr.groupby('cust_id')['gds_grp_nm'].agg([('구매상품다양성', lambda x: len(x.unique()) / n)]).reset_index()
features.append(f); f
tr['sales_date'] = tr.tran_date.str[:10]
f = tr.groupby(by = 'cust_id')['sales_date'].agg([('내점일수','nunique')]).reset_index()
features.append(f); f
def weekday(x):
    w = x.dayofweek 
    if w < 4:
        return 1 # 주중
    else:
        return 0 # 주말
f = tr.groupby(by = 'cust_id')['sales_date'].agg([('요일구매패턴', lambda x : pd.to_datetime(x).apply(weekday).value_counts().index[0])]).reset_index()
features.append(f); f
def f1(x):
    k = x.month
    if 3 <= k <= 5 :
        return('봄-구매건수')
    elif 6 <= k <= 8 :
        return('여름-구매건수')
    elif 9 <= k <= 11 :    
        return('가을-구매건수')
    else :
        return('겨울-구매건수')    
    
tr['season'] = pd.to_datetime(tr.sales_date).apply(f1)
f = pd.pivot_table(tr, index='cust_id', columns='season', values='amount', 
                   aggfunc=np.size, fill_value=0).reset_index()
features.append(f); f
f = tr.groupby('cust_id')['gds_grp_mclas_nm'].agg([('주구매코너', lambda x: x.value_counts().index[0])]).reset_index()
f = pd.get_dummies(f, columns=['주구매코너'])  # This method performs One-hot-encoding
features.append(f); f
X_train = DataFrame({'cust_id': tr_train.cust_id.unique()})
for f in features :
    X_train = pd.merge(X_train, f, how='left')
display(X_train)

X_test = DataFrame({'cust_id': tr_test.cust_id.unique()})
for f in features :
    X_test = pd.merge(X_test, f, how='left')
display(X_test)
IDtest = X_test.cust_id;
X_train.drop(['cust_id'], axis=1, inplace=True)
X_test.drop(['cust_id'], axis=1, inplace=True)
y_train = pd.read_csv('../input/y_train.csv').gender
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.constraints import max_norm
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier

# Set common values
N_HIDDEN = 128
DROPOUT = 0.5
FEATURES = X_train.shape[1]

# Transforms features by min-max scaling
from sklearn.preprocessing import MinMaxScaler , StandardScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define model architecture
def dnn_model() :
    model = Sequential()
    model.add(Dense(N_HIDDEN, input_shape=(FEATURES,), activation='elu', 
                    kernel_constraint=max_norm(2.), kernel_initializer='he_normal'))
    model.add(Dense(int(N_HIDDEN/2), activation='elu', 
                    kernel_constraint=max_norm(2.), kernel_initializer='he_normal'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Instantiate the Scikit-Learn classifier interface
dnn = KerasClassifier(build_fn=dnn_model, batch_size=16, epochs=20, verbose=1)

# Evaluate the model using k-fold CV
score = cross_val_score(dnn, X_train, y_train, cv=5, scoring='roc_auc')
print('{}\nmean = {:.5f}\nstd = {:.5f}'.format(score, score.mean(), score.std()))
dnn.fit(X_train, y_train)
pred = dnn.predict_proba(X_test)[:,1]
fname = 'submissions.csv'
submissions = pd.concat([IDtest, pd.Series(pred, name="gender")] ,axis=1)
submissions.to_csv(fname, index=False)
print("'{}' is ready to submit." .format(fname))