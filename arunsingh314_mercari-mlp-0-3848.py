## Idea for this code is taken from the kernel https://www.kaggle.com/lopuhin/mercari-golf-0-3875-cv-in-75-loc-1900-s
## Thanks to the authors for their elegant idea of using MLPs for this problem.

import warnings
warnings.filterwarnings('ignore')
import os
import gc
import time
from datetime import datetime

from contextlib import contextmanager

import keras as ks
import pandas as pd
import numpy as np
import scipy
import tensorflow as tf
from keras.models import load_model

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf


from scipy.sparse import hstack
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.model_selection import KFold
os.chdir('.../Mercari_NN')
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['name'] = df['name'].fillna('') + ' ' + df['brand_name'].fillna('')
    df['text'] = df['item_description'].fillna('') + ' ' + df['name'] + ' ' + df['category_name'].fillna('')
    return df[['name', 'text', 'shipping', 'item_condition_id']]
def load_train():
    
    train = pd.read_table('train.tsv')
    train = train[train['price'] > 0].reset_index(drop=True)
    cv = KFold(n_splits=20, shuffle=True, random_state=42)
    train_ids, valid_ids = next(cv.split(train))
    train, valid = train.iloc[train_ids], train.iloc[valid_ids]

    global y_scaler
    y_scaler = StandardScaler()

    global train_price, valid_price 
    train_price = train['price'].values.reshape(-1, 1)
    valid_price = valid['price'].values.reshape(-1, 1)

    y_train = y_scaler.fit_transform(np.log1p(train_price))
    
    y_valid = y_scaler.transform(np.log1p(valid_price))

    return train, valid, y_train, y_valid
def process_train(train, valid):
    
    global vectorizer1, vectorizer2, vectorizer3, vectorizer4
    with timer('process train'):
        train = preprocess(train)
        vectorizer1 = Tfidf(max_features=100000, token_pattern='\w+', dtype=np.float32)
        train_namevec  = vectorizer1.fit_transform(train['name'].values)

        vectorizer2 = Tfidf(max_features=100000, token_pattern='\w+', ngram_range=(1, 2), dtype=np.float32)
        train_textvec  = vectorizer2.fit_transform(train['text'].values)

        vectorizer3 = OneHotEncoder(dtype=np.float32)
        train_shipvec = vectorizer3.fit_transform(train['shipping'].values.reshape(-1, 1))

        vectorizer4 = OneHotEncoder(dtype=np.float32)
        train_conditionvec = vectorizer4.fit_transform(train['item_condition_id'].values.reshape(-1, 1))

        X_train = hstack((train_namevec, train_textvec, train_shipvec, train_conditionvec)).tocsr()

    with timer('process valid'):
        valid = preprocess(valid)

        valid_namevec  = vectorizer1.transform(valid['name'].values)

        valid_textvec  = vectorizer2.transform(valid['text'].values)

        valid_shipvec = vectorizer3.transform(valid['shipping'].values.reshape(-1, 1))

        valid_conditionvec = vectorizer4.transform(valid['item_condition_id'].values.reshape(-1, 1))

        X_valid = hstack((valid_namevec, valid_textvec, valid_shipvec, valid_conditionvec)).tocsr()
    
        # Binarizing input
        Xb_train, Xb_valid = [x.astype(np.bool).astype(np.float32) for x in [X_train, X_valid]]
        
    
    return X_train, X_valid, Xb_train, Xb_valid 
def load_process_test():

    test = pd.read_table('test.tsv')

    global predictions
    predictions = pd.DataFrame(test['test_id'])

    with timer('process test'):
        test = preprocess(test)

        test_namevec  = vectorizer1.transform(test['name'].values)

        test_textvec  = vectorizer2.transform(test['text'].values)

        test_shipvec = vectorizer3.transform(test['shipping'].values.reshape(-1, 1))

        test_conditionvec = vectorizer4.transform(test['item_condition_id'].values.reshape(-1, 1))

        X_test = hstack((test_namevec, test_textvec, test_shipvec, test_conditionvec)).tocsr()
    
        # Binarizing input
        Xb_test = X_test.astype(np.bool).astype(np.float32)
    
    return X_test, Xb_test

def run_model1(X_train, y_train, X_valid, y_valid):
    '''- returns an MLP model trained on tfidf vectorized sparse input.
    - Does not perform best on binarized input.
    - Uses Adam optimizer with constant learning rate. 
    - trains 2 epochs, Batch size is doubled at every epoch to speed up the optimization'''

    model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
    out = ks.layers.Dense(256, activation='relu')(model_in)
    # out = ks.layers.Dropout(0.1)(out)     ## performance is better without dropouts
    out = ks.layers.Dense(64, activation='relu')(out)
    # out = ks.layers.Dropout(0.1)(out)
    out = ks.layers.Dense(64, activation='relu')(out)
    # out = ks.layers.Dropout(0.2)(out)
    out = ks.layers.Dense(32, activation='relu')(out)
    out = ks.layers.Dense(1)(out)
    model = ks.Model(model_in, out)
    
    model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=3e-3))
    for i in range(2):
        with timer(f'epoch {i + 1}'):
            model.fit(x=X_train, y=y_train, batch_size=2**(9 + i), epochs=1, verbose=1, validation_data=(X_valid, y_valid))
    
    return model

def run_model2(Xb_train, y_train, Xb_valid, y_valid):
    '''- returns an MLP model trained on binarized sparse input.
    - Does not perform best on non-binarized(regular) input.
    - Uses Adam optimizer with constant learning rate. 
    - trains 3 epochs, Batch size is doubled at every epoch to speed up the optimization'''
    
    model_in = ks.Input(shape=(Xb_train.shape[1],), dtype='float32', sparse=True)
    out = ks.layers.Dense(256, activation='relu')(model_in)
    # out = ks.layers.Dropout(0.1)(out)     ## performance is better without dropouts
    out = ks.layers.Dense(64, activation='relu')(out)
    # out = ks.layers.Dropout(0.1)(out)
    out = ks.layers.Dense(64, activation='relu')(out)
    # out = ks.layers.Dropout(0.2)(out)
    out = ks.layers.Dense(32, activation='relu')(out)
    out = ks.layers.Dense(1)(out)
    model = ks.Model(model_in, out)
    
    model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=3e-3))
    for i in range(3):
        with timer(f'epoch {i + 1}'):
            model.fit(x=Xb_train, y=y_train, batch_size=2**(9 + i), epochs=1, verbose=1, validation_data=(Xb_valid, y_valid))
    
    return model
DEVELOP = True # Set to True for only trainng and validation

def main():

    start_time = datetime.now()

    print('\n\nLoading and processing train data.....')
    train, valid, y_train, y_valid = load_train()

    X_train, X_valid, Xb_train, Xb_valid = process_train(train, valid)
    print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)

    del train, valid
    gc.collect()

    ## Running model1 on regualr data X_train, X_valid
    print('\n\nRunning model on regular (non-binary) input.....')
    model1 = run_model1(X_train, y_train, X_valid, y_valid)
    model1.save('model1.h5')
    model1 = load_model('model1.h5')
    model1.summary()
    pred1 = model1.predict(X_valid)[:, 0]

    y_pred = np.expm1(y_scaler.inverse_transform(pred1.reshape(-1, 1))[:, 0])
    print('1st run val RMSLE: {:.4f}'.format(np.sqrt(msle(valid_price, y_pred))))
    
    ## Running again
    model2 = run_model1(X_train, y_train, X_valid, y_valid)
    model2.save('model2.h5')
    model2 = load_model('model2.h5')
    model2.summary()
    pred2 = model2.predict(X_valid)[:, 0]

    y_pred = np.expm1(y_scaler.inverse_transform(pred2.reshape(-1, 1))[:, 0])
    print('2nd run val RMSLE: {:.4f}'.format(np.sqrt(msle(valid_price, y_pred))))

    ## Running model2 on binarized data Xb_train, Xb_valid
    print('\n\nRunning model on binarized input.....')
    model3 = run_model2(Xb_train, y_train, Xb_valid, y_valid)
    model3.save('model3.h5')
    model3 = load_model('model3.h5')
    model3.summary()
    pred3 = model3.predict(Xb_valid)[:, 0]

    y_pred = np.expm1(y_scaler.inverse_transform(pred3.reshape(-1, 1))[:, 0])
    print('3rd run val RMSLE: {:.4f}'.format(np.sqrt(msle(valid_price, y_pred))))
    
    ## Running again
    model4 = run_model2(Xb_train, y_train, Xb_valid, y_valid)
    model4.save('model4.h5')
    model4 = load_model('model4.h5')
    model4.summary()
    pred4 = model4.predict(Xb_valid)[:, 0]

    y_pred = np.expm1(y_scaler.inverse_transform(pred4.reshape(-1, 1))[:, 0])
    print('4th run val RMSLE: {:.4f}'.format(np.sqrt(msle(valid_price, y_pred))))


    ## Final Prediction = weighted average of predictions of 4 models/runs
    print('\n\nEnsemble (weighted average of predictions from 4 models/runs).....')
    y_pred = np.average([pred1, pred2, pred3, pred4], weights=[0.33, 0.33, 0.17, 0.17], axis=0)
    y_pred = np.expm1(y_scaler.inverse_transform(y_pred.reshape(-1, 1))[:, 0])
    print('Final valid RMSLE: {:.4f}'.format(np.sqrt(msle(valid_price, y_pred))))

    if DEVELOP==False:
        ## This block loads and predicts on test data if DEVELOP is not set

        # print('\n\nLoading and processing test data.....')
        X_test, Xb_test = load_process_test()
        print(X_test.shape, Xb_test.shape)

        with timer('predict test'):
            test_pred1 = model1.predict(X_test)[:, 0]
            test_pred2 = model2.predict(X_test)[:, 0]
            test_pred3 = model3.predict(Xb_test)[:, 0]
            test_pred4 = model4.predict(Xb_test)[:, 0]


        test_pred = np.average([test_pred1, test_pred2, test_pred3, test_pred4], weights=[0.33, 0.33, 0.17, 0.17], axis=0)
        test_pred = np.expm1(y_scaler.inverse_transform(test_pred.reshape(-1, 1))[:, 0])

        print('\n\nCreating submisssion file.....')
        predictions['price'] = test_pred

        predictions.to_csv('predictions.csv', index=False)


    print(f'Code finished execution in {datetime.now() - start_time}')

if __name__ == '__main__':
    main()