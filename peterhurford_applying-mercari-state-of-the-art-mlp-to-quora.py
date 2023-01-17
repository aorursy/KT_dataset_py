print('Starting')
import os; os.environ['OMP_NUM_THREADS'] = '1'
from contextlib import contextmanager
from functools import partial
from operator import itemgetter
from multiprocessing.pool import ThreadPool
import time
from typing import List, Dict

import keras as ks
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from sklearn.pipeline import make_pipeline, make_union, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')

def on_field(f: str, *vec) -> Pipeline:
    return make_pipeline(FunctionTransformer(itemgetter(f), validate=False), *vec)

def to_records(df: pd.DataFrame) -> List[Dict]:
    return df.to_dict(orient='records')

def fit_predict(xs, y_train) -> np.ndarray:
    X_train, X_valid, X_test = xs
    config = tf.ConfigProto(
        intra_op_parallelism_threads=1, use_per_session_threads=1, inter_op_parallelism_threads=1)
    with tf.Session(graph=tf.Graph(), config=config) as sess, timer('fit_predict'):
        ks.backend.set_session(sess)
        model_in = ks.Input(shape=(X_train.shape[1],), dtype='float32', sparse=True)
        out = ks.layers.Dense(192, activation='relu')(model_in)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(64, activation='relu')(out)
        out = ks.layers.Dense(1, activation="sigmoid")(out)
        model = ks.Model(model_in, out)
        model.compile(loss='binary_crossentropy', optimizer=ks.optimizers.Adam(lr=3e-3), metrics=['accuracy'])
        for i in range(3):
            with timer(f'epoch {i + 1}'):
                model.fit(x=X_train, y=y_train, batch_size=2**(11 + i), epochs=1, verbose=0)
        with timer('predict'):
            return model.predict(X_valid)[:, 0], model.predict(X_test)[:, 0]

vectorizer = make_union(
    on_field('question_text', Tfidf(max_features=100000, token_pattern='\w+', ngram_range=(1, 2))),
    n_jobs=4)
with timer('process train'):
    train = pd.read_csv("../input/train.csv")
    cv = KFold(n_splits=20, shuffle=True, random_state=42)
    train_ids, valid_ids = next(cv.split(train))
    train, valid = train.iloc[train_ids], train.iloc[valid_ids]
    y_train = train['target']
    X_train = vectorizer.fit_transform(train).astype(np.float32)
    print(f'X_train: {X_train.shape} of {X_train.dtype}')
    del train
with timer('process valid'):
    X_valid = vectorizer.transform(valid).astype(np.float32)
    print(f'X_valid: {X_valid.shape} of {X_valid.dtype}')
with timer('process test'):
    test = pd.read_csv("../input/test.csv")
    X_test = vectorizer.transform(test).astype(np.float32)
    print(f'X_test: {X_test.shape} of {X_test.dtype}')
with ThreadPool(processes=4) as pool:
    Xb_train, Xb_valid, Xb_test = [x.astype(np.bool).astype(np.float32) for x in [X_train, X_valid, X_test]]
    xs = [[Xb_train, Xb_valid, Xb_test], [X_train, X_valid, X_test]] * 2
    preds = pool.map(partial(fit_predict, y_train=y_train), xs)
val_preds = [p[0] for p in preds]
val_pred_average = np.mean(val_preds, axis=0)
test_preds = [p[1] for p in preds]
test_pred_average = np.mean(test_preds, axis=0)
from sklearn.metrics import f1_score

y_val = valid['target']
for i, pred in enumerate(val_preds + [val_pred_average]):
    print('-')
    if i == 4:
        print('Ensemble')
    else:
        print('Model {}'.format(i))
    pred = np.array(pred)
    best_threshold = 0.01
    best_score = 0.0
    for threshold in range(1, 100):
        threshold = threshold / 100
        score = f1_score(y_val, pred > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    print("Score at threshold=0.5 is {}".format(f1_score(y_val, pred > 0.5)))
    print("Optimal threshold is {} with a score of {}".format(best_threshold, best_score))
y_te = (np.array(test_pred_average) > best_threshold).astype(np.int)
submit_df = pd.DataFrame({"qid": test["qid"], "prediction": y_te})
submit_df.head()
submit_df['prediction'].value_counts()
submit_df.to_csv("submission.csv", index=False)