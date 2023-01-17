import tensorflow as tf
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import re


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.metrics import f1_score

from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import gensim

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
%matplotlib inline
data_dir = "/kaggle/input/ykc-2nd/"
train = pd.read_csv(data_dir+"train.csv", index_col=0)
test = pd.read_csv(data_dir+"test.csv", index_col=0)
sub = pd.read_csv(data_dir+"sample_submission.csv", index_col=0)

# merge train and test
df = pd.concat([train, test])

# map words to list
# 記号や'sなど、いらない部分を取り除いている
df["product_name"] = df["product_name"].map(lambda w: list(
    filter(None, list(filter(None, 
                             re.sub("\'", "", 
                                    re.sub(r"-|/|\\|\|\"|\:|\;|\@|\^", " ", 
                                    re.sub(r"[0-9]+|\+|\&|\?|!|%|\.|,|\)|\(|\'s|®|™|#|~|©|",
                                           "", w.lower().replace("\xa0", " ")))).split(" "))))))

# get train dataframe
df_train = df[~df.department_id.isna()]
df_test = df[df.department_id.isna()]
n_department = 21
n_split = 5
df
# for tf-idf
docs = []
for i in range(n_department):
    names = itertools.chain.from_iterable(df_train[df_train["department_id"] == i]["product_name"].tolist())
    docs.append(" ".join([n for n in names]))
    

vectorizer = TfidfVectorizer(max_df=0.9)
X = vectorizer.fit_transform(docs)
words = vectorizer.get_feature_names()

word_to_index = dict(zip(words, np.arange(len(words))))
X = X.toarray()
def calculate_score(word_list: list):
    """
    convert word list to score 
    
    Args:
        word_list: list of string. word list

    Returns:
        np.ndarray: averaged score of the words. shape = [n_columns]
    """
    result = np.zeros([n_department])
    n = 0
    for w in word_list:
        try:
            result += X[:, word_to_index[w]]
            n += 1
        except KeyError as e:
            pass
    return result / n
tf_idf = df["product_name"].map(calculate_score)

# 正規化と、なぜかnanがいるので、nanを0にする(おそらく未知語?)
tf_idf = np.nan_to_num(np.array(tf_idf.tolist()))
max_value, min_value = tf_idf.max(), tf_idf.min()
tf_idf = (tf_idf - min_value) / (max_value - min_value)

# それぞれのスコアをカラムにする
df = df.merge(pd.DataFrame(tf_idf.tolist(), columns=["dept_"+str(i) for i in range(n_department)]), left_index=True, right_index=True)


df = df.reset_index()
## 訓練済みの単語ベクトルを読み込んで，product_nameに含まれる単語をベクトルに変換して平均を取ることで，各product_idに対して特徴量ベクトルを作成する

## gensimでvecから読み込む場合（５分ぐらいかかる）
model_ft = gensim.models.KeyedVectors.load_word2vec_format('../input/ykc-2nd/wiki-news-300d-1M.vec/wiki-news-300d-1M.vec')
# よくわからないが、それらしいライブラリをインストールした
# つながった単語を分解するライブラリ
# https://pypi.org/project/hyphenate/
# pip install hyphenate

!pip install hyphenate
from hyphenate import hyphenate_word

from collections import defaultdict
unused_words = defaultdict(int)
def to_vec(x, model_ft):
    v = np.zeros(model_ft.vector_size)
    for w in x:
        try:
            v += model_ft[w] ## 単語が訓練済みモデルのvocabにあったら
        except:
            # 単語を分解できたら分解
            hw = hyphenate_word(w)
            # 分解した単語に対して同じ処理
            for w2 in hw:
                try:
                    v += model_ft[w2]
                except:
                    unused_words[w2] += 1
    v = v / (np.sqrt(np.sum(v ** 2)) + 1e-16) ## 長さを1に正規化
    return v

def to_vec_max(x, model_ft):
    v = [np.zeros(model_ft.vector_size), np.zeros(model_ft.vector_size)]
    for w in x:    
        try:
            v.append(model_ft[w])
        except:
            hw = hyphenate_word(w)
            for w2 in hw:
                try:
                    v.append(model_ft[w2])
                except:
                    pass
    v = np.max(v, axis=0)
    v = v / (np.sqrt(np.sum(v ** 2)) + 1e-16) ## 長さを1に正規化
    return v

# SWEM_mean
vecs = df["product_name"].apply(lambda x : to_vec(x, model_ft))
vecs = np.vstack(vecs)
vecs = np.nan_to_num(vecs)

# SWEM_max_pooling
vecs_max = df["product_name"].apply(lambda x : to_vec_max(x, model_ft))
vecs_max = np.vstack(vecs_max)
vecs_max = np.nan_to_num(vecs_max)


fasttext_pretrain_cols = [f"fasttext_pretrain_vec{k}" for k in range(vecs.shape[1])]
fasttext_pretrain_cols_max = [f"fasttext_pretrain_vec_max{k}" for k in range(vecs.shape[1])]

# merge dataframes
vec_df = pd.DataFrame(vecs, columns=fasttext_pretrain_cols)
vec_max_df = pd.DataFrame(vecs_max, columns=fasttext_pretrain_cols_max)
df = pd.concat([df, vec_df], axis = 1)
df = pd.concat([df, vec_max_df], axis = 1)

df.head()
df_train = df[~df.department_id.isna()]
df_test = df[df.department_id.isna()]
np.random.seed(42)

# fasttext系の入ったカラム
trainCols = fasttext_pretrain_cols + fasttext_pretrain_cols_max + \
            ["order_rate", "order_dow_mode", "order_hour_of_day_mode"]

# tf-idf系のカラム
trainCols2 = ["dept_"+str(i) for i in range(n_department)] + ["order_rate", "order_dow_mode", "order_hour_of_day_mode"]

# NN用のtf-idf系
nn_cols = ["dept_"+str(i) for i in range(n_department)]

# NN用のtf-idf + fasttext
nn_cols2 = fasttext_pretrain_cols + fasttext_pretrain_cols_max + ["dept_"+str(i) for i in range(n_department)]

# それぞれX_trainという名前で取り出す
X_train = df_train[trainCols]
X2_train = df_train[trainCols2]
Xnn_train = df_train[nn_cols]
Xnn2_train = df_train[nn_cols2]

# 正解のカラム
Y_train = df_train["department_id"]

# NN用のOneHotの正解データ
Ynn_train = pd.get_dummies(df_train, columns=["department_id"])[[f"department_id_{i}.0" for i in range(21)]]

# validationとtrainのインデックスをランダムに決める
validInds = np.random.choice(X_train.index.values, 4000, replace = False)
trainInds = np.setdiff1d(X_train.index.values, validInds)
params = {
    'lambda_l1': 0.0,
    'lambda_l2': 0.0,
    'num_leaves': 200,
    'feature_fraction': 0.9840000000000001,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'min_child_samples': 20,
    'task': 'train',
    'boosting_type': 'dart',
    'objective': 'multiclass',
    'num_class': 21,
    'metric': 'multi_error',
    'verbose': 1,
    'learning_rate': 0.1,
    'num_iterations': 2000,
    'max_depth': 7
}
import lightgbm as lgb

lgb_train = lgb.Dataset(X_train.iloc[trainInds], Y_train.iloc[trainInds])

# valid data
lgb_val = lgb.Dataset(X_train.iloc[validInds], Y_train.iloc[validInds])

# train LGBM model
best_params, history = {}, []
lgb_model = lgb.train(params, lgb_train, 
                  valid_sets=lgb_val,
                  verbose_eval=10)
## predict on valid
pred_val_lgb = lgb_model.predict(X_train.iloc[validInds])

## evaluate
score = {
    "logloss"  : log_loss(Y_train.iloc[validInds], pred_val_lgb),
    "f1_micro" : f1_score(Y_train.iloc[validInds], np.argmax(pred_val_lgb, axis = 1), average = "micro")}

## predict on test
pred_test_lgb = lgb_model.predict(df_test[trainCols])
score
params = {
    'lambda_l1': 0.0,
    'lambda_l2': 0.0,
    'num_leaves': 111,
    'feature_fraction': 0.9840000000000001,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'min_child_samples': 20,
    'task': 'train',
    'boosting_type': 'dart',
    'objective': 'multiclass',
    'num_class': 21,
    'metric': 'multi_error',
    'verbose': 1,
    'learning_rate': 0.05,
    'num_iterations': 1000
}
lgb_train2 = lgb.Dataset(X2_train.iloc[trainInds], Y_train.iloc[trainInds])

# valid data
lgb_val2 = lgb.Dataset(X2_train.iloc[validInds], Y_train.iloc[validInds])

# train LGBM model
best_params, history = {}, []
lgb_model2 = lgb.train(params, lgb_train2, 
                  valid_sets=lgb_val2,
                  verbose_eval=10)
## predict on valid
pred_val_lgb2 = lgb_model2.predict(X2_train.iloc[validInds])

## evaluate
score = {
    "logloss"  : log_loss(Y_train.iloc[validInds], pred_val_lgb2),
    "f1_micro" : f1_score(Y_train.iloc[validInds], np.argmax(pred_val_lgb2, axis = 1), average = "micro")}

## predict on test
pred_test_lgb2 = lgb_model2.predict(df_test[trainCols2])
score
from tensorflow.keras import regularizers
nn_model1 = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(21, activation='softmax')
])
nn_model1.compile(optimizer=tf.optimizers.Adam(lr=0.001), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# early stopping
es_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
history = nn_model1.fit(Xnn2_train.iloc[trainInds], Ynn_train.iloc[trainInds], validation_data=(Xnn2_train.iloc[validInds], Ynn_train.iloc[validInds]),  epochs=100, batch_size=16, verbose=2,  callbacks=[es_cb])
pred_val_nn1 = nn_model1.predict(Xnn2_train.iloc[validInds])
pred_test_nn1 = nn_model1.predict(df_test[nn_cols2])
log_loss(Y_train.iloc[validInds], pred_val_nn1), f1_score(Y_train.iloc[validInds], np.argmax(pred_val_nn1, axis = 1), average = "micro")
from tensorflow.keras import regularizers
nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(21, activation='softmax')
])
nn_model.compile(optimizer=tf.optimizers.Adam(lr=0.001), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# early stopping
es_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
nn_cols = ["dept_"+str(i) for i in range(n_department)]

Xnn_train = df_train[nn_cols]
Y_train = df_train["department_id"]

history = nn_model.fit(Xnn_train.iloc[trainInds], Ynn_train.iloc[trainInds], validation_data=(Xnn_train.iloc[validInds], Ynn_train.iloc[validInds]),  epochs=100, batch_size=16, verbose=2,  callbacks=[es_cb])
# 学習曲線
plt.plot(history.history['accuracy'], label="train")
plt.plot(history.history['val_accuracy'], label="validation")
plt.legend()
plt.show()
# Lossの学習曲線
plt.plot(history.history['loss'], label="train")
plt.plot(history.history['val_loss'], label="validation")
plt.legend()
plt.show()
# 予測を行う
pred_val_nn = nn_model.predict(Xnn_train.iloc[validInds])
Xnn_test = df_test[nn_cols]
pred_test_nn = nn_model.predict(Xnn_test)
log_loss(Y_train.iloc[validInds], pred_val_nn), f1_score(Y_train.iloc[validInds], np.argmax(pred_val_nn, axis = 1), average = "micro")
rfc = RandomForestClassifier(random_state=0, n_estimators=1000, n_jobs=-1, verbose=1)
rfc.fit(Xnn_train.iloc[trainInds], Y_train.iloc[trainInds])
pred_val_rfc = rfc.predict_proba(Xnn_train.iloc[validInds])
pred_test_rfc = rfc.predict_proba(Xnn_test)
log_loss(Y_train.iloc[validInds], pred_val_rfc), f1_score(Y_train.iloc[validInds], np.argmax(pred_val_rfc, axis = 1), average = "micro")
rfc2 = RandomForestClassifier(random_state=0, n_estimators=1000, n_jobs=-1, verbose=1)
rfc2.fit(X_train.iloc[trainInds], Y_train.iloc[trainInds])
pred_val_rfc2 = rfc2.predict_proba(X_train.iloc[validInds])
pred_test_rfc2 = rfc2.predict_proba(df_test[trainCols])
log_loss(Y_train.iloc[validInds], pred_val_rfc2), f1_score(Y_train.iloc[validInds], np.argmax(pred_val_rfc2, axis = 1), average = "micro")
svc_clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
svc_clf.fit(Xnn_train.iloc[trainInds],  Y_train.iloc[trainInds])
pred_val_svc = svc_clf.predict_proba(Xnn_train.iloc[validInds])
pred_test_svc = svc_clf.predict_proba(Xnn_test)
log_loss(Y_train.iloc[validInds], pred_val_svc), f1_score(Y_train.iloc[validInds], np.argmax(pred_val_svc, axis = 1), average = "micro")
svc_clf2 = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
svc_clf2.fit(X_train.iloc[trainInds],  Y_train.iloc[trainInds])
pred_val_svc2 = svc_clf2.predict_proba(X_train.iloc[validInds])
pred_test_svc2 = svc_clf2.predict_proba(df_test[trainCols])
log_loss(Y_train.iloc[validInds], pred_val_svc2), f1_score(Y_train.iloc[validInds], np.argmax(pred_val_svc2, axis = 1), average = "micro")
len(validInds), len(trainInds)
# まずは、予測値(probability)を一つにくっつける
val_preds_all =  np.hstack([pred_val_lgb2,  pred_val_lgb,  pred_val_nn,  pred_val_nn1,   pred_val_svc,  pred_val_svc2,  pred_val_rfc,  pred_val_rfc2])
test_preds_all = np.hstack([pred_test_lgb2, pred_test_lgb, pred_test_nn, pred_test_nn1, pred_test_svc, pred_test_svc2, pred_test_rfc, pred_test_rfc2])
# Logistic Regressionの学習。ここではOne-vs-Restの学習になっている。
clf = LogisticRegression(multi_class="ovr").fit(val_preds_all, Y_train.iloc[validInds])
# テストd−得たに対する予測結果
prediction = clf.predict(test_preds_all)
prediction
# 予測結果を提出データにまとめる
sub["department_id"] = prediction.astype(int)
sub.to_csv("submission.csv", index = False)
sub.head()
sub.to_csv('submission.csv')
df.to_csv("dataframe_all.csv")
np.savetxt("val_preds_all.csv", val_preds_all, delimiter=",")
np.savetxt("test_preds_all.csv", test_preds_all, delimiter=",")
np.save("trainInds", trainInds)
np.save("validInds", validInds)
df_test.to_csv("dataframe_test.csv")
