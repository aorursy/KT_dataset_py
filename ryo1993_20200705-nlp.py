# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv")
df
df.info()
df.isnull().sum()
df.describe()
print(df["Message"][:20])
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer ,TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
def preprocessing_text(text):
    text = re.sub("[^a-zA-Z]"," ",text) # かなり大胆なsub
    text = text.lower()
    
    text = nltk.word_tokenize(text)
    
    lemma = nltk.WordNetLemmatizer()
    text= [lemma.lemmatize(word) for word in text]
    
    text = " ".join(text)
    return text
df["Message"] = df["Message"].apply(preprocessing_text)
df["Message"]
count_vectorizer = CountVectorizer(analyzer='word', stop_words = "english", ngram_range=(1,2))
tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words = "english", ngram_range=(1,2))

count_mat = count_vectorizer.fit_transform(df["Message"])
tfidf_mat = tfidf_vectorizer.fit_transform(df["Message"])
print(count_mat.shape)
%%time
n_comp_SVD = 20
n_comp_TSNE = 2

# SVD
text_svd = TruncatedSVD(n_components=n_comp_SVD,random_state=0)
df_count_svd = pd.DataFrame(text_svd.fit_transform(count_mat))
df_count_svd.columns = ['count_vec_svd_'+str(j+1) for j in range(n_comp_SVD)]

text_svd = TruncatedSVD(n_components=n_comp_SVD,random_state=0)
df_tfidf_svd = pd.DataFrame(text_svd.fit_transform(tfidf_mat))
df_tfidf_svd.columns = ['tfidf_vec_svd_'+str(j+1) for j in range(n_comp_SVD)]

# TSNE(ちょっと重いので、sklearnではない方のmodule使うとか(こっちの方が速いらしい))
tsne = TSNE(n_components=n_comp_TSNE, random_state = 0)
df_count_tsne = pd.DataFrame(tsne.fit_transform(count_mat))
df_count_tsne.columns = ['count_vec_tsne_'+str(j+1) for j in range(n_comp_TSNE)]

tsne = TSNE(n_components=n_comp_TSNE, random_state = 0)
df_tfidf_tsne = pd.DataFrame(tsne.fit_transform(tfidf_mat))
df_tfidf_tsne.columns = ['tfidf_vec_tsne_'+str(j+1) for j in range(n_comp_TSNE)]

df = pd.concat([df, df_count_svd,df_tfidf_svd,df_count_tsne,df_tfidf_tsne],axis=1)
df.columns
pd.set_option("display.max_columns", 80)
pd.set_option("display.max_rows", 80)
df.head()
import matplotlib.pyplot as plt
import seaborn as sns
sns.scatterplot(x = "count_vec_tsne_1", y = "count_vec_tsne_2", hue = "Category", data = df)
sns.scatterplot(x = "tfidf_vec_tsne_1", y = "tfidf_vec_tsne_2", hue = "Category", data = df)
y = df.iloc[:,0].values
x = df.drop(["Category", "Message"], axis = 1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)
#We make model for predict
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)
print("the accuracy of our model: {}".format(nb.score(x_test,y_test)))
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter = 200)
lr.fit(x_train,y_train)
print("our accuracy is: {}".format(lr.score(x_test,y_test)))
from sklearn.feature_selection import SelectKBest, mutual_info_classif
x_train_tfidf, x_test_tfidf, y_train, y_test = train_test_split(tfidf_mat,y, test_size = 0.2, random_state = 42)
# 重い
selector = SelectKBest(k = 10000, score_func = mutual_info_classif)
selector.fit(x_train_tfidf, y_train)
x_train_new = selector.transform(x_train_tfidf)
x_test_new = selector.transform(x_test_tfidf)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter = 200)
lr.fit(x_train_new,y_train)
print("our accuracy is: {}".format(lr.score(x_test_new,y_test)))
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import lightgbm as lgb
le = LabelEncoder()
df["Category"] = le.fit_transform(df["Category"])
y = df["Category"]
x = df.drop(["Category", "Message"], axis = 1)
# 学習
y_preds = []
models = []
oof_train = np.zeros(len(x))

kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

params = {
    "boosting_type" : "gbdt",
    "objective" : "binary",
    "metric" : "binary_logloss",
    "max_depth" : -1, # デフォルト値は-1で0以下の値は制限なしを意味する．
    #"num_leaves" : 31,  #木にある分岐の個数．sklearnのmax_depthsklearnのmax_depthとの関係はnum_leaves=2^max_depth
                                    #デフォルトは31．大きくすると精度は上がるが過学習が進む．
    "learning_rate": 0.03, 
    "feature_fraction" : 0.7, #学習の高速化と過学習の抑制に使用される．データの特徴量のfeature_fraction * 100 % だけ使用する．
    'bagging_fraction': 0.7, #like feature_fraction, but this will randomly select part of data without resampling
    "bagging_freq" : 5, #frequency for bagging. 0 means disable bagging; k means perform bagging at every k iteration
                                    # Note: to enable bagging, bagging_fraction should be set to value smaller than 1.0 as well
    "early_stopping_rounds" : 100,
    'n_estimators':2000, # aliases: num_iteration, n_iter, num_tree, num_trees, num_round, num_rounds, num_boost_round, n_estimators. 
                                        # number of boosting iterations
     "seed" : 42,
    "reg_alpha":0.1,
    "reg_lambda":0.1 #0.1
}


for fold, (train_index, val_index) in enumerate(kf.split(x, y)):
    tr_x = x.iloc[train_index, :]
    tr_y = y[train_index]
    val_x = x.iloc[val_index, :]
    val_y = y[val_index]
    
    lgb_train = lgb.Dataset(tr_x, tr_y)
    lgb_eval = lgb.Dataset(val_x, val_y, reference = lgb_train)
    
    model = lgb.train(params, 
                      lgb_train,
                      valid_sets = [lgb_train, lgb_eval], 
                      verbose_eval= 10
                     )
    oof_train[val_index] = model.predict(val_x, num_iteration = model.best_iteration)
    #y_pred = model.predict(test, num_iteration = model.best_iteration)
    
    #y_preds.append(y_pred)
    models.append(model)
accuracy_score(np.round(oof_train), y)
fi = pd.DataFrame(index = x.columns)
for i, m in enumerate(models):
    fi["model_"+str(i+1)] = m.feature_importance(importance_type = "gain")
fi["fi_ave"] = fi.mean(axis = 1)
fi.sort_values(by = "fi_ave", inplace = True, ascending = False)
plt.figure(figsize = (15,12))
sns.barplot(x = fi["fi_ave"][:30], y = fi.index[:30])
plt.show()
fi
sns.scatterplot(x = "tfidf_vec_tsne_1", y = "tfidf_vec_svd_5", data = df, hue = "Category")
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=10,random_state=0)
count_lda = lda.fit_transform(count_mat)
count_features = count_vectorizer.get_feature_names()
for tn in range(10):
    print("topic #"+str(tn))
    row = lda.components_[tn]
    words = ', '.join([count_features[i] for i in row.argsort()[:-20-1:-1]])
    print(words, "\n")
tfidf_lda = lda.fit_transform(tfidf_mat)
tfidf_features = tfidf_vectorizer.get_feature_names()
for tn in range(10):
    print("topic #"+str(tn))
    row = lda.components_[tn]
    words = ', '.join([tfidf_features[i] for i in row.argsort()[:-20-1:-1]])
    print(words, "\n")
lda_count_df = pd.DataFrame(count_lda)
lda_count_df = lda_count_df.add_prefix("lda_count_")
lda_tfidf_df = pd.DataFrame(tfidf_lda)
lda_tfidf_df = lda_tfidf_df.add_prefix("lda_tfidf_")
df_lda = pd.concat([df,lda_count_df, lda_tfidf_df], axis = 1)
df_lda
y = df_lda["Category"]
x = df_lda.drop(["Category", "Message"], axis = 1)
# 学習
y_preds = []
models = []
oof_train = np.zeros(len(x))

kf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 0)

params = {
    "boosting_type" : "gbdt",
    "objective" : "binary",
    "metric" : "binary_logloss",
    "max_depth" : -1, # デフォルト値は-1で0以下の値は制限なしを意味する．
    #"num_leaves" : 31,  #木にある分岐の個数．sklearnのmax_depthsklearnのmax_depthとの関係はnum_leaves=2^max_depth
                                    #デフォルトは31．大きくすると精度は上がるが過学習が進む．
    "learning_rate": 0.03, 
    "feature_fraction" : 0.7, #学習の高速化と過学習の抑制に使用される．データの特徴量のfeature_fraction * 100 % だけ使用する．
    'bagging_fraction': 0.7, #like feature_fraction, but this will randomly select part of data without resampling
    "bagging_freq" : 5, #frequency for bagging. 0 means disable bagging; k means perform bagging at every k iteration
                                    # Note: to enable bagging, bagging_fraction should be set to value smaller than 1.0 as well
    "early_stopping_rounds" : 100,
    'n_estimators':2000, # aliases: num_iteration, n_iter, num_tree, num_trees, num_round, num_rounds, num_boost_round, n_estimators. 
                                        # number of boosting iterations
     "seed" : 42,
    "reg_alpha":0.1,
    "reg_lambda":0.1 #0.1
}


for fold, (train_index, val_index) in enumerate(kf.split(x, y)):
    tr_x = x.iloc[train_index, :]
    tr_y = y[train_index]
    val_x = x.iloc[val_index, :]
    val_y = y[val_index]
    
    lgb_train = lgb.Dataset(tr_x, tr_y)
    lgb_eval = lgb.Dataset(val_x, val_y, reference = lgb_train)
    
    model = lgb.train(params, 
                      lgb_train,
                      valid_sets = [lgb_train, lgb_eval], 
                      verbose_eval= 10
                     )
    oof_train[val_index] = model.predict(val_x, num_iteration = model.best_iteration)
    #y_pred = model.predict(test, num_iteration = model.best_iteration)
    
    #y_preds.append(y_pred)
    models.append(model)
accuracy_score(np.round(oof_train), y)
fi = pd.DataFrame(index = x.columns)
for i, m in enumerate(models):
    fi["model_"+str(i+1)] = m.feature_importance(importance_type = "gain")
fi["fi_ave"] = fi.mean(axis = 1)
fi.sort_values(by = "fi_ave", inplace = True, ascending = False)
fi[:50]
plt.figure(figsize = (15,12))
sns.barplot(x = fi["fi_ave"][:30], y = fi.index[:30])
plt.show()
fi
