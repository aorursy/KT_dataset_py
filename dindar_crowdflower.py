import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import scipy
import nltk
from scipy.sparse import coo_matrix, hstack
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import pairwise
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib
import matplotlib.patches as mpatches
from sklearn.model_selection import LeaveOneOut
from xgboost import XGBClassifier
from sklearn.metrics import cohen_kappa_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from Levenshtein import distance

loo = LeaveOneOut()
train = pd.read_csv('C:/Users/Dindar/crowdflower/tables1/train.csv')
test = pd.read_csv('C:/Users/Dindar/crowdflower/tables1/test.csv')
df = train[train['query'] == 'bridal shower decorations']
df['product_description'].fillna('0', inplace=True)
stop_words = set(stopwords.words('english')) 
df['query'] = df['query'].str.lower()
df['product_title'] = df['product_title'].str.lower()
df['product_description'] = df['product_description'].str.lower()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
df['query'] = df['query'].apply(lemmatize_text)
df['product_title'] = df['product_title'].apply(lemmatize_text)
df['product_description'] = df['product_description'].apply(lemmatize_text)
def listToString(s):  
    str1 = ""  
    for ele in s:  
        str1 = str1 + ' ' + ele      
    return str1[1:]
df['query'] = df['query'].apply(listToString)
df['product_title'] = df['product_title'].apply(listToString)
df['product_description'] = df['product_description'].apply(listToString)
import gensim

word2vec_path = "G:/Downloads/GoogleNews-vectors-negative300.bin.gz"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
type(word2vec)
def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_questions,col, generate_missing=False):
    embeddings = clean_questions[col].apply(lambda x: get_average_word2vec(x, vectors, 
                                                                                generate_missing=generate_missing))
    return list(embeddings)
df['product_title'] = df['product_title'].str.split()
df['product_description'] = df['product_description'].str.split()
embeddings = get_word2vec_embeddings(word2vec, df,'product_description')
embeddings1 = get_word2vec_embeddings(word2vec, df,'product_title')
embeddings = np.asarray(embeddings)
embeddings1 = np.asarray(embeddings1)
embeddings = pd.DataFrame(embeddings)
embeddings1 = pd.DataFrame(embeddings1)
embeddings['target'] = list(df['median_relevance'])
embeddings1['target1'] = list(df['median_relevance'])
a = pairwise.cosine_similarity(embeddings)
a = pd.DataFrame(a)
meann = []
maxx = []
minn = []
for i in a.index:
    iii = a.iloc[i].sort_values(ascending=False)[:6].index
    meann.append(embeddings.iloc[iii]['target'].mean())
    maxx.append(embeddings.iloc[iii]['target'].max())
    minn.append(embeddings.iloc[iii]['target'].min())
embeddings['mean_6_cos_word2vec'] = meann
embeddings['max_6_cos_word2vec'] = maxx
embeddings['min_6_cos_word2vec'] = minn

one = []
two = []
three = []
four = []
for i in range(43):
    one.append(0)
    two.append(0)
    three.append(0)
    four.append(0)
for i in a.index:
    iii = a.iloc[i].sort_values()[:6].index
    for j in embeddings.iloc[iii]['target']:
        if j == 1:
            one[i] = one[i] + 1
        elif j == 2:
            two[i] = two[i] + 1
        elif j == 3:
            three[i] = three[i] + 1
        elif j == 4:
            four[i] = four[i] + 1
embeddings['one_6_cos_word2vec'] = one
embeddings['two_6_cos_word2vec'] = two
embeddings['three_6_cos_word2vec'] = three
embeddings['four_6_cos_word2vec'] = four

a = pairwise.cosine_similarity(embeddings1)
a = pd.DataFrame(a)
meann = []
maxx = []
minn = []
for i in a.index:
    iii = a.iloc[i].sort_values(ascending=False)[:3].index
    meann.append(embeddings1.iloc[iii]['target1'].mean())
    maxx.append(embeddings1.iloc[iii]['target1'].max())
    minn.append(embeddings1.iloc[iii]['target1'].min())
embeddings1['mean_3_cos_word2vec_title'] = meann
embeddings1['max_3_cos_word2vec_title'] = maxx
embeddings1['min_3_cos_word2vec_title'] = minn
one = []
two = []
three = []
four = []
for i in range(43):
    one.append(0)
    two.append(0)
    three.append(0)
    four.append(0)
for i in a.index:
    iii = a.iloc[i].sort_values()[:3].index
    for j in embeddings1.iloc[iii]['target1']:
        if j == 1:
            one[i] = one[i] + 1
        elif j == 2:
            two[i] = two[i] + 1
        elif j == 3:
            three[i] = three[i] + 1
        elif j == 4:
            four[i] = four[i] + 1
embeddings1['one_3_cos_word2vec_title'] = one
embeddings1['two_3_cos_word2vec_title'] = two
embeddings1['three_3_cos_word2vec_title'] = three
embeddings1['four_3_cos_word2vec_title'] = four
embeddings.shape, embeddings1.shape
embeddings1.drop('target1',axis=1,inplace=True)
data = pd.concat([embeddings, embeddings1], axis=1)


vectorizer = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 7))
vectorizer1 = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 7))
vectorizer2 = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 7))
vectorizer3 = TfidfVectorizer(stop_words=stop_words, ngram_range=(1, 4), analyzer='char')
X3 = vectorizer3.fit_transform(df['product_description'])
X1 = vectorizer1.fit_transform(df['product_title'])
X2 = vectorizer2.fit_transform(df['product_description'])
f1 = vectorizer1.get_feature_names()
f2 = vectorizer2.get_feature_names()
f3 = vectorizer3.get_feature_names()
tr = hstack([X1,X2,X3])
y = df['median_relevance']
tr
data = pd.DataFrame(tr.A)
data['target'] = list(y)

svd = TruncatedSVD(n_components=4, random_state=42)
X_reduced1 = svd.fit_transform(tr)
tr.shape
X_reduced1.shape
data = pd.DataFrame(X_reduced1)
data['target'] = list(y)
data.shape
a = pairwise.cosine_similarity(data.drop('target', axis=1).values)
a = pd.DataFrame(a)
meann = []
maxx = []
minn = []
for i in a.index:
    iii = a.iloc[i].sort_values(ascending=False)[:4].index
    meann.append(data.iloc[iii]['target'].mean())
    maxx.append(data.iloc[iii]['target'].max())
    minn.append(data.iloc[iii]['target'].min())
data['mean_4_cos'] = meann
data['max_4_cos'] = maxx
data['min_4_cos'] = minn
one = []
two = []
three = []
four = []
for i in range(43):
    one.append(0)
    two.append(0)
    three.append(0)
    four.append(0)
for i in a.index:
    iii = a.iloc[i].sort_values()[:4].index
    for j in data.iloc[iii]['target']:
        if j == 1:
            one[i] = one[i] + 1
        elif j == 2:
            two[i] = two[i] + 1
        elif j == 3:
            three[i] = three[i] + 1
        elif j == 4:
            four[i] = four[i] + 1
data['one_4_cos'] = one
data['two_4_cos'] = two
data['three_4_cos'] = three
data['four_4_cos'] = four

df.set_index(pd.Index(list(range(43))), inplace=True)
a = []
for i in df.index:
    b = []
    for j in df.index:
        b.append(distance(df.iloc[i]['product_description'], df.iloc[j]['product_description']))
    a.append(b)
len(a)
a = np.asarray(a)
a = pd.DataFrame(a)
meann = []
maxx = []
minn = []
b = []
for i in a.index:
    iii = a.iloc[i].sort_values()[:4].index

    b.append(np.mean(a.iloc[i].iloc[iii]))
    meann.append(data.iloc[iii]['target'].mean())
    maxx.append(data.iloc[iii]['target'].max())
    minn.append(data.iloc[iii]['target'].min())
data['mean_4_lev'] = meann
data['max_4_lev'] = maxx
data['min_4_lev'] = minn
data['mean_dis_lev_4'] = b
one = []
two = []
three = []
four = []
for i in range(43):
    one.append(0)
    two.append(0)
    three.append(0)
    four.append(0)
for i in a.index:
    iii = a.iloc[i].sort_values()[:4].index
    for j in data.iloc[iii]['target']:
        if j == 1:
            one[i] = one[i] + 1
        elif j == 2:
            two[i] = two[i] + 1
        elif j == 3:
            three[i] = three[i] + 1
        elif j == 4:
            four[i] = four[i] + 1
data['one_4_lev'] = one
data['two_4_lev'] = two
data['three_4_lev'] = three
data['four_4_lev'] = four

a = []
for i in df.index:
    b = []
    for j in df.index:
        b.append(distance(df.iloc[i]['product_title'], df.iloc[j]['product_title']))
    a.append(b)
a = np.asarray(a)
a = pd.DataFrame(a)
meann = []
maxx = []
minn = []
b = []
for i in a.index:
    iii = a.iloc[i].sort_values()[:4].index
    b.append(np.mean(a.iloc[i].iloc[iii]))
    meann.append(data.iloc[iii]['target'].mean())
    maxx.append(data.iloc[iii]['target'].max())
    minn.append(data.iloc[iii]['target'].min())
data['mean_4_lev_title'] = meann
data['max_4_lev_title'] = maxx
data['min_4_lev_title'] = minn
data['mean_dis_lev_4_title'] = b
one = []
two = []
three = []
four = []
for i in range(43):
    one.append(0)
    two.append(0)
    three.append(0)
    four.append(0)
for i in a.index:
    iii = a.iloc[i].sort_values()[:4].index
    for j in data.iloc[iii]['target']:
        if j == 1:
            one[i] = one[i] + 1
        elif j == 2:
            two[i] = two[i] + 1
        elif j == 3:
            three[i] = three[i] + 1
        elif j == 4:
            four[i] = four[i] + 1
data['one_4_lev_title'] = one
data['two_4_lev_title'] = two
data['three_4_lev_title'] = three
data['four_4_lev_title'] = four

from distance import jaccard
a = []
for i in df.index:
    b = []
    for j in df.index:
        b.append(jaccard(df.iloc[i]['product_title'], df.iloc[j]['product_title']))
    a.append(b)
a = np.asarray(a)
a = pd.DataFrame(a)
meann = []
maxx = []
minn = []
b = []
for i in a.index:
    iii = a.iloc[i].sort_values()[:4].index
    b.append(np.mean(a.iloc[i].iloc[iii]))
    meann.append(data.iloc[iii]['target'].mean())
    maxx.append(data.iloc[iii]['target'].max())
    minn.append(data.iloc[iii]['target'].min())
data['mean_4_jar_title'] = meann
data['max_4_jar_title'] = maxx
data['min_4_jar_title'] = minn
data['mean_dis_jar_4_title'] = b
one = []
two = []
three = []
four = []
for i in range(43):
    one.append(0)
    two.append(0)
    three.append(0)
    four.append(0)
for i in a.index:
    iii = a.iloc[i].sort_values()[:4].index
    for j in data.iloc[iii]['target']:
        if j == 1:
            one[i] = one[i] + 1
        elif j == 2:
            two[i] = two[i] + 1
        elif j == 3:
            three[i] = three[i] + 1
        elif j == 4:
            four[i] = four[i] + 1
data['one_4_jar_title'] = one
data['two_4_jar_title'] = two
data['three_4_jar_title'] = three
data['four_4_jar_title'] = four
from catboost import CatBoostClassifier
X = data.drop(['target','min_4_lev_title','max_4_lev_title'],axis=1)
y = data['target']
weight = 1 / (1 + df['relevance_variance'])
%%time
y_ts = []
y_pr = []
for train_index, test_index in loo.split(X):
    X_train, X_test = X.values[train_index], X.values[test_index]
    y_train, y_test = y.values[train_index], y.values[test_index]
    w = weight.values[train_index]
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train, sample_weight=w)
    y_preds = xgb.predict(X_test)
    y_ts.append(y_test)
    y_pr.append(y_preds)
cohen_kappa_score(y_ts, y_pr ,weights='quadratic')# word2vec(avg) title(3), description(6,5) + sample weight, cos
cohen_kappa_score(y_ts, y_pr ,weights='quadratic')# word2vec(avg) title(3), description(6,5) + sample weight, cos

cohen_kappa_score(y_ts, y_pr ,weights='quadratic')# word2vec(avg) title(3), description(6) + sample weight, cos
cohen_kappa_score(y_ts, y_pr ,weights='quadratic')# desc char 1,4, compon = 5, iter = жок
                                                  # lev 4 descr, title
                                                  # data.drop(['target',3,4,'one_4_lev_title','one_4_lev','mean_4_lev']
                                                  # lev adding count features one, two, three, four
cohen_kappa_score(y_ts, y_pr ,weights='quadratic')# des char 1,4, compon = 5, iter = жок
                                                  # lev 4 descr, title
                                                  # data.drop(['target',3,4,'one_4_lev_title', 'one_4_lev']
                                                  # lev adding count features one, two, three, four
cohen_kappa_score(y_ts, y_pr ,weights='quadratic')# des char 1,4, compon = 5, iter = жок
                                                  # lev 4 descr, title
                                                  # data.drop(['target']
                                                  # lev adding count features one, two, three, four
cohen_kappa_score(y_ts, y_pr ,weights='quadratic')# des char 1,4, compon = 5, iter = жок
                                                  # lev 4 descr, title
                                                  # data.drop(['target']
                                                  # lev adding count features one, two, three, four 
cohen_kappa_score(y_ts, y_pr ,weights='quadratic')# des char 1,4, compon = 5, iter = жок
                                                  # lev 4 + cos 4
                                                  # data.drop(['target',3,4]
                                                  # lev adding count features one, two, three, four 
cohen_kappa_score(y_ts, y_pr ,weights='quadratic')# des char 1,4, compon = 5, iter = жок
                                                  # lev 4 + cos 4
cohen_kappa_score(y_ts, y_pr ,weights='quadratic')# des char 1,4, compon = 5
                                                  # lev 4 + cos 4
cohen_kappa_score(y_ts, y_pr ,weights='quadratic')# des char 1,4, compon = 10
                                                  # lev 4 + cos 4
cohen_kappa_score(y_ts, y_pr ,weights='quadratic')# des char 1,4, compon = 10
                                                  # lev 4
cohen_kappa_score(y_ts, y_pr ,weights='quadratic')# des char 1,4, compon = 10
                                                  # lev 4 + cos 6
cohen_kappa_score(y_ts, y_pr ,weights='quadratic')# des char 1,4, compon = 10
                                                  # lev 3 + cos 6
cohen_kappa_score(y_ts, y_pr ,weights='quadratic')# des char 1,4, compon = 10
                                                  # lev 3 
X.columns
feature_imp = dict(zip(list(X.columns), list(xgb.feature_importances_)))
sorted_x = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)
from xgboost import plot_importance
def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)
xgb = XGBClassifier()
xgb.fit(X, y, sample_weight=weight)
plot_features(xgb, (14,10))
X.columns


sns.heatmap(confusion_matrix(y_ts, y_pr), annot=True)


