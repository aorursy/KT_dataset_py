# Data Analysis
import pandas as pd
import numpy as np

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
#import umap.plot
#from yellowbrick.text import UMAPVisualizer

# Text Processing
import re
import itertools
import spacy
import string
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
import en_core_web_sm
from collections import Counter

### Dimensionality reduction and embedding
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import umap

# Machine Learning packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.cluster as cluster
from sklearn.preprocessing import LabelEncoder

# Ignore noise warning
import warnings
warnings.filterwarnings("ignore")

# Export data
import pickle as pkl
from scipy import sparse
from numpy import asarray
from numpy import savetxt

pd.set_option("display.max_column", None)
mbti_df = pd.read_csv("../input/nlp-to-predict-myers-briggs-personality-type/data/mbti_1.csv")
type = ["type"]
posts = ["posts"]
columns = [*type, *posts]
mbti_df_raw = mbti_df
mbti_df_raw[type] = mbti_df[type].fillna("")
mbti_df_raw[posts] = mbti_df[posts].fillna("")
mbti_df_raw.head()
def clean_url(str_text_raw):
    """This function eliminate a string URL in a given text"""
    str_text = re.sub("url_\S+", "", str_text_raw)
    str_text = re.sub("email_\S+", "", str_text)
    str_text = re.sub("phone_\S+", "", str_text)
    return(re.sub("http[s]?://\S+", "", str_text))
    
def clean_punctuation(str_text_raw):
    """This function replace some of the troublemaker puntuation elements in a given text"""
    return(re.sub("[$\(\)/|{|\}#~\[\]^#;:!?¿]", " ", str_text_raw))

def clean_unicode(str_text_raw):
    """This function eliminate non-unicode text"""
    str_text = re.sub("&amp;", "", str_text_raw)
    return(re.sub(r"[^\x00-\x7F]+"," ", str_text))
                      
def clean_dot_words(str_text_raw):
    """This function replace dots between words"""
    return(re.sub(r"(\w+)\.+(\w+)", r"\1 \2",str_text_raw))

def clean_text(str_text_raw):
    """This function sets the text to lowercase and applies previous cleaning functions """
    str_text = str_text_raw.lower()
    str_text = clean_dot_words(clean_punctuation(clean_unicode(clean_url(str_text))))
    return(str_text)
tokens_to_drop=["+"]

def string_to_token(string, str_pickle = None):
    """
    This function takes a sentence and returns the list of tokens and all their information
    * Text: The original text of the lemma.
    * Lemma: Lemma.
    * Orth: The hash value of the lemma.
    * is alpha: Does the lemma consist of alphabetic characters?
    * is digit: Does the lemma consist of digits?
    * is_title: Is the token in titlecase? 
    * is_punct: Is the token punctuation?
    * is_space: Does the token consist of whitespace characters?
    * is_stop: Is the token part of a “stop list”?
    * is_digit: Does the token consist of digits?
    * lang: Language of the token
    * tag: Fine-grained part-of-speech. The complete list is in: 
    https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html, also using: spacy.explain("RB")
    * pos: Coarse-grained part-of-speech.
    * has_vector: A boolean value indicating whether a word vector is associated with the token.
    * vector_norm: The L2 norm of the token’s vector representation.
    * is_ovv: """
    doc = nlp(string)
    l_token = [[token.text, token.lemma_, token.orth, token.is_alpha, token.is_digit, token.is_title, token.lang_, 
        token.tag_, token.pos_, token.has_vector, token.vector_norm, token.is_oov]
        for token in doc if not token.is_punct | token.is_space | token.is_stop | token.is_digit | token.like_url 
               | token.like_num | token.like_email & token.is_oov]
    pd_token = pd.DataFrame(l_token, columns=["text", "lemma", "orth", "is_alpha", "is_digit", "is_title", "language",
                                          "tag", "part_of_speech", "has_vector", "vector_norm", "is_oov"])
    #drop problematic tokens
    pd_token = pd_token[~pd_token["text"].isin(tokens_to_drop)]
    #Convert plural text to singular
    pd_token["text_to_singular"] = np.where(pd_token["tag"].isin(["NNPS", "NNS"]), pd_token["lemma"], pd_token["text"])
    #if(str_pickle!=None):
    #    pd_token.to_pickle(f"../input/nlp-to-predict-myers-briggs-personality-type/data/output_pickles/{str_pickle}.pkl")
    del l_token
    return(pd_token)

def apply_cleaning(string):
    """
    This function takes a sentence and returns a clean text
    """
    doc = nlp(clean_text(string))
    l_token = [token.text for token in doc if not token.is_punct | token.is_space | token.is_stop | 
               token.is_digit | token.like_url | token.like_num | token.like_email & token.is_oov]
    return " ".join(l_token)

def apply_lemma(string):
    """
    This function takes a sentence and returns a clean text
    """
    doc = nlp(clean_text(string))
    l_token = [token.lemma_ for token in doc if not token.is_punct | token.is_space | token.is_stop | 
               token.is_digit | token.like_url | token.like_num | token.like_email & token.is_oov]
    return " ".join(l_token)

def list_to_bow(l_words):
    """
    This function takes a list of words and create the bag of words ordered by desc order
    """
    cv = CountVectorizer(l_words)
    # show resulting vocabulary; the numbers are not counts, they are the position in the sparse vector.
    count_vector=cv.fit_transform(l_words)
    word_freq = Counter(l_words)
    print(f"Bag of words size: {count_vector.shape}\nUnique words size: {len(word_freq)}")
    dict_word_freq = dict(word_freq.most_common())
    return(dict_word_freq)
mbti_df_clean = pd.DataFrame(mbti_df_raw[["type", "posts"]])
for c in columns:
    mbti_df_clean[c] = mbti_df_raw[c].apply(lambda row: clean_text(row))
mbti_df_clean["posts"] = mbti_df_raw[posts].apply(lambda x: " ".join(x), axis=1)
mbti_df_clean.head()
raise SystemExit("This is a very consumming memory process, with average wall time: ~ 20 min. If you don't want to wait please go to the next step")
nlp = spacy.load("en_core_web_sm", disable = ["ner", "parser"]) 
nlp.max_length = 33000000
mbti_df_clean.shape
mbti_df_clean_first = mbti_df_clean.iloc[:2169]
mbti_df_clean_second = mbti_df_clean[2169:4338]
mbti_df_clean_third = mbti_df_clean.iloc[4338:6507]
mbti_df_clean_fourth = mbti_df_clean.iloc[6507:8675]
%%time
for column in columns:    
    str_bow_column_first = " ".join(mbti_df_clean_first[column])
    pd_token_first = string_to_token(str_bow_column_first, f"token_first_{column}")
    print(f"Length of {column} column: {len(str_bow_column_first)}")
    print(f"Number of tokens created: {pd_token_first.shape[0]}\n")
%%time
for column in columns:    
    str_bow_column_second = " ".join(mbti_df_clean_second[column])
    pd_token_second = string_to_token(str_bow_column_second, f"token_second_{column}")
    print(f"Length of {column} column: {len(str_bow_column_second)}")
    print(f"Number of tokens created: {pd_token_second.shape[0]}\n")
%%time
for column in columns:    
    str_bow_column_third = " ".join(mbti_df_clean_third[column])
    pd_token_third = string_to_token(str_bow_column_third, f"token_third_{column}")
    print(f"Length of {column} column: {len(str_bow_column_third)}")
    print(f"Number of tokens created: {pd_token_third.shape[0]}\n")
%%time
for column in columns:    
    str_bow_column_fourth = " ".join(mbti_df_clean_fourth[column])
    pd_token_fourth = string_to_token(str_bow_column_fourth, f"token_fourth_{column}")
    print(f"Length of {column} column: {len(str_bow_column_fourth)}")
    print(f"Number of tokens created: {pd_token_fourth.shape[0]}\n")
%%time
pd_token_first = pd.DataFrame(columns=["column", "text", "lemma", "orth", "is_alpha", "is_digit", "is_title", "language", "tag", 
                                 "part_of_speech", "has_vector", "vector_norm", "is_oov", "text_to_singular"])
for column in columns:
    pd_temp = pd.read_pickle(f"../input/nlp-to-predict-myers-briggs-personality-type/data/output_pickles/token_first_{column}.pkl") #Modified
    pd_temp["column"] = column
    print(f"Loading {column} info with {pd_temp.shape[0]} rows")
    pd_token_first = pd.concat([pd_token_first, pd_temp])
print(f"Total rows loaded: {pd_token_first.shape[0]}")
%%time
pd_token_second = pd.DataFrame(columns=["column", "text", "lemma", "orth", "is_alpha", "is_digit", "is_title", "language", "tag", 
                                 "part_of_speech", "has_vector", "vector_norm", "is_oov", "text_to_singular"])
for column in columns:
    pd_temp = pd.read_pickle(f"../input/nlp-to-predict-myers-briggs-personality-type/data/output_pickles/token_second_{column}.pkl") #Modified
    pd_temp["column"] = column
    print(f"Loading {column} info with {pd_temp.shape[0]} rows")
    pd_token_second = pd.concat([pd_token_second, pd_temp])
print(f"Total rows loaded: {pd_token_second.shape[0]}")
%%time
pd_token_third = pd.DataFrame(columns=["column", "text", "lemma", "orth", "is_alpha", "is_digit", "is_title", "language", "tag", 
                                 "part_of_speech", "has_vector", "vector_norm", "is_oov", "text_to_singular"])
for column in columns:
    pd_temp = pd.read_pickle(f"../input/nlp-to-predict-myers-briggs-personality-type/data/output_pickles/token_third_{column}.pkl") #Modified
    pd_temp["column"] = column
    print(f"Loading {column} info with {pd_temp.shape[0]} rows")
    pd_token_third = pd.concat([pd_token_third, pd_temp])
print(f"Total rows loaded: {pd_token_third.shape[0]}")
%%time
pd_token_fourth = pd.DataFrame(columns=["column", "text", "lemma", "orth", "is_alpha", "is_digit", "is_title", "language", "tag", 
                                 "part_of_speech", "has_vector", "vector_norm", "is_oov", "text_to_singular"])
for column in columns:
    pd_temp = pd.read_pickle(f"../input/nlp-to-predict-myers-briggs-personality-type/data/output_pickles/token_fourth_{column}.pkl") #Modified
    pd_temp["column"] = column
    print(f"Loading {column} info with {pd_temp.shape[0]} rows")
    pd_token_fourth = pd.concat([pd_token_fourth, pd_temp])
print(f"Total rows loaded: {pd_token_fourth.shape[0]}")
pd_token_first.head()

pd_token_first.tail()
mbti_df_clean['type_clean'] = mbti_df_clean['type'].apply(lambda x: apply_cleaning(x))
mbti_df_clean['posts_clean']   = mbti_df_clean['posts'].apply(lambda x: apply_cleaning(x))
mbti_df_clean['type_lemma'] = mbti_df_clean['type'].apply(lambda x: apply_lemma(x))
mbti_df_clean['posts_lemma']   = mbti_df_clean['posts'].apply(lambda x: apply_lemma(x))
mbti_df_clean.shape
#mbti_df_clean.to_pickle('data/output_pickles/mbti_clean_text.pkl')
"""
This pickle is too heavy to upload it in the GitHub repository and generating it in the previous cells is quite demmanding. 
"""
#mbti_df_clean = pd.read_pickle("data/output_pickles/mbti_clean_text.pkl")
mbti_df_clean.head()
mbti_text = mbti_df[["type","posts"]].copy()
mbti_text = mbti_text.fillna("")
text_columns = mbti_text[["type"]]
text_columns["text"] = mbti_text.iloc[:,1:].apply(lambda row: " ".join(row.values.astype(str)), axis=1)
text_columns.head()
text_columns = pd.DataFrame()
text_columns["type"] = mbti_df_clean[["type_lemma"]].apply(lambda row: " ".join(row.values.astype(str)), axis=1)
text_columns["text"] = mbti_df_clean[["posts_lemma"]].apply(lambda row: " ".join(row.values.astype(str)), axis=1)
text_columns.head()
text_columns['text'].isnull().sum()
corpus = text_columns['text']
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(corpus)
dictionary = dict(zip(vectorizer.get_feature_names(), tfidf))
tfidf.shape
#sparse.save_npz("data/output_sparse/tfidf.npz", tfidf)
tfidf_df = pd.DataFrame(tfidf)
#tfidf_df.to_csv("data/output_csv/tfidf_df.csv")
tfidf_df.head()
possible_types= ["infj", "entp", "intp", "intj", "entj", "enfj", "infp", "enfp", "isfp", "istp", "isfj", "istj", "estp", "esfp", "estj", "esfj"]
lab_encoder = LabelEncoder().fit(possible_types)
def encode_personalities(text_columns):

    list_personality = []
    len_data = len(text_columns)
    i=0
    
    for row in text_columns.iterrows():
        i+=1
        if i % 500 == 0:
            print("%s | %s rows" % (i, len_data))

        ##### Remove and clean comments
        type_labelized = lab_encoder.transform([row[1].type])[0]
        list_personality.append(type_labelized)

    #del data
    list_personality = np.array(list_personality)
    return list_personality

list_personality = encode_personalities(text_columns)
#savetxt("data/output_csv/personality.csv", list_personality, delimiter=",")
list_personality.shape
svd = TruncatedSVD(n_components=100, n_iter=10, random_state=42)
svd_vec = svd.fit_transform(tfidf)

print("TSNE")
X_tsne = TSNE(n_components=3, verbose=1, perplexity=40).fit_transform(svd_vec)
svd_vec.shape
np.amin(svd_vec)
svd_vec_positive = svd_vec-np.amin(svd_vec)
text_columns_svd_vec = pd.DataFrame(svd_vec_positive)
col = list_personality

plt.figure(0, figsize=(18,10))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=col, cmap=plt.get_cmap('tab20') , s=12)
#plt.savefig("images/output_images/TSNE0.png")

plt.figure(1, figsize=(18,10))
plt.scatter(X_tsne[:,0], X_tsne[:,2], c=col, cmap=plt.get_cmap('tab20') , s=12)
#plt.savefig("images/output_images/TSNE1.png")

plt.figure(2, figsize=(18,10))
plt.scatter(X_tsne[:,1], X_tsne[:,2], c=col, cmap=plt.get_cmap('tab20') , s=12)
#plt.savefig("images/output_images/TSNE2.png")

sns.set_context("talk")
plt.show()
embedding = umap.UMAP(metric='hellinger', random_state=42).fit_transform(tfidf)
np.amin(embedding)
embedding_positive = embedding-np.amin(embedding)
text_columns_umap = pd.DataFrame(embedding_positive)
plt.figure(figsize=(18,10))
plt.scatter(embedding_positive[:, 0], embedding_positive[:, 1], c=col, cmap='Spectral', s=12)


sns.set_context("talk")
#plt.savefig("images/output_images/UMAP_embedding_positive.png")
plt.show()
embedding_svd = umap.UMAP(metric='hellinger', random_state=42).fit_transform(svd_vec_positive)
np.amin(embedding_svd)
embedding_svd_positive = embedding_svd-np.amin(embedding_svd)
text_columns_umap_svd = pd.DataFrame(embedding_svd_positive)
plt.figure(figsize=(18,10))
plt.scatter(embedding_svd_positive[:, 0], embedding_svd_positive[:, 1], c=col, cmap='Spectral', s=12)


sns.set_context("talk")
#plt.savefig("images/output_images/UMAP_embedding_svd_positive.png")
plt.show()
def var_row(row):
    lst = []
    for word in row.split("|||"):
        lst.append(len(word.split()))
    return np.var(lst)

mbti_df["words_per_comment"] = mbti_df["posts"].apply(lambda x: len(x.split())/50)
mbti_df["variance_of_word_counts"] = mbti_df["posts"].apply(lambda x: var_row(x))
type_dummies = pd.get_dummies(text_columns["type"])
text_columns.drop(["text"], axis=1, inplace=True)
text_columns = pd.concat([text_columns, mbti_df["words_per_comment"], mbti_df["variance_of_word_counts"], 
                          type_dummies], axis=1,levels=None ,sort=False)
map1 = {"i": 0, "e": 1}
map2 = {"n": 0, "s": 1}
map3 = {"t": 0, "f": 1}
map4 = {"j": 0, "p": 1}
text_columns["i-e"] = text_columns["type"].astype(str).str[0]
text_columns["i-e"] = text_columns["i-e"].map(map1)
text_columns["n-s"] = text_columns["type"].astype(str).str[1]
text_columns["n-s"] = text_columns["n-s"].map(map2)
text_columns["t-f"] = text_columns["type"].astype(str).str[2]
text_columns["t-f"] = text_columns["t-f"].map(map3)
text_columns["j-p"] = text_columns["type"].astype(str).str[3]
text_columns["j-p"] = text_columns["j-p"].map(map4)
text_columns.head()
tfidf.shape
tfidf_T = np.transpose(tfidf)
tfidf_T.shape
list_personality.shape
train_array_T = sparse.vstack((list_personality, tfidf_T), format="csr")
train_array_types = np.transpose(train_array_T)
train_array_types.shape
#sparse.save_npz("data/output_sparse/train_array_types.npz", train_array_types)
dimensions_array = text_columns[["i-e", "n-s", "t-f", "j-p"]].to_numpy()
dimensions_array.shape
#savetxt("data/output_csv/dimensions.csv", dimensions_array, delimiter=",")
train_array_dimensions = sparse.hstack((dimensions_array, tfidf), format="csr")
train_array_dimensions.shape
#sparse.save_npz("data/output_sparse/train_array_dimensions.npz", train_array_dimensions)
#text_columns.drop(["type"], axis=1, inplace=True)
result_svd_vec_dimensions = pd.concat([text_columns, text_columns_svd_vec], axis=1,levels=None ,sort=False)
result_svd_vec_dimensions.drop(["enfj", "enfp", "entj", "entp", "esfj", "esfp", "estj", "estp","infj", "infp", "intj", 
                             "intp", "isfj", "isfp", "istj", "istp"], axis=1, inplace=True)
#result_svd_vec_dimensions.to_csv("data/output_csv/result_svd_vec_dimensions.csv")
result_svd_vec_dimensions.head()
result_svd_vec_types = pd.concat([text_columns, text_columns_svd_vec], axis=1,levels=None ,sort=False)
result_svd_vec_types.drop(["i-e", "n-s", "t-f", "j-p"], axis=1, inplace=True)
#result_svd_vec_types.to_csv("data/output_csv/result_svd_vec_types.csv")
result_svd_vec_types.head()
result_umap_dimensions = pd.concat([text_columns, text_columns_umap], axis=1,levels=None ,sort=False)
result_umap_dimensions.drop(["enfj", "enfp", "entj", "entp", "esfj", "esfp", "estj", "estp","infj", "infp", "intj", 
                             "intp", "isfj", "isfp", "istj", "istp"], axis=1, inplace=True)
#result_umap_dimensions.to_csv("data/output_csv/result_umap_dimensions.csv")
result_umap_dimensions.head()
result_umap_types = pd.concat([text_columns, text_columns_umap], axis=1,levels=None ,sort=False)
result_umap_types.drop(["i-e", "n-s", "t-f", "j-p"], axis=1, inplace=True)
#result_umap_types.to_csv("data/output_csv/result_umap_types.csv")
result_umap_types.head()
result_umap_dimensions = pd.concat([text_columns, text_columns_umap], axis=1,levels=None ,sort=False)
result_umap_dimensions.drop(["enfj", "enfp", "entj", "entp", "esfj", "esfp", "estj", "estp","infj", "infp", "intj", 
                             "intp", "isfj", "isfp", "istj", "istp"], axis=1, inplace=True)
#result_umap_dimensions.to_csv("data/output_csv/result_umap_dimensions.csv")
result_umap_dimensions.head()
result_umap_types = pd.concat([text_columns, text_columns_umap], axis=1,levels=None ,sort=False)
result_umap_types.drop(["i-e", "n-s", "t-f", "j-p"], axis=1, inplace=True)
#result_umap_types.to_csv("data/output_csv/result_umap_types.csv")
result_umap_types.head()
result_umap_svd_dimensions = pd.concat([text_columns, text_columns_umap_svd], axis=1,levels=None ,sort=False)
result_umap_svd_dimensions.drop(["enfj", "enfp", "entj", "entp", "esfj", "esfp", "estj", "estp","infj", "infp", "intj", 
                             "intp", "isfj", "isfp", "istj", "istp"], axis=1, inplace=True)
#result_umap_svd_dimensions.to_csv("data/output_csv/result_umap_svd_dimensions.csv")
result_umap_svd_dimensions.head()
result_umap_svd_types = pd.concat([text_columns, text_columns_umap_svd], axis=1,levels=None ,sort=False)
result_umap_svd_types.drop(["i-e", "n-s", "t-f", "j-p"], axis=1, inplace=True)
#result_umap_svd_types.to_csv("data/output_csv/result_umap_svd_types.csv")
result_umap_svd_types.head()