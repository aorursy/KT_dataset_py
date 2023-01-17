!pip install jcopml
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from jcopml.pipeline import num_pipe, cat_pipe
from jcopml.utils import save_model, load_model
from jcopml.plot import plot_missing_value
from jcopml.feature_importance import mean_score_decrease
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
df = pd.read_csv("../input/shopee-sentiment-analysis/train.csv", encoding='latin1')
df.head()
len(df)
# plot missing value
# plot_missing_value(df, return_df=True)
# drop too missing values data
# df.drop(columns=["Tweet_ID", "Tanggal_Tweet", "Kandidat", "Aplikasi", "Lokasi User"], inplace=True)

# plot_missing_value(df, return_df=True)
# total of words of old df
total_words_old_df = df.review.apply(lambda x: len(x.split(" "))).sum()
print(total_words_old_df)
import nltk 
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from string import punctuation

sw = stopwords.words("indonesian") + stopwords.words("english")
for i in range(len(df)):
    print(df.review[i])
    if i == 10:
        break
print("The length of dataframe is", len(df), "rows")
def clean_text(text):
    text = text.lower()
    text = re.sub("\n", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = text.split()
    text = " ".join(text)
    return text

df1 = df.review.apply(str).apply(lambda x:clean_text(x))
for i in range(len(df1)):
    print(df1[i])
    if i == 10:
        break
print("The length of dataframe is", len(df1), "rows")
df1_clean_text = []
for i in range(len(df1)):
    x = df1[i]
    # x_sent_token = sent_tokenize(x)
    # x_sent_token
    x_word_tokens = word_tokenize(x)
    # x_word_tokens
#     print(x)
    
#     print(df1[i])
    
    # punctuation removal
    x_word_tokens_removed_punctuations = [w for w in x_word_tokens if w not in punctuation]
#     print(x_word_tokens_removed_punctuations, "punctuation")
    
    # numeric removal
    x_word_tokens_removed_punctuations = [w for w in x_word_tokens_removed_punctuations if w.isalpha()]
#     print(x_word_tokens_removed_punctuations, "numeric")
    
    # stopwords removal
    x_word_tokens_removed_punctuation_removed_sw = [w for w in x_word_tokens_removed_punctuations if w not in sw]
#     print(x_word_tokens_removed_punctuation_removed_sw, "stopwords")

    # rejoining the words into one string/sentence as inputted before being tokenized
    x_word_tokens_removed_punctuation_removed_sw = " ".join(x_word_tokens_removed_punctuation_removed_sw)
#     print(x_word_tokens_removed_punctuation_removed_sw)
    
    df1_clean_text.append(x_word_tokens_removed_punctuation_removed_sw)
# text vs processed text
for i,j in zip(df1[0:10], df1_clean_text[0:10]):
    print(i)
    print(j)
    print()
# list (df1_clean_text) to series (df1_clean_text_series)

# list
print(type(df1_clean_text))
print(len(df1_clean_text))

# converting list to pandas series
df1_clean_text_series = pd.Series(df1_clean_text)

print(type(df1_clean_text_series))
print(len(df1_clean_text_series))
# new series
df1_clean_text_series.head()
# df.drop(columns=['Isi_Tweet'], inplace=True)
# new df
df['review'] = df1_clean_text_series
df.head(10)
# total of words of old df vs new df 

total_words_new_df = df.review.apply(lambda x: len(x.split(" "))).sum()

print("old df: ", total_words_old_df, "words")
print("new df: ", total_words_new_df, "words")
print("text processing has reduced the number of words by", round((total_words_old_df-total_words_new_df)/total_words_old_df*100), "%")
X = df.review
y = df.rating

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
# Check the  dataset balance
y_train.value_counts() / len(y_train) *100

# the train data is more or less balance
X_train.head(), X_test.head(), y_train.head(), y_test.head()
X_train.head(11)
import seaborn as sns
import matplotlib.pyplot as plt
y_train.shape, y_test.shape
sns.set(style="darkgrid")
sns.countplot(x=y_train)
plt.title("y_train");
sns.set(style="darkgrid")
sns.countplot(x=y_test)
plt.title("y_test");
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from jcopml.tuning import random_search_params as rsp
from jcopml.tuning import random_search_params as gsp

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
pipeline = Pipeline([
    ('prep', TfidfVectorizer(tokenizer=word_tokenize, ngram_range=(1, 3))),
    ('algo', LogisticRegression(solver='lbfgs', n_jobs=-1, random_state=42))
])

model_logreg_tfidf = RandomizedSearchCV(pipeline, rsp.logreg_params, cv=5, n_iter=30, n_jobs=-1, verbose=1, random_state=42)
model_logreg_tfidf.fit(X_train, y_train)

print(model_logreg_tfidf.best_params_)
print(model_logreg_tfidf.score(X_train, y_train), model_logreg_tfidf.best_score_, model_logreg_tfidf.score(X_test, y_test))


# for the purpose of saving model
# pipeline = Pipeline([
#     ('prep', TfidfVectorizer(tokenizer=word_tokenize, ngram_range=(1, 3))),
#     ('algo', LogisticRegression(solver='lbfgs', C=17.71884735480683, fit_intercept= False, n_jobs=-1, verbose=1, random_state=42))
# ])

# model_logreg = pipeline
# model_logreg.fit(X_train, y_train)

# # print(model_logreg.best_params_)
# # print(model_logreg.score(X_train, y_train), model_logreg.best_score_, model_logreg.score(X_test, y_test))
pipeline = Pipeline([
    ('prep', CountVectorizer(tokenizer=word_tokenize, ngram_range=(1, 3))),
    ('algo', LogisticRegression(solver='lbfgs', n_jobs=-1, random_state=42))
])

model_logreg_bow = RandomizedSearchCV(pipeline, rsp.logreg_params, cv=5, n_iter=30, n_jobs=-1, verbose=1, random_state=42)
model_logreg_bow.fit(X_train, y_train)

print(model_logreg_bow.best_params_)
print(model_logreg_bow.score(X_train, y_train), model_logreg_bow.best_score_, model_logreg_bow.score(X_test, y_test))
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from jcopml.tuning import random_search_params as rsp
from jcopml.tuning import random_search_params as gsp

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
pipeline = Pipeline([
    ('prep', CountVectorizer(tokenizer=word_tokenize, ngram_range=(1, 3))),
    ('algo', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
])


parameter = {
    'algo__loss': ['hinge', 'log', 'modified_huber', 'perceptron'],
    'algo__penalty': ['l2', 'l1', 'elasticnet'],
    'algo__alpha': [0.0001, 0.0002, 0.0003], 
    'algo__max_iter': [6, 7, 8, 9, 10, 11],
    'algo__tol': [0.0001, 0.0002, 0.0003]
}
# model = GridSearchCV(pipeline, parameter, cv=5, n_jobs=-1, verbose=1)
model_sgd_bow = RandomizedSearchCV(pipeline, parameter, cv=50, n_jobs=-1, verbose=1)
model_sgd_bow.fit(X_train, y_train)


print(model_sgd_bow.best_params_)
print(model_sgd_bow.score(X_train, y_train), model_sgd_bow.best_score_, model_sgd_bow.score(X_test, y_test))
pipeline = Pipeline([
    ('prep', TfidfVectorizer(tokenizer=word_tokenize, ngram_range=(1, 3))),
    ('algo', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
])


parameter = {
    'algo__loss': ['hinge', 'log', 'modified_huber'],
    'algo__penalty': ['l2', 'l1'],
    'algo__alpha': [0.0001, 0.0002], 
    'algo__max_iter': [5, 6, 7, 8, 9, 10],
    'algo__tol': [0.0001, 0.0002, 0.0003]
}
model_sgd_tfidf = RandomizedSearchCV(pipeline, parameter, cv=50, n_jobs=-1, verbose=1, random_state=42)
model_sgd_tfidf.fit(X_train, y_train)


print(model_sgd_tfidf.best_params_)
print(model_sgd_tfidf.score(X_train, y_train), model_sgd_tfidf.best_score_, model_sgd_tfidf.score(X_test, y_test))
from jcopml.utils import save_model, load_model
# save_model(model_logreg_tfidf, "model_logreg_tfidf.pkl")
save_model(model_logreg_bow, "model_logreg_bow.pkl")
# save_model(model_sgd_tfidf, "model_sgd_tfidf.pkl")
# save_model(model_sgd_bow, "model_sgd_bow.pkl")
# loading model
model_sgd_tfidf = load_model("../input/shopee-product-review-sentiment-model/model_sgd_tfidf.pkl")
model_logreg_bow = load_model("../input/shopee-product-review-sentiment-model/model_logreg_bow.pkl")
import pickle
pickle.dump(model_logreg_tfidf, open("model_logreg_tfidf.pkl", 'wb'))
pickle.dump(model_logreg_bow, open("model_logreg_bow.pkl", 'wb'))
pickle.dump(model_sgd_tfidf, open("model_sgd_tfidf.pkl", 'wb'))
pickle.dump(model_sgd_bow, open("model_sgd_bow.pkl", 'wb'))
df_submit = pd.read_csv("../input/shopee-sentiment-analysis-test-set/test.csv")
# df_submit.drop(columns=["keyword", "location"], inplace=True)
df_submit_review_id_list = df_submit.review_id.values.tolist()
df_submit_review_list = df_submit.review.values.tolist()
df_submit.head()
len(df_submit)
print("df_submit.review_id: ", len(df_submit.review_id))
print("df_submit.review: ", len(df_submit.review))
print("df_submit_review_id_list: ", len(df_submit_review_id_list))
print("df_submit_review_list: ", len(df_submit_review_list))
# logreg_tfidf
# target = model_logreg_tfidf.predict(df_submit_review_list)
# target

# sgd_tfidf
target_sgd_tfidf = model_sgd_tfidf.predict(df_submit_review_list)
target_sgd_tfidf

# logreg_bow
target_logreg_bow = model_logreg_bow.predict(df_submit_review_list)
target_logreg_bow
print(type(target_sgd_tfidf))
print(len(target_sgd_tfidf))
print(type(target_logreg_bow))
print(len(target_logreg_bow))
df_submit_final_sgd_tfidf = pd.DataFrame({
    "review_id": df_submit_review_id_list,
    "rating": target_sgd_tfidf
})

df_submit_final_logreg_bow = pd.DataFrame({
    "review_id": df_submit_review_id_list,
    "rating": target_logreg_bow
})
df_submit_final_sgd_tfidf.head(), df_submit_final_logreg_bow.head()
# df_submit_final_sgd_tfidf.set_index('review_id', inplace=True)
# df_submit_final_logreg_bow.set_index('review_id', inplace=True)
df_submit_final_sgd_tfidf.to_csv("shopee_product_sentiment_1.csv") #linear svm
df_submit_final_logreg_bow.to_csv("shopee_product_sentiment_2.csv") #logistic Regression
import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#Avoid warning messages
import warnings
warnings.filterwarnings("ignore")

#plotly libraries
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

from sklearn.feature_extraction.text import CountVectorizer
train_set = pd.read_csv("../input/shopee-sentiment-analysis/train.csv")
test_set = pd.read_csv("../input/shopee-sentiment-analysis/test.csv")
data_set = train_set.append(test_set).reset_index(drop=True)
data_set
all_text = '||'.join(data_set['review']) #unique characters
chars = list(set(all_text))
# get special characters
chars[:50], len(chars)
# get the list
chars_count = {}
def append_chars_count(text):
    for char in list(set(text)):
        chars_count[char] = chars_count.get(char, 0) + 1
        
data_set['review'].map(append_chars_count);
len(chars_count), len(chars)
# Convert into DataFrame
df_chars = pd.DataFrame()
df_chars['char'] = list(chars_count.keys())
df_chars['count'] = list(chars_count.values())
df_chars = df_chars.sort_values("count", ascending = False).reset_index(drop = True)
df_chars
df_chars['count'].value_counts().head(20)
df_chars.loc[(df_chars['count'] >= 2) & (df_chars['count'] < 4)].sample(10, random_state = 100)
