!pip install jcopml
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from jcopml.pipeline import num_pipe, cat_pipe
# from jcopml.utils import save_model, load_model
from jcopml.plot import plot_missing_value
from jcopml.feature_importance import mean_score_decrease
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
df.head()

# why "id" is not set as index? because it is treated as list in the phase of model prediction to test dataset
# plot missing value
plot_missing_value(df, return_df=True)

# drop too missing values data
df.drop(columns=["keyword", "location"], inplace=True)
plot_missing_value(df, return_df=True)
print('the data is ready to be engineered')
df.head(10)
# total of words of old df
total_words_old_df = df.text.apply(lambda x: len(x.split(" "))).sum()
print(total_words_old_df)
import nltk 
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from string import punctuation
for i in range(len(df.text)):
    print(df.text[i])
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

df1 = df.text.apply(str).apply(lambda x:clean_text(x))
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
    x_word_tokens_removed_punctuation_removed_sw = [w for w in x_word_tokens_removed_punctuations if w not in stopwords.words('english')]
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
# new df
df['text'] = df1_clean_text_series
df.head(10)
# total of words of old df vs new df 

total_words_new_df = df.text.apply(lambda x: len(x.split(" "))).sum()

print("old df: ", total_words_old_df, "words")
print("new df: ", total_words_new_df, "words")
print("text processing has reduced the number of words by", round((total_words_old_df-total_words_new_df)/total_words_old_df*100), "%")
X = df.text
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
# Check the imbalance dataset
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

model_logreg_tfidf = RandomizedSearchCV(pipeline, rsp.logreg_params, cv=5, n_iter=50, n_jobs=-1, verbose=1, random_state=42)
model_logreg_tfidf.fit(X_train, y_train)

print(model_logreg_tfidf.best_params_)
print(model_logreg_tfidf.score(X_train, y_train), model_logreg_tfidf.best_score_, model_logreg_tfidf.score(X_test, y_test))
pipeline = Pipeline([
    ('prep', CountVectorizer(tokenizer=word_tokenize, ngram_range=(1, 3))),
    ('algo', LogisticRegression(solver='lbfgs', n_jobs=-1, random_state=42))
])

# model = RandomizedSearchCV(pipeline, rsp.logreg_params, cv=5, n_iter=50, n_jobs=-1, verbose=1, random_state=42)
model_logreg_bow = RandomizedSearchCV(pipeline, rsp.logreg_params, cv=5, n_jobs=-1, verbose=1)
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
    'algo__max_iter': [7, 8, 9, 10, 11, 12, 13, 14, 15],
    'algo__tol': [0.0001, 0.0002, 0.0003]
}
# model_sgd_bow = GridSearchCV(pipeline, parameter, cv=5, n_jobs=-1, verbose=1)
model_sgd_bow = RandomizedSearchCV(pipeline, parameter, cv=5, n_jobs=-1, verbose=1, random_state=42)
model_sgd_bow.fit(X_train, y_train)


print(model_sgd_bow.best_params_)
print(model_sgd_bow.score(X_train, y_train), model_sgd_bow.best_score_, model_sgd_bow.score(X_test, y_test))
pipeline = Pipeline([
    ('prep', TfidfVectorizer(tokenizer=word_tokenize, ngram_range=(1, 3))),
    ('algo', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
])


parameter = {
    'algo__loss': ['hinge', 'log', 'modified_huber', 'perceptron'],
    'algo__penalty': ['l2', 'l1', 'elasticnet'],
    'algo__alpha': [0.0001, 0.0002, 0.0003], 
    'algo__max_iter': [7, 8, 9, 10, 11, 12, 13, 14, 15],
    'algo__tol': [0.0001, 0.0002, 0.0003]
}
model_sgd_tfidf = RandomizedSearchCV(pipeline, parameter, cv=5, n_jobs=-1, verbose=1, random_state=42)
# modelt_sgd_fidf = GridSearchCV(pipeline, parameter, cv=5, n_jobs=-1, verbose=1)
model_sgd_tfidf.fit(X_train, y_train)


print(model_sgd_tfidf.best_params_)
print(model_sgd_tfidf.score(X_train, y_train), model_sgd_tfidf.best_score_, model_sgd_tfidf.score(X_test, y_test))
from jcopml.utils import save_model
# save_model(model, "disaster_tweet_v1.pkl")
# save_model(model, "disaster_tweet_v2.pkl")
# save_model(model, "disaster_tweet_v3.pkl")
# save_model(model, "disaster_tweet_v4.pkl")
# save_model(model, "disaster_tweet_v5.pkl")
# save_model(model, "disaster_tweet_v6.pkl")
# save_model(model, "disaster_tweet_v7.pkl")
# save_model(model, "disaster_tweet_v8.pkl")
# save_model(model, "disaster_tweet_v9.pkl")
# save_model(model, "disaster_tweet_v10.pkl")
save_model(model, "disaster_tweet_v11.pkl")
df_submit = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
df_submit.drop(columns=["keyword", "location"], inplace=True)
df_submit_id_list = df_submit.id.values.tolist()
df_submit_text_list = df_submit.text.values.tolist()
df_submit.head()
print("df_submit.id: ", len(df_submit.id))
print("df_submit.text: ", len(df_submit.text))
print("df_submit_id_list: ", len(df_submit_id_list))
print("df_submit_text_list: ", len(df_submit_text_list))
target = modeltfidf.predict(df_submit_text_list)
target
print(type(target))
print(len(target))
df_submit_final = pd.DataFrame({
    "id": df_submit_id_list,
    "target": target
})
# df_submit_final.set_index('id', inplace=True)
df_submit_final.head()
# df_submit_final.to_csv("disaster_tweet_v1.csv")
# df_submit_final.to_csv("disaster_tweet_v2.csv")
# df_submit_final.to_csv("disaster_tweet_v3.csv")
# df_submit_final.to_csv("disaster_tweet_v4.csv")
# df_submit_final.to_csv("disaster_tweet_v5.csv")
# df_submit_final.to_csv("disaster_tweet_v6.csv") #numeric removal, tfidf
# df_submit_final.to_csv("disaster_tweet_v7.csv") #numeric removal, bow, n_iter default, random_state default
# df_submit_final.to_csv("disaster_tweet_v8.csv") #SGD Classifier
# df_submit_final.to_csv("disaster_tweet_v9.csv") #SGD Classifier
# df_submit_final.to_csv("disaster_tweet_v10.csv") #SGD Classifier
# df_submit_final.to_csv("disaster_tweet_v11.csv") #SGD Classifier
df_submit_final.to_csv("disaster_tweet_v12.csv") #SGD Classifier + tfidf
import pickle
pickle.dump(model_logreg_tfidf, open("model_logreg_tfidf.pkl", 'wb'))
pickle.dump(model_logreg_bow, open("model_logreg_bow.pkl", 'wb'))
pickle.dump(model_sgd_tfidf, open("model_sgd_tfidf.pkl", 'wb'))
pickle.dump(model_sgd_bow, open("model_sgd_bow.pkl", 'wb'))
# model = pickle.load(open("knn.pkl", 'rb'))