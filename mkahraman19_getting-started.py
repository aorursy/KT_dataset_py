# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from textblob import TextBlob

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn import decomposition, ensemble



import pandas, xgboost, numpy, textblob, string

from keras.preprocessing import text, sequence

from keras import layers, models, optimizers





from warnings import filterwarnings

filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

submission=pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
train.head()
train.target.value_counts()
train.loc[train.target==0][:10]
train.loc[train.target==1][:10]
train["target"].replace(0, value = "positive", inplace = True)

train["target"].replace(1, value = "negative", inplace = True)
train.head()
train.groupby("target").count()
df = pd.DataFrame()

df["text"] = train["text"]

df["label"] = train["target"]
df.head()
#uppercase-lowercase transformation

df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))

#change symbols

df['text'] = df['text'].str.replace('[^\w\s]','')

#numbers

df['text'] = df['text'].str.replace('\d','')

#stopwords

import nltk

#nltk.download('stopwords')

from nltk.corpus import stopwords

sw = stopwords.words('english')

df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))



#delete

dlt = pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]

df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in dlt))

#lemmi

from textblob import Word

#nltk.download('wordnet')

df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) 
df.head()
df.head()
df.iloc[0]
train_x, test_x, train_y, test_y = model_selection.train_test_split(df["text"],

                                                                   df["label"], 

                                                                    random_state = 1)
train_y[0:5]
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)

test_y = encoder.fit_transform(test_y)
train_y[0:5]
test_y[0:5]
vectorizer = CountVectorizer()

vectorizer.fit(train_x)
x_train_count = vectorizer.transform(train_x)

x_test_count = vectorizer.transform(test_x)
x_train_count
vectorizer.get_feature_names()[0:5]
x_train_count.toarray()
#wordlevel
tf_idf_word_vectorizer = TfidfVectorizer()

tf_idf_word_vectorizer.fit(train_x)
x_train_tf_idf_word = tf_idf_word_vectorizer.transform(train_x)

x_test_tf_idf_word = tf_idf_word_vectorizer.transform(test_x)
tf_idf_word_vectorizer.get_feature_names()[0:5]
x_train_tf_idf_word.toarray()
# ngram level tf-idf
tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range = (2,3))

tf_idf_ngram_vectorizer.fit(train_x)
x_train_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(train_x)

x_test_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(test_x)
# characters level tf-idf
tf_idf_chars_vectorizer = TfidfVectorizer(analyzer = "char", ngram_range = (2,3))

tf_idf_chars_vectorizer.fit(train_x)
x_train_tf_idf_chars = tf_idf_chars_vectorizer.transform(train_x)

x_test_tf_idf_chars = tf_idf_chars_vectorizer.transform(test_x)
loj = linear_model.LogisticRegression()

loj_model = loj.fit(x_train_count, train_y)

accuracy = model_selection.cross_val_score(loj_model, 

                                           x_test_count, 

                                           test_y, 

                                           cv = 10).mean()



print("Count Vectors Accuracy:", accuracy)
loj = linear_model.LogisticRegression()

loj_model = loj.fit(x_train_tf_idf_word,train_y)

accuracy = model_selection.cross_val_score(loj_model, 

                                           x_test_tf_idf_word, 

                                           test_y, 

                                           cv = 10).mean()



print("Word-Level TF-IDF Accuracy:", accuracy)
loj = linear_model.LogisticRegression()

loj_model = loj.fit(x_train_tf_idf_ngram,train_y)

accuracy = model_selection.cross_val_score(loj_model, 

                                           x_test_tf_idf_ngram, 

                                           test_y, 

                                           cv = 10).mean()



print("N-GRAM TF-IDF Accuracy:", accuracy)
loj = linear_model.LogisticRegression()

loj_model = loj.fit(x_train_tf_idf_chars,train_y)

accuracy = model_selection.cross_val_score(loj_model, 

                                           x_test_tf_idf_chars, 

                                           test_y, 

                                           cv = 10).mean()



print("CHARLEVEL Accuracy:", accuracy)
nb = naive_bayes.MultinomialNB()

nb_model = nb.fit(x_train_count,train_y)

accuracy = model_selection.cross_val_score(nb_model, 

                                           x_test_count, 

                                           test_y, 

                                           cv = 10).mean()



print("Count Vectors Accuracy:", accuracy)
nb = naive_bayes.MultinomialNB()

nb_model = nb.fit(x_train_tf_idf_word,train_y)

accuracy = model_selection.cross_val_score(nb_model, 

                                           x_test_tf_idf_word, 

                                           test_y, 

                                           cv = 10).mean()



print("Word-Level TF-IDF Accuracy:", accuracy)
nb = naive_bayes.MultinomialNB()

nb_model = nb.fit(x_train_tf_idf_ngram,train_y)

accuracy = model_selection.cross_val_score(nb_model, 

                                           x_test_tf_idf_ngram, 

                                           test_y, 

                                           cv = 10).mean()



print("N-GRAM TF-IDF Accuracy:", accuracy)
nb = naive_bayes.MultinomialNB()

nb_model = nb.fit(x_train_tf_idf_chars,train_y)

accuracy = model_selection.cross_val_score(nb_model, 

                                           x_test_tf_idf_chars, 

                                           test_y, 

                                           cv = 10).mean()



print("CHARLEVEL Accuracy:", accuracy)
rf = ensemble.RandomForestClassifier()

rf_model = rf.fit(x_train_count,train_y)

accuracy = model_selection.cross_val_score(rf_model, 

                                           x_test_count, 

                                           test_y, 

                                           cv = 10).mean()



print("Count Vectors Accuracy:", accuracy)
rf = ensemble.RandomForestClassifier()

rf_model = rf.fit(x_train_tf_idf_word,train_y)

accuracy = model_selection.cross_val_score(rf_model, 

                                           x_test_tf_idf_word, 

                                           test_y, 

                                           cv = 10).mean()



print("Word-Level TF-IDF Accuracy:", accuracy)
rf = ensemble.RandomForestClassifier()

rf_model = loj.fit(x_train_tf_idf_ngram,train_y)

accuracy = model_selection.cross_val_score(rf_model, 

                                           x_test_tf_idf_ngram, 

                                           test_y, 

                                           cv = 10).mean()



print("N-GRAM TF-IDF Accuracy:", accuracy)
rf = ensemble.RandomForestClassifier()

rf_model = loj.fit(x_train_tf_idf_chars,train_y)

accuracy = model_selection.cross_val_score(rf_model, 

                                           x_test_tf_idf_chars, 

                                           test_y, 

                                           cv = 10).mean()



print("CHARLEVEL Accuracy:", accuracy)
xgb = xgboost.XGBClassifier()

xgb_model = xgb.fit(x_train_count,train_y)

accuracy = model_selection.cross_val_score(xgb_model, 

                                           x_test_count, 

                                           test_y, 

                                           cv = 10).mean()



print("Count Vectors Accuracy:", accuracy)
xgb = xgboost.XGBClassifier()

xgb_model = xgb.fit(x_train_tf_idf_word,train_y)

accuracy = model_selection.cross_val_score(xgb_model, 

                                           x_test_tf_idf_word, 

                                           test_y, 

                                           cv = 10).mean()



print("Word-Level TF-IDF Accuracy:", accuracy)
xgb = xgboost.XGBClassifier()

xgb_model = xgb.fit(x_train_tf_idf_ngram,train_y)

accuracy = model_selection.cross_val_score(xgb_model, 

                                           x_test_tf_idf_ngram, 

                                           test_y, 

                                           cv = 10).mean()



print("N-GRAM TF-IDF Accuracy:", accuracy)
xgb = xgboost.XGBClassifier()

xgb_model = xgb.fit(x_train_tf_idf_chars,train_y)

accuracy = model_selection.cross_val_score(xgb_model, 

                                           x_test_tf_idf_chars, 

                                           test_y, 

                                           cv = 10).mean()



print("CHARLEVEL Accuracy:", accuracy)
test
train
df
y_pred= pd.Series(test.text)
v = CountVectorizer()

v.fit(train_x)

y_pred = v.transform(y_pred)
y_pred
y_pred
predictions=pd.DataFrame(loj_model.predict(y_pred))
predictions['target']=predictions[0]
del predictions[0]
predictions
submission['target']=predictions.target
submission
test.head()
submission.to_csv('model_submission.csv', index=False)

submission.describe()