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
df_djia = pd.read_csv("/kaggle/input/stocknews/upload_DJIA_table.csv")
df_combined = pd.read_csv("/kaggle/input/stocknews/Combined_News_DJIA.csv")
df_reddit = pd.read_csv("/kaggle/input/stocknews/RedditNews.csv")
print(df_djia.head())
print(df_combined.head())
print(df_reddit.head())
import nltk
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords 
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
punctuation = ["'", ":", "b'", ".", "\"", "]", "[", ".'", ".\"", "?"]

stemmer = SnowballStemmer("english")
for p in punctuation:
    stop_words.add(p)
tokenizer = WordPunctTokenizer()
def custom_tokenize(text):
    tokens = tokenizer.tokenize(str(text))
    # filtered_tokens = [stemmer.stem(t.lower()) for t in tokens if not t in stop_words and len(t) > 1]
    filtered_tokens = [t.lower() for t in tokens if not t in stop_words and len(t) > 1]
    return " ".join(filtered_tokens)
    
print(custom_tokenize("let's go to mall!! Watch Jim's match"))
cols= df_combined.columns
for col in cols:
    if "Top" in col:
        df_combined[str(col)+"_"]=df_combined[col].apply(custom_tokenize)
        df_combined.drop([col], axis=1, inplace=True)
df_combined.head()
all_headlines = []
for row in range(0,len(df_combined.index)):
    all_headlines.append(' '.join(str(x) for x in df_combined.iloc[row,2:27]))
    
all_headlines[0:4]
len(df_combined),len(all_headlines)
df_combined["combined"]=all_headlines
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
train=df_combined[0:int(len(df_combined)*.80)]
test=df_combined[int(len(df_combined)*.80):]
train_x, test_x, y_train, y_test=train["combined"], test["combined"], train["Label"], test["Label"]
print(train_x.shape, y_train.shape)
tfvectorizer= TfidfVectorizer(max_features = 50000, ngram_range=(2,3))
X_train=tfvectorizer.fit_transform(train_x)
X_test=tfvectorizer.transform(test_x)
print(tfvectorizer.get_feature_names())
print(X_train.shape)
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
print(y_test.shape, X_test.shape)
def all_classifier():
    clf = LogisticRegressionCV(Cs=[0.1,1.0,10.0], cv=5, solver='liblinear').fit(X_train, y_train)
    print("Logistic Classifier", clf.score(X_test, y_test))
    y_preds=clf.predict_proba(X_test)
    y_preds = y_preds[:, 1]
    print("Logistic ROC Curve", roc_auc_score(y_test, y_preds))
    
    rclf = RandomForestClassifier(max_depth=10)
    rclf.fit(X_train, y_train)
    print("RandomForest Classifier",rclf.score(X_test, y_test))
    
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    print("MultinomialNB Classifier", mnb.score(X_test, y_test))
    
    boost = XGBClassifier()
    boost.fit(X_train, y_train)
    y_pred = boost.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("XGBClassifier Accuracy: %.2f%%" % (accuracy * 100.0))

    return clf, rclf, mnb, boost
    
def model_words(model_obj, vectorizer):
    coeffs_list = model_obj.coef_.tolist()[0]
    features = vectorizer.get_feature_names()
    print(len(features), len(coeffs_list))

    
    coeff_df = pd.DataFrame({'Words' : features, 
                        'Coefficient' : coeffs_list})
    coeff_df = coeff_df.sort_values(['Coefficient', 'Words'], ascending=[0, 1])
    print("Positive", coeff_df.head(10))
    print("Negative", coeff_df.tail(10))
a,b,c,d = all_classifier()
model_words(a, tfvectorizer)
c_vectorizer= CountVectorizer(max_features = 50000, ngram_range=(2,3))
X_train=c_vectorizer.fit_transform(train_x)
X_test=c_vectorizer.transform(test_x)
a,b,c,d = all_classifier()
model_words(a, c_vectorizer)
c_vectorizer= CountVectorizer(max_features = 50000, ngram_range=(2,3))
X_train=c_vectorizer.fit_transform(train_x)
X_test=c_vectorizer.transform(test_x)
a,b,c,d = all_classifier()
model_words(a, c_vectorizer)
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
t=Tokenizer()
t.fit_on_texts(df_combined["combined"])
max_length = max([len(d.split(" ")) for d in df_combined["combined"]])
max_length
print(t.document_count)
print(len(t.word_index))
doc_encoded=t.texts_to_sequences(df_combined['combined'].values)
print(doc_encoded[0])
print(t.word_index["world"])
doc_encoded=pad_sequences(doc_encoded, maxlen=400)
print(doc_encoded[0])
vocab_size=len(t.word_index)+1
EMB_OUTPUT_DIMS = 100
e = Embedding(input_dim=vocab_size, output_dim=EMB_OUTPUT_DIMS, input_length=400)
model=Sequential()
model.add(e)
model.add(LSTM(32))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
train_len = int(len(doc_encoded)*0.7)
X_train, y_train, X_test, y_test = doc_encoded[0:train_len], df_combined[0:train_len]["Label"], doc_encoded[train_len:], df_combined[train_len:]["Label"]
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
model.summary()
model.fit(X_train, y_train, batch_size=16, epochs=3)
results = model.evaluate(X_test, y_test)
print('test loss, test acc:', results)