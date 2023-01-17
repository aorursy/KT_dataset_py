import nltk

import pandas as pd

import numpy as np
train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train_df.shape,test_df.shape
all_data = pd.concat((train_df.loc[:,:'text'],test_df.loc[:,:'text']))
all_data.shape
train_df.head()
test_df.head()
train_df['text'].values[1]
from nltk.corpus import stopwords

from nltk.tokenize import sent_tokenize

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
text = "All residents asked to 'shelter in place' are being notified by officers. No other evacuation or shelter in place orders are expected"
import re

def preprocess_doc(docs):

    documents = []

    for text in docs:

        sents = [s for s in sent_tokenize(text)]

        words = [w for s in sents for w in word_tokenize(s)]

        last_words = [l for l in words if l not in stop_words]

        #making evrythin gto lower case

        lw_cs = [lw.lower() for lw in last_words]

        # emoving the numbers

        number_words = [re.sub(r'\d+', '', text) for text in lw_cs]

        #removing the special characters

        sc_words = [re.sub(r'#*', '', text) for text in number_words]

        stem_words = [WordNetLemmatizer().lemmatize(w) for w in sc_words]

        documents.append(" ".join(stem_words))

    return np.array(documents)
docs = train_df['text'].values[1:10]

docs
preprocess_doc(docs)
all_data['text'] = preprocess_doc(all_data['text'].values)
all_data['text'].head(10)
# using the count vectorizer

from sklearn.feature_extraction.text import CountVectorizer
features = CountVectorizer().fit_transform(all_data['text'].values)
# Using the tf - idf vectorizer to create features

from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf = TfidfVectorizer()

feature_set = tf_idf.fit_transform(all_data['text'].values)
feature_set.shape
print(tf_idf.get_feature_names())
X = feature_set[:7613]

y = train_df['target']

Test = feature_set[7613:]
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,y, test_size = 0.33, random_state = 45)
from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score

def get_result(model,x_train,x_test,y_train,y_test):

    model = model.fit(x_train,y_train)

    y_train_predict =  model.predict(x_train)

    y_test_predict =  model.predict(x_test)

    print("-----------------------------------------------------------------")

    print("Train set accuracy:",accuracy_score(y_train,y_train_predict))

    print("Test set accuracy:",accuracy_score(y_test,y_test_predict))

    print("F1 score is:",f1_score(y_test,y_test_predict))

    print("-----------------------------------------------------------------")
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

print("Logistic_regression results")

get_result(lr,x_train,x_test,y_train,y_test)
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

print("Decision Tree results")

get_result(dt,x_train,x_test,y_train,y_test)
Y_preds = dt.predict(Test)
sln = pd.DataFrame({'id':test_df.id,'target':Y_preds})
sln.to_csv("Detect the fake tweet.csv", index = False)