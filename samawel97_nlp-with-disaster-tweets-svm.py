import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import feature_extraction, linear_model, model_selection, preprocessing
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
test_set= test_df.copy(deep=True)

test_set
test_df.head(10)
train_df.head()
train_df['text'].values[0]
train_df[train_df["target"] == 0]["text"].values[1]
train_df[train_df["target"] == 1]["text"].values[1]
#Text preproceesin 

import re

def  clean_text(df, text_field, new_text_field_name):

    df[new_text_field_name] = df[text_field].str.lower()

    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))  

    # remove numbers

    df[new_text_field_name] = df[new_text_field_name].apply(lambda elem: re.sub(r"\d+", "", elem))

    

    return df



train_df = clean_text(train_df, 'text', 'text_clean')

test_df =clean_text(test_df, 'text', 'text_clean')

train_df

#cleaning text

import nltk.corpus

nltk.download('stopwords')

from nltk.corpus import stopwords

stop = stopwords.words('english')

train_df['text_clean'] = train_df['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

test_df['text_clean'] = test_df['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))



train_df.head()
# text tokenization

import nltk 

nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize

train_df['text_tokens'] = train_df['text_clean'].apply(lambda x: word_tokenize(x))

train_df.head()
#Stemming words with NLTK

from nltk.stem import PorterStemmer 

from nltk.tokenize import word_tokenize

def word_stemmer(text):

    stem_text = [PorterStemmer().stem(i) for i in text]

    return stem_text

train_df['text_clean_tokens'] = train_df['text_tokens'].apply(lambda x: word_stemmer(x))

train_df.head()
# text lemmatisation 

nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

def word_lemmatizer(text):

    lem_text = [WordNetLemmatizer().lemmatize(i) for i in text]

    return lem_text

train_df['text_clean_tokens'] = train_df['text_tokens'].apply(lambda x: word_lemmatizer(x))

train_df['text_clean_tokens'] = train_df['text_clean_tokens'].astype(str)

train_df.head()
test_df
count_vectorizer = feature_extraction.text.CountVectorizer()



## let's get counts for the first 5 tweets in the data

example_train_vectors = count_vectorizer.fit_transform(train_df["text_clean"][0:5])
## we use .todense() here because these vectors are "sparse" (only non-zero elements are kept to save space)

print(example_train_vectors[0].todense().shape)

print(example_train_vectors[0].todense())
train_vectors = count_vectorizer.fit_transform(train_df["text_clean"])



## note that we're NOT using .fit_transform() here. Using just .transform() makes sure

# that the tokens in the train vectors are the only ones mapped to the test vectors - 

# i.e. that the train and test vectors use the same set of tokens.

test_vectors = count_vectorizer.transform(test_df["text_clean"])
## Our vectors are really big, so we want to push our model's weights

## toward 0 without completely discounting different words - ridge regression 

## is a good way to do this.

from sklearn.svm import SVC



classifier = SVC(kernel = 'sigmoid', gamma='scale')
# and now let's test our model and see how well it does on the total training data

scores = model_selection.cross_val_score(classifier, train_vectors, train_df["target"], cv=3, scoring="f1")

scores
classifier.fit(train_vectors, train_df["target"])
sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = classifier.predict(test_vectors)
sample_submission.head()
df_test_sub= pd.merge(sample_submission, test_df , how = 'outer')

df_test_sub
sample_submission.to_csv("submission.csv", index=False)
sample_submission.head(500)