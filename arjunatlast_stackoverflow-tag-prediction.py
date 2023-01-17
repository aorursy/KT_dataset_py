import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
dirname = "/kaggle/input/stacksample"



questions_csv = os.path.join(dirname, "Questions.csv")

answers_csv = os.path.join(dirname, "Answers.csv")

tags_csv = os.path.join(dirname, "Tags.csv")



ENCODING = 'ISO-8859-1'
# The questions



df_q = pd.read_csv(questions_csv, encoding=ENCODING)



df_q.head()
# the tags



df_t = pd.read_csv(tags_csv, encoding=ENCODING)



df_t.head()
df_t['Tag'] = df_t['Tag'].astype(str)



# group all tags given to same question into a single string

grouped_tags = df_t.groupby('Id')['Tag'].apply(lambda tags: ' '.join(tags))



grouped_tags.head()
# reset index for simplicity

grouped_tags.reset_index()



df_tags_final = pd.DataFrame({'Id': grouped_tags.index, 'Tags': grouped_tags.values})



df_tags_final.head()
# Drop unnecessary columns from questions

df_q.drop(columns=['OwnerUserId', 'CreationDate', 'ClosedDate'], inplace=True)
# Merge questions and tags into a single dataframe

df = df_q.merge(df_tags_final, on='Id')



df.head()
# remove questions with score lower than 5

df = df[df['Score'] > 5]



print(df.shape)
# split tags into list

df['Tags'] = df['Tags'].apply(lambda tags: tags.lower().split())



df.head()
# get all tags in the dataset

all_tags = []



for tags in df['Tags'].values:

    for tag in tags:

        all_tags.append(tag)

        

print(all_tags)
import nltk



# create a frequency list of the tags

tag_freq = nltk.FreqDist(list(all_tags))



# get most common tags

tag_freq.most_common(25)
# get the most common 50 tags without the count

tag_features = list(map(lambda x: x[0], tag_freq.most_common(50)))



print(tag_features)
# Filter the tags from the dataset and remove all tags that does not belong to the tag_features

def keep_common(tags):

    

    filtered_tags = []

    

    # filter tags

    for tag in tags:

        if tag in tag_features:

            filtered_tags.append(tag)

    

    # return the filtered tag list

    return filtered_tags



# apply the function to filter in dataset

df['Tags'] = df['Tags'].apply(lambda tags: keep_common(tags))



# set the Tags column as None for those that do not have a most common tag

df['Tags'] = df['Tags'].apply(lambda tags: tags if len(tags) > 0 else None)



# Now we will drop all the columns that contain None in Tags column

df.dropna(subset=['Tags'], inplace=True)



df.shape
import re

from nltk.corpus import stopwords

from nltk.tokenize import ToktokTokenizer

from nltk.stem.snowball import SnowballStemmer



tokenizer = ToktokTokenizer()

stemmer = SnowballStemmer('english')

stop_words = set(stopwords.words('english'))



# Preprocess the text for vectorization

# - Remove HTML

# - Remove stopwords

# - Remove special characters

# - Convert to lowercase

# - Stemming



def remove_html(text):

    # Remove html and convert to lowercase

    return re.sub(r"\<[^\>]\>", "", text).lower()



def remove_stopwords(text):    

    # tokenize the text

    words = tokenizer.tokenize(text)

    

    filtered = [w for w in words if not w in stop_words]

    return ' '.join(map(str, filtered))



def remove_punc(text):

    #tokenize

    tokens = tokenizer.tokenize(text)

    

    # remove punctuations from each token

    tokens = list(map(lambda token: re.sub(r"[^A-Za-z0-9]+", " ", token).strip(), tokens))

    

    # remove empty strings from tokens

    tokens = list(filter(lambda token: token, tokens))

    

    return ' '.join(map(str, tokens))



def stem_text(text):

    #tokenize

    tokens = tokenizer.tokenize(text)

    

    # stem each token

    tokens = list(map(lambda token: stemmer.stem(token), tokens))

    

    return ' '.join(map(str, tokens))
# drop Id and Score columns since we don't need them

df.drop(columns=['Id', 'Score'], inplace=True)



df.head()
# apply preprocessing to title and body

df['Title'] = df['Title'].apply(lambda x: remove_html(x))

df['Title'] = df['Title'].apply(lambda x: remove_stopwords(x))

df['Title'] = df['Title'].apply(lambda x: remove_punc(x))

df['Title'] = df['Title'].apply(lambda x: stem_text(x))



df.head()
# apply preprocessing to title and body

df['Body'] = df['Body'].apply(lambda x: remove_html(x))

df['Body'] = df['Body'].apply(lambda x: remove_stopwords(x))

df['Body'] = df['Body'].apply(lambda x: remove_punc(x))

df['Body'] = df['Body'].apply(lambda x: stem_text(x))



df.head()
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from scipy.sparse import hstack
X_title = df['Title']

X_body = df['Body']

y = df['Tags']



# binarize our tags 

binarizer = MultiLabelBinarizer()

y_bin = binarizer.fit_transform(y)
# vectorize

vectorizer_title = TfidfVectorizer(

    analyzer = 'word', 

    strip_accents = None, 

    encoding = 'utf-8', 

    preprocessor=None, 

    max_features=10000)



vectorizer_body = TfidfVectorizer(

    analyzer = 'word', 

    strip_accents = None, 

    encoding = 'utf-8', 

    preprocessor=None, 

    max_features=10000)



X_title_vect = vectorizer_title.fit_transform(X_title)

X_body_vect = vectorizer_body.fit_transform(X_body)



X = hstack([X_title_vect, X_body_vect])
X.shape
# train test split

X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size = 0.2, random_state = 0)
# Develop the model

from sklearn.svm import LinearSVC

from skmultilearn.problem_transform import BinaryRelevance # gives better precision



svc = LinearSVC()

clf = BinaryRelevance(svc)



# fit training data

clf.fit(X_train, y_train)

# Metrics

from sklearn.metrics import accuracy_score, precision_score, recall_score, hamming_loss, f1_score



# make prediction

y_pred = clf.predict(X_test)
# calculate Accuracy

print("Accuracy:", accuracy_score(y_test, y_pred))



# calculate recall

print("Recall:", recall_score(y_true=y_test, y_pred=y_pred, average='weighted'))



# calculate precision

print("Precision: ", precision_score(y_true=y_test, y_pred=y_pred, average='weighted'))



# calculate hamming loss

print("Hamming Loss (%): ", hamming_loss(y_pred, y_test)*100)



# calculate F1 score

print("F1 Score: ", f1_score(y_pred, y_test, average='weighted'))

# Actual Application

q_title = "How to handle or avoid a stack overflow in C++"

q_body = "In C++ a stack overflow usually leads to an unrecoverable crash of the program. For programs that need to be really robust, this is an unacceptable behaviour, particularly because stack size is limited. A few questions about how to handle the problem. Is there a way to prevent stack overflow by a general technique. (A scalable, robust solution, that includes dealing with external libraries eating a lot of stack, etc.) Is there a way to handle stack overflows in case they occur? Preferably, the stack gets unwound until there's a handler to deal with that kinda issue. There are languages out there, that have threads with expandable stacks. Is something like that possible in C++? Any other helpful comments on the solution of the C++ behaviour would be appreciated."



# preprocessing title and body

def preprocess_text(text):

    text = remove_html(text)

    text = remove_stopwords(text)

    text = remove_punc(text)

    text = stem_text(text)

    

    return text



q_title = preprocess_text(q_title)

q_body = preprocess_text(q_body)



print("Title:", q_title)

print("Body:", q_body)
# X_app_title = vectorizer_title.fit_transform([q_title])

# X_app_body = vectorizer_body.fit_transform([q_body])



# X_app = hstack([X_app_title, X_app_body])



# y_app = clf.predict(X_app)



# print(y_app)