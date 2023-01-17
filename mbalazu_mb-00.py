

import numpy as np

import pandas as pd



from sklearn import svm

from sklearn.metrics import accuracy_score

from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score

from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt 

import re



def preprocess_reviews(data):



    # remove punctuation 

    REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")

    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    data['text'] = data['text'].apply(lambda x: re.sub(REPLACE_NO_SPACE, "", x))

    data['text'] = data['text'].apply(lambda x: re.sub(REPLACE_WITH_SPACE, " ", x))



    # replace urls

    URL = re.compile("https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)")

    data['text'] = data['text'].apply(lambda x: re.sub(URL, "URL", x))

    

    # replace numbers with [NUMBER]

    data['text'] = data['text'].apply(lambda x: re.sub("(\d+)", "NUMBER", x))

    

    

    return data





"""

def preprocess_reviews(data):



    # remove punctuation 

    REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")

    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

    data['text'] = data['text'].apply(lambda x: re.sub(REPLACE_NO_SPACE, "", x))

    data['text'] = data['text'].apply(lambda x: re.sub(REPLACE_WITH_SPACE, " ", x))



    # replace urls

    URL = re.compile("https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)")

    data['text'] = data['text'].apply(lambda x: re.sub(URL, "URL", x))

    

    # replace hashtags with [HASHTAG]

    data['text'] = data['text'].apply(lambda x: re.sub("\#[\w]*", "HASHTAG", x))

    

    # replace numbers with [NUMBER]

    data['text'] = data['text'].apply(lambda x: re.sub("(\d+)", "NUMBER", x))

    

    # replace mentions/responses with [MENTION]

    data['text'] = data['text'].apply(lambda x: re.sub("\@[\w]*", "MENTION", x))

    

    return data



"""



def get_stemmed_text(corpus):

    from nltk.stem.porter import PorterStemmer

    stemmer = PorterStemmer()

    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

# Datasets

pd_train_set = pd.read_csv("../input/nlp-getting-started/train.csv")[['text', 'target']]

pd_test_set = pd.read_csv("../input/nlp-getting-started/test.csv")[['text']]

pd_train_set.shape
pd_train_set = pd_train_set.drop_duplicates()

pd_train_set = preprocess_reviews(pd_train_set)

pd_train_set.head
cv = CountVectorizer(

    binary=True,

    strip_accents='ascii',

    stop_words='english',

    min_df=2)

"""

cv = CountVectorizer(

    binary=True,

    strip_accents='ascii',

    stop_words='english',

    min_df=2)

"""
X = cv.fit_transform(get_stemmed_text(pd_train_set.text))

y = pd_train_set.target



model = svm.SVC(probability=True,random_state=42)

model.fit(X, y)
y_prima = model.predict(X)

accuracy_score = accuracy_score(y, y_prima)

cv_score = cross_val_score(model, X, y, cv=2, n_jobs=-1).mean()

print(f"accuracy_score={accuracy_score}, CV-Score={cv_score}")

#46,406

labels = ['fake','real']

plot_confusion_matrix(model, X, y, display_labels=labels, normalize=None)

plt.show()





#67,492 todo

#70,458 todo menos hastag

#71,487
print(classification_report(y, y_prima, target_names=labels))
# Test Set

pd_test_set = preprocess_reviews(pd_test_set)

X = cv.transform(get_stemmed_text(pd_test_set['text']))

y_prima = model.predict(X)
sample = pd.read_csv("../input/nlp-getting-started/sample_submission.csv") 

sample["target"] = y_prima

sample.to_csv(r"sample_submission.csv", index=False)