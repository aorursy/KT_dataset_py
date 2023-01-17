import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import numpy as np 

import pandas as pd

import spacy

import seaborn as sns

import string
def multiclass_logloss(actual, predicted, eps=1e-15):

    """Multi class version of Logarithmic Loss metric.

    :param actual: Array containing the actual target classes

    :param predicted: Matrix with class predictions, one probability per class

    """

    # Convert 'actual' to a binary array if it's not already:

    if len(actual.shape) == 1:

        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))

        for i, val in enumerate(actual):

            actual2[i, val] = 1

        actual = actual2



    clip = np.clip(predicted, eps, 1 - eps)

    rows = actual.shape[0]

    vsota = np.sum(actual * np.log(clip))

    return -1.0 / rows * vsota
nlp = spacy.load("en_core_web_sm")

#READING INPUT

data = pd.read_csv("/kaggle/input/spooky-author-identification/train.csv")

data.head()
test = pd.read_csv("/kaggle/input/spooky-author-identification/test.csv")
doc = nlp(data["text"][0])

for token in doc[0:5]:

    print(token.text, token.pos , token.pos_, token.dep_) # part of speach and syntax dependency
for token in doc[0:5]:

    print(token.text,  token.pos_, token.lemma_) # part of speach and syntax dependency
sns.barplot(x=['Edgar Allen Poe', 'Mary Wollstonecraft Shelley', 'H.P. Lovecraft'], y=data['author'].value_counts())
data['author_num'] = data["author"].map({'EAP':0, 'HPL':1, 'MWS':2})

data.head()
from nltk.corpus import stopwords

stopwords = stopwords.words('english')

print (stopwords)
## Number of words in the text ##

data["num_words"] = data["text"].apply(lambda x: len(str(x).split()))

test["num_words"] = test["text"].apply(lambda x: len(str(x).split()))

## Number of unique words in the text ##

data["num_unique_words"] = data["text"].apply(lambda x: len(set(str(x).split())))

test["num_unique_words"] = test["text"].apply(lambda x: len(set(str(x).split())))

## Number of characters in the text ##

data["num_chars"] = data["text"].apply(lambda x: len(str(x)))

test["num_chars"] = test["text"].apply(lambda x: len(str(x)))

## Number of stopwords in the text ##

data["num_stopwords"] = data["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in stopwords]))

test["num_stopwords"] = test["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in stopwords]))

## Number of punctuations in the text ##

data["num_punctuations"] =data['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

test["num_punctuations"] =test['text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

## Number of title case words in the text ##

data["num_words_upper"] = data["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

test["num_words_upper"] = test["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

## Number of title case words in the text ##

data["num_words_title"] = data["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

test["num_words_title"] = test["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

## Max length of the words in the text ##

data["max_word_len"] = data["text"].apply(lambda x: np.max([len(w) for w in str(x).split()]))

test["max_word_len"] = test["text"].apply(lambda x: np.max([len(w) for w in str(x).split()]))




# Define function to cleanup text by removing personal pronouns, stopwords, and puncuation

def cleanup_text(docs, logging=False):

    texts = []

    counter = 1

    for doc in docs:

        if counter % 1000 == 0 and logging:

            print("Processed %d out of %d documents." % (counter, len(docs)))

        counter += 1

        doc = nlp(doc, disable=['parser', 'ner'])

        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']

        tokens = [tok for tok in tokens if tok not in stopwords and tok not in string.punctuation]

        #tokens = [tok for tok in tokens if tok not in punctuations]

        tokens = ' '.join(tokens)

        texts.append(tokens)

    return pd.Series(texts)
print('Original training data shape: ', data['text'].shape)

data["text_cleaned"]= cleanup_text(data['text'], logging=True)

print('Cleaned up training data shape: ', data["text_cleaned"].shape)
print('Original training data shape: ', test['text'].shape)

test["text_cleaned"] = cleanup_text(test['text'], logging=True)

print('Cleaned up training data shape: ', test["text_cleaned"].shape)


data["num_unique_words_clenaed"] = data["text_cleaned"].apply(lambda x: len(set(str(x).split())))

test["num_unique_words_cleaned"] = test["text_cleaned"].apply(lambda x: len(set(str(x).split())))
def numberOfADV(docs, logging=False):

    numberOfADV = []

    counter = 1

    for doc in docs:

        if counter % 1000 == 0 and logging:

            print("Processed %d out of %d documents." % (counter, len(docs)))

        counter += 1

        doc = nlp(doc, disable=['parser', 'ner'])

        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.pos_ == 'ADP']

        #tokens = [tok for tok in tokens if tok not in stopwords and tok not in string.punctuation]

        #tokens = [tok for tok in tokens if tok not in punctuations]

        #tokens = ' '.join(tokens)

        

        numberOfADV.append(len(tokens))

    return pd.Series(numberOfADV)
data["num_of_ADV"] = numberOfADV(data['text_cleaned'], logging=True)
test["num_of_ADV"] = numberOfADV(test['text_cleaned'], logging=True)
def numberOfADJ(docs, logging=False):

    numberOfADV = []

    counter = 1

    for doc in docs:

        if counter % 1000 == 0 and logging:

            print("Processed %d out of %d documents." % (counter, len(docs)))

        counter += 1

        doc = nlp(doc, disable=['parser', 'ner'])

        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.pos_ == 'ADJ']

        

        numberOfADV.append(len(tokens))

    return pd.Series(numberOfADV)
data["num_of_ADJ"] = numberOfADJ(data['text_cleaned'], logging=True)
test["num_of_ADJ"] = numberOfADJ(test['text_cleaned'], logging=True)
data.head()
test.head()
sns.barplot(x= data["author"], y = data["num_of_ADJ"])
sns.barplot(x= data["author"], y = data["num_of_ADV"])
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
X_train_matrix = vect.fit_transform(data["text_cleaned"]) 

X_test_matrix = vect.transform(test["text_cleaned"]) 
features = vect.get_feature_names()

df_X_train_matrix = pd.DataFrame(X_train_matrix.toarray(), columns=features)

df_X_train_matrix.head()

df_X_test_matrix = pd.DataFrame(X_test_matrix.toarray(), columns=features)

df_X_test_matrix.head()
data_df = data.drop(["id","text", "text_cleaned", "author"], axis = 1)



df_train = pd.concat([data_df, df_X_train_matrix], axis=1)



test_df = test.drop(["id","text", "text_cleaned"], axis = 1)



df_test = pd.concat([test_df, df_X_test_matrix], axis=1)
df_train.head()
X = df_train.drop("author_num", axis = 1)

y = data['author_num']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify = y)
from sklearn.naive_bayes import MultinomialNB

clf=MultinomialNB()

clf.fit(X_train, y_train)

print(clf.score(X_train, y_train))



print (clf.score(X_test, y_test))
predicted_result=clf.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test,predicted_result))
predictions = clf.predict_proba(X_test)



print ("logloss: %0.3f " % multiclass_logloss(y_test, predictions))
sample = pd.read_csv("/kaggle/input/spooky-author-identification/sample_submission.csv")

sample.head()


predicted_result = clf.predict_proba(df_test)
result=pd.DataFrame()

result["id"]=test["id"]

result["EAP"]=predicted_result[:,0]

result["HPL"]=predicted_result[:,1]

result["MWS"]=predicted_result[:,2]

result.head()
result.to_csv("submission_v3.csv", index=False)