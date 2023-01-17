import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Loading data
train_data = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv", na_values = 'nan')
test_data = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv", na_values = 'nan')
submission_data = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
#Replacing NaN with Missing
train_data = train_data.replace(np.nan,'missing',regex=True)
test_data = test_data.replace(np.nan,'missing',regex=True)
import re
def substituteUrls(inputString):
    inputString = str(inputString)
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,inputString)
    if(len(url)>0):
        return ""
    else:
        return inputString
#Expanding Custom Words
def customWordsExpaned(inputString):
    inputString = str(inputString).lower()
    if(inputString == "i'll"):
        return "i will"
    elif(inputString == "you'll"):
        return "you will"
    elif(inputString == "he'll"):
        return "he will"
    elif(inputString == "she'll"):
        return "she will"
    elif(inputString == "it'll"):
        return "it will"
    elif(inputString == "we'll"):
        return "we will"
    elif(inputString == "they'll"):
        return "they will"
    elif(inputString == "i'd"):
        return "i would"
    elif(inputString == "you'd"):
        return "you would"
    elif(inputString == "he'd"):
        return "he would"
    elif(inputString == "she'd"):
        return "she would"
    elif(inputString == "we'd"):
        return "we would"
    elif(inputString == "they'd"):
        return "they would"
    elif(inputString == "i've"):
        return "i have"
    elif(inputString == "you've"):
        return "you have"
    elif(inputString == "we've"):
        return "we have"
    elif(inputString == "they've"):
        return "they have"
    elif(inputString == "could've"):
        return "could have"
    elif(inputString == "should've"):
        return "should have"
    elif(inputString == "would've"):
        return "would have"
    elif(inputString == "i'm"):
        return "i am"
    elif(inputString == "you're"):
        return "you are"
    elif(inputString == "he's"):
        return "he is"
    elif(inputString == "she's"):
        return "she is"
    elif(inputString == "it's"):
        return "it is"
    elif(inputString == "there's"):
        return "there is"
    elif(inputString == "that's"):
        return "that is"
    elif(inputString == "who's"):
        return "who is"
    elif(inputString == "what's"):
        return "what is"
    elif(inputString == "where's"):
        return "where is"
    elif(inputString == "how's"):
        return "how is"
    elif(inputString == "we're"):
        return "we are"
    elif(inputString == "they're"):
        return "they are"
    elif(inputString == "let's"):
        return "let us"
    elif(inputString == "y'all"):
        return "you all"
    elif(inputString == "didn't"):
        return "did not"
    elif(inputString == "ain't"):
        return "am not"
    elif(inputString == "don't"):
        return "do not"
    elif(inputString == "haven't"):
        return "have not"
    elif(inputString == "weren't"):
        return "were not"
    elif(inputString == "can't"):
        return "cannot"
    elif(inputString == "doesn't"):
        return "does not"
    elif(inputString == "won't"):
        return "will not"
    elif(inputString == "wasn't"):
        return "was not"
    elif(inputString == "aren't"):
        return "are not"
    elif(inputString == "isn't"):
        return "is not"
    else:
        return inputString
#Removing Stopwords from Text column
from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from wordcloud import STOPWORDS
def removeStopWords(text):
    text = str(text).lower()
    text_tokens = word_tokenize(text)
    tokens_without_sw1 = [word for word in text_tokens if not word in stopwords.words()]
    tokens_without_sw2 = [word for word in tokens_without_sw1 if not word in STOPWORDS]
    return " ".join(tokens_without_sw2)
def tweetCleaner(tweet):
    #tweet = "Severe Thunderstorm 7 Miles North of Arcadia Moving SE At 20 MPH. 60 MPH Wind Gusts and Large Hail. Hail... #okwx http://t.co/WEd69e4KRg"
    tweet = str(tweet)
    tweet_list = re.split(r'[;,?.\s]', tweet)
    updated_list = []
    for item in tweet_list:
        if(len(item)>0):
            #item = substituteUrls(item)
            item = item.replace("#","")
            if(re.search("^[A-Za-z]+$", item)):
                item = customWordsExpaned(item)
                updated_list.append(item)
    updated_tweet = removeStopWords(" ".join(updated_list))
    return updated_tweet
#Creating a second vector for classification model later.
disasterVec2 = train_data["keyword"].to_numpy()

#dropping keyword and location
train_data.drop(columns =['keyword','location'], axis=1)
test_data.drop(columns =['keyword','location'], axis=1)
#Feature Engineering - Adding column "clean_text"
train_data["clean_text"] = train_data["text"].apply(tweetCleaner)
test_data["clean_text"] = test_data["text"].apply(tweetCleaner)
#now that we have clean text, we no longer need the text column
train_data.drop(columns =['text'], axis=1)
test_data.drop(columns =['text'], axis=1)

count_vectorizer1 = feature_extraction.text.CountVectorizer() # will use unique words in Clean text column
count_vectorizer2 = feature_extraction.text.CountVectorizer() # will use unique words in keyword column (disasterVec2)
count_vectorizer3 = feature_extraction.text.CountVectorizer() # will use a custom list (disasterVec1)

tfid_vectorizer1 = feature_extraction.text.TfidfVectorizer() # will use unique words in Clean text column
tfid_vectorizer2 = feature_extraction.text.TfidfVectorizer() # will use unique words in keyword column (disasterVec2)
tfid_vectorizer3 = feature_extraction.text.TfidfVectorizer() # will use a custom list (disasterVec1)

#Using Count Vectorizer 1
train_vectors_cv1 = count_vectorizer1.fit_transform(train_data["clean_text"])
test_vectors_cv1 = count_vectorizer1.transform(test_data["clean_text"])

#Using Tfid Vectorizer 1
train_vectors_tfid1 = tfid_vectorizer1.fit_transform(train_data["clean_text"])
test_vectors_tfid1 = tfid_vectorizer1.transform(test_data["clean_text"])


# Our vectors are really big, so we want to push our model's weights toward 0 without completely discounting different words
#Let's use Ridge regression here 
clf_cv1 = linear_model.RidgeClassifier(random_state=0, solver='sparse_cg') # for Count Vectorizer
clf_tfid1 = linear_model.RidgeClassifier(random_state=0, solver='sparse_cg') # for Tfid Vectorizer

scores_cv1 = model_selection.cross_val_score(clf_cv1, train_vectors_cv1, train_data["target"], cv=3, scoring="f1")
print("F1 Scores for Count Vectorizer: ", scores_cv1)

clf_cv1.fit(train_vectors_cv1, train_data["target"])
submission_cv1 = pd.DataFrame(columns =['id','target'])
submission_cv1["id"] = submission_data["id"]
submission_cv1["target"] = clf_cv1.predict(test_vectors_cv1)


scores_tfid1 = model_selection.cross_val_score(clf_tfid1, train_vectors_tfid1, train_data["target"], cv=3, scoring="f1")
print("F1 Scores for Tfid Vectorizer: ", scores_tfid1)

clf_tfid1.fit(train_vectors_tfid1, train_data["target"])
submission_tfid1 = pd.DataFrame(columns =['id','target'])
submission_tfid1["id"] = submission_data["id"]
submission_tfid1["target"] = clf_tfid1.predict(test_vectors_tfid1)

print("Classification Report - CV1")
print(classification_report(submission_data["target"],submission_cv1["target"]))
print("Classification Report - TFID1")
print(classification_report(submission_data["target"],submission_tfid1["target"]))

print("Confusion Matrix - CV1")
print(confusion_matrix(submission_data["target"],submission_cv1["target"]))
print("Confusion Matrix - TFID1")
print(confusion_matrix(submission_data["target"],submission_tfid1["target"]))
#Using Count Vectorizer 2
count_vectorizer2.fit(disasterVec2)
train_vectors_cv2 = count_vectorizer2.transform(train_data["clean_text"])
test_vectors_cv2 = count_vectorizer2.transform(test_data["clean_text"])

#Using Tfid Vectorizer 2
tfid_vectorizer2.fit(disasterVec2)
train_vectors_tfid2 = tfid_vectorizer2.transform(train_data["clean_text"])
test_vectors_tfid2 = tfid_vectorizer2.transform(test_data["clean_text"])

# Our vectors are really big, so we want to push our model's weights toward 0 without completely discounting different words
#Let's use Ridge regression here 
clf_cv2 = linear_model.RidgeClassifier(random_state=0, solver='sparse_cg') # for Count Vectorizer
clf_tfid2 = linear_model.RidgeClassifier(random_state=0, solver='sparse_cg') # for Tfid Vectorizer

scores_cv2 = model_selection.cross_val_score(clf_cv2, train_vectors_cv2, train_data["target"], cv=3, scoring="f1")
print("F1 Scores for Count Vectorizer: ", scores_cv2)

clf_cv2.fit(train_vectors_cv2, train_data["target"])
submission_cv2 = pd.DataFrame(columns =['id','target'])
submission_cv2["id"] = submission_data["id"]
submission_cv2["target"] = clf_cv2.predict(test_vectors_cv2)


scores_tfid2 = model_selection.cross_val_score(clf_tfid2, train_vectors_tfid2, train_data["target"], cv=3, scoring="f1")
print("F1 Scores for Tfid Vectorizer: ", scores_tfid2)

clf_tfid2.fit(train_vectors_tfid2, train_data["target"])
submission_tfid2 = pd.DataFrame(columns =['id','target'])
submission_tfid2["id"] = submission_data["id"]
submission_tfid2["target"] = clf_tfid2.predict(test_vectors_tfid2)

print("Classification Report - CV2")
print(classification_report(submission_data["target"],submission_cv2["target"]))
print("Classification Report - TFID2")
print(classification_report(submission_data["target"],submission_tfid2["target"]))

print("Confusion Matrix - CV2")
print(confusion_matrix(submission_data["target"],submission_cv2["target"]))
print("Confusion Matrix - TFID2")
print(confusion_matrix(submission_data["target"],submission_tfid2["target"]))
#Custom Vector compiled
disasterDF = pd.read_csv("/kaggle/input/disastervec1/disasterVec1.csv", names = ['keyword'])
disasterVec1 = disasterDF["keyword"].to_numpy()
#Using Count Vectorizer 3
count_vectorizer3.fit(disasterVec1)
train_vectors_cv3 = count_vectorizer3.transform(train_data["clean_text"])
test_vectors_cv3 = count_vectorizer3.transform(test_data["clean_text"])

#Using Tfid Vectorizer 3
tfid_vectorizer3.fit(disasterVec1)
train_vectors_tfid3 = tfid_vectorizer3.transform(train_data["clean_text"])
test_vectors_tfid3 = tfid_vectorizer3.transform(test_data["clean_text"])

# Our vectors are really big, so we want to push our model's weights toward 0 without completely discounting different words
#Let's use Ridge regression here 
clf_cv3 = linear_model.RidgeClassifier(random_state=0, solver='sparse_cg') # for Count Vectorizer
clf_tfid3 = linear_model.RidgeClassifier(random_state=0, solver='sparse_cg') # for Tfid Vectorizer

scores_cv3 = model_selection.cross_val_score(clf_cv3, train_vectors_cv3, train_data["target"], cv=3, scoring="f1")
print("F1 Scores for Count Vectorizer: ", scores_cv3)

clf_cv3.fit(train_vectors_cv3, train_data["target"])
submission_cv3 = pd.DataFrame(columns =['id','target'])
submission_cv3["id"] = submission_data["id"]
submission_cv3["target"] = clf_cv3.predict(test_vectors_cv3)

scores_tfid3 = model_selection.cross_val_score(clf_tfid3, train_vectors_tfid3, train_data["target"], cv=3, scoring="f1")
print("F1 Scores for Tfid Vectorizer: ", scores_tfid3)

clf_tfid3.fit(train_vectors_tfid3, train_data["target"])
submission_tfid3 = pd.DataFrame(columns =['id','target'])
submission_tfid3["id"] = submission_data["id"]
submission_tfid3["target"] = clf_tfid3.predict(test_vectors_tfid3)

print("Classification Report - CV3")
print(classification_report(submission_data["target"],submission_cv3["target"]))
print("Classification Report - TFID3")
print(classification_report(submission_data["target"],submission_tfid3["target"]))

print("Confusion Matrix - CV3")
print(confusion_matrix(submission_data["target"],submission_cv3["target"]))
print("Confusion Matrix - TFID3")
print(confusion_matrix(submission_data["target"],submission_tfid1["target"]))