import pandas as pd
import numpy as np
import nltk
import re
train = pd.read_csv('../input/nlp-getting-started/train.csv')
test = pd.read_csv('../input/nlp-getting-started/test.csv')
train.shape, test.shape
train.head()
test.head()
train.isna().sum()
merged = pd.concat([train,test])
merged.shape
merged.head()
sentences = merged['text']
sentences
for i in range(len(sentences)):
    sentences[i] = sentences[i].str.lower()
    sentences[i] = re.sub(r'[^a-zA-Z]',' ',str(sentences[i]))
def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)
merged['text']=merged['text'].apply(lambda x : remove_url(x))
merged
def remove_html(text):    
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

merged['text']= merged['text'].apply(lambda x : remove_html(x))
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
merged['text']= merged['text'].apply(lambda x: remove_emoji(x))
delete = pd.Series(' '.join(merged['text']).split()).value_counts()[-3000:]
delete
merged['text'] = merged['text'].apply(lambda x: " ".join(x for x in x.split() if x not in delete))
merged.head()
nltk.download("stopwords")
from nltk.corpus import stopwords
sw = stopwords.words("english")
merged["text"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
train.shape
train_df = merged[:7613]
train_df.shape
test_df = merged[7613:]
test_df.head()
test_df.drop('target',axis=1,inplace=True)
test_df
train_df['target'] =train_df['target'].astype(int)
train_df
X = train_df['text']
y = train_df['target']
X
y
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)
from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)
y_test
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()
train_vectors = count_vectorizer.fit_transform(X_train)
test_vectors = count_vectorizer.transform(X_test)
# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf_model = rf.fit(train_vectors,y_train)
from sklearn import model_selection
accuracy = model_selection.cross_val_score(rf_model, 
                                           train_vectors, 
                                           y_train, 
                                           cv = 10).mean()

print("Accuracy of the model: ", accuracy)
# Naive Bayes
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb_model = nb.fit(train_vectors, y_train)
accuracy = model_selection.cross_val_score(nb_model, 
                                           train_vectors, 
                                           y_train, 
                                           cv = 10).mean()

print("Accuracy of naive bayes model: ", accuracy)
test_df = count_vectorizer.transform(test_df["text"])
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")
sample_submission["target"] = nb_model.predict(test_df)
sample_submission.to_csv("submission.csv", index=False)
