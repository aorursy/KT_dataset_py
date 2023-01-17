import os
os.listdir('../input/')
import pandas as pd
data = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv", encoding = "latin-1")
data = data[['v1', 'v2']]
data = data.rename(columns = {'v1': 'label', 'v2': 'text'})
data.head()
data.label.value_counts()
def review_messages(msg):
    # converting messages to lowercase
    msg = msg.lower()
    return msg
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))
def alternative_review_messages(msg):
    # converting messages to lowercase
    msg = msg.lower()
    # removing stopwords 
    msg = [word for word in msg.split() if word not in stopwords]
    # using a lemmatized 
    msg = " ".join([lemmatizer.lemmatize(word) for word in msg])
    return msg
data['text'] = data['text'].apply(review_messages)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size = 0.1, random_state = 1)
# training the vectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
from sklearn import svm
svm = svm.SVC(C=1000)
svm.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix
X_test = vectorizer.transform(X_test)
y_pred = svm.predict(X_test)
print(confusion_matrix(y_test, y_pred))
def pred(msg):
    msg = vectorizer.transform([msg])
    prediction = svm.predict(msg)
    return prediction[0]
pred("How are You ?")
pred("You HAVE won 20 million CASH PRIZE")