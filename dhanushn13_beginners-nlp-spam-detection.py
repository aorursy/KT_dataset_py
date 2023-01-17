import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
df = pd.read_csv(r'/kaggle/input/sms-spam-collection-dataset/spam.csv', encoding='latin-1')
df.head()
df.info()
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1) 
df.head()
df['v2'].isnull().sum()
df = df.rename(columns={"v1": "status", "v2": "content"})
df.shape
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
def stop_words(text):
    words = text.split(" ")
    # Lemmatization 
    lemmatizer = WordNetLemmatizer()
    words_lemmatizer = [lemmatizer.lemmatize(ele) for ele in words]
    nltk_remove_sw = [ele for ele in words_lemmatizer if ele not in stopWords]

    sentence_lemmatized = " ".join(nltk_remove_sw)
    
    return sentence_lemmatized
def pre_process(text):
    tx1 = re.sub('(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})', '', text) # URL's without http
    tx2 = re.sub('[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', '', tx1)# URL's with HTTP
    tx3 = re.sub('[\w\.-]+@[\w\.-]+', '', tx2)  # Emails
    tx4 = re.sub(r'[0-9]+', '', tx3) # Numbers
    tx5 = re.sub(r"[^A-Za-z0-9 ]", ' ', tx4) # Non aplha-numerics
    tx6 = re.sub("\s\s+", " ", tx5) 
    tx7 = tx6.lower() 
    sentence_lemmatized  = stop_words(tx7)
    return sentence_lemmatized
pre_processed_text = []
for ele in df['content']:
    pre_processed_text.append(pre_process(ele))
    
df.insert(loc=2, column='Pre_processed', value=pre_processed_text)
df.head()
df.info()
X_train, X_test, y_train, y_test = train_test_split(df['Pre_processed'], df['status'], test_size=0.35, random_state=42)
vector = CountVectorizer(max_df=0.7)
X_train_vector = vector.fit_transform(X_train)
X_test_vector = vector.transform(X_test)
pd.DataFrame(X_train_vector.toarray(), columns=vector.get_feature_names())
from sklearn.naive_bayes import GaussianNB
clf_GNB = GaussianNB()
clf_GNB.fit(X_train_vector.toarray(), y_train)
print(f'accuracy: {clf_GNB.score(X_test_vector.toarray(), y_test )}')
from sklearn.tree import DecisionTreeClassifier
clf_DTC = DecisionTreeClassifier()
clf_DTC.fit(X_train_vector, y_train)
print(f'accuracy: {clf_DTC.score(X_test_vector, y_test)}')
from sklearn.svm import SVC
clf_SVC = SVC()
clf_SVC.fit(X_train_vector, y_train)
print(f'accuracy: {clf_SVC.score(X_test_vector, y_test)}')
from sklearn.linear_model import LogisticRegression
clf_LR = LogisticRegression()
clf_LR.fit(X_train_vector, y_train)
print(f'accuracy: {clf_LR.score(X_test_vector, y_test)}')