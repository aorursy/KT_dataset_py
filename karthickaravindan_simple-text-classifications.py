import pandas as pd
import re
df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
df_test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
df.head()
# select particular columns
df=df[['text','target']]
df.head()
test=df['text'].sample(5).to_list()
test
import nltk
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words[:10]
"""
->  Text cleaning 
        * Remove urls,hastags,usernames etc
        * Remove stopwords like ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves'] etc .,
"""


def remove_stopwords(text):
    text = ' '.join([i for i in text.split(' ') if i not in stop_words ])
    return text


def clean_text(text):
    text = re.sub('https?://[A-Za-z0-9./]*','', text) # Remove https..(URL)
    text = re.sub('RT @[\w]*:','', text) # Removed RT 
    text = re.sub('@[A-Za-z0-9_]+', '', text) # Removed @mention
    text = re.sub('#', '', text) # hastag into text
    text = re.sub('&amp; ','',text) # Removed &(and) 
    text = re.sub("[^a-zA-Z0-9_']",' ',text) # remove punctuation
    text = re.sub('[0-9]*','', text) # Removed digits
   
    text = text.strip().lower()
    text = remove_stopwords(text)
    return text

for i in test:
    print(i)
    print("-"*10)
    print("clean_text \n")
    print(clean_text(i))
    print("\n")
    print("*"*10)
from nltk.stem import WordNetLemmatizer
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)])

lemmatize_text("Apples and oranges are similar. Boots and hippos aren't.")

df['clean_text'] = df['text'].apply(clean_text).apply(lemmatize_text)
df_test['text'] = df_test['text'].apply(clean_text).apply(lemmatize_text)
df['text']=df['clean_text']
df=df[['text','target']]
df.head()
df['target'].value_counts()
df.shape

# Term Frequency – Inverse Document
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
cv=CountVectorizer() 

count_vectorizer = CountVectorizer(stop_words=stop_words)
train_vectors = count_vectorizer.fit_transform(df['text']).toarray()

test_vectors = count_vectorizer.fit_transform(df_test['text']).toarray()

from collections import Counter 
words=Counter(count_vectorizer.vocabulary_)
words.most_common(5)
tf_idf_ngram = TfidfVectorizer(ngram_range=(1,2))
tf_idf_ngram.fit(df.text)
x_train_tf_bigram = tf_idf_ngram.transform(df.text)
x_test_tf_bigram = tf_idf_ngram.transform(df_test.text)
tf_idf_words=Counter(tf_idf_ngram.vocabulary_)
tf_idf_words.most_common(5)
x_train_tf_bigram.shape,x_test_tf_bigram.shape
from sklearn.model_selection import train_test_split

X = x_train_tf_bigram
y = df.target.values

# Train-Test Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.linear_model import LogisticRegression

clf_LR = LogisticRegression(C=2,dual=True, solver='liblinear',random_state=0)
clf_LR.fit(X_train,y_train)

y_pred_LR = clf_LR.predict(X_test)
from sklearn.metrics import accuracy_score , confusion_matrix, f1_score
accuracy = accuracy_score(y_test, y_pred_LR) * 100
f1score = f1_score(y_test, y_pred_LR) * 100

print(f"Logistic Regression Accuracy: {round(accuracy,2)} %")
print(f"Logistic Regression F1 Score: {round(f1score,2)} %")
submission =pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
submission['target']=clf_LR.predict(x_test_tf_bigram)
submission.to_csv("/kaggle/submission.csv",index=False)
submission.head()