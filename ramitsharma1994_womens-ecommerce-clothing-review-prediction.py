import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import nltk 
import string

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("../input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv")
data = data.iloc[:,2:]
data.head()
data.isnull().sum()
filter1 = data['Rating'] >= 4
df1 = data[filter1]

filter2 = data['Rating'] < 4
df2 = data[filter2]
df1
text1 = df1['Review Text'].str.lower().str.strip().str.cat(sep = " ")
text1 = text1.translate(str.maketrans("","", string.punctuation))

text2 = df2['Review Text'].str.lower().str.strip().str.cat(sep = " ")
text2 = text2.translate(str.maketrans("","", string.punctuation))
words_rating_above4 = nltk.word_tokenize(text1)
words_rating_below4 = nltk.word_tokenize(text2)
stopwords = nltk.corpus.stopwords.words("english")

li1 = []
li2 = []

word_net_lemmatizer = WordNetLemmatizer()

for word in words_rating_above4: 
    if(word in stopwords):
        continue
    li1.append(word_net_lemmatizer.lemmatize(word, pos='v'))
    
for word in words_rating_below4: 
    if(word in stopwords):
        continue
    li2.append(word_net_lemmatizer.lemmatize(word, pos='v'))
freq1 = nltk.FreqDist(li1)

freq2 = nltk.FreqDist(li2)

fig = plt.figure(figsize = (20,10))
plt.gcf().subplots_adjust() # to avoid x-ticks cut-off
freq1.plot(100, cumulative=False)
plt.show()

fig = plt.figure(figsize = (20,10))
plt.gcf().subplots_adjust() # to avoid x-ticks cut-off
freq2.plot(100, cumulative=False)
plt.show()

df1=df1.fillna("")
df2=df2.fillna("")
vectorizer = CountVectorizer(stop_words= stopwords)
matrix = vectorizer.fit_transform(df1['Review Text'])
feature_names = np.array(vectorizer.get_feature_names())
lda = LatentDirichletAllocation(n_components=10, learning_method="batch",
                                max_iter=200, random_state=0) 
document_topics = lda.fit_transform(matrix)
# Set n to your desired number of tokens 
n = 8
# Find top n tokens
topics = dict()
for idx, component in enumerate(lda.components_): 
    top_n_indices = component.argsort()[:-(n + 1): -1] 
    topic_tokens = [feature_names[i] for i in top_n_indices] 
    topics[idx] = topic_tokens

topics
vectorizer2 = CountVectorizer(stop_words= stopwords)
matrix2 = vectorizer2.fit_transform(df2['Review Text'])
feature_names2 = np.array(vectorizer2.get_feature_names())
lda2 = LatentDirichletAllocation(n_components=10, learning_method="batch",
                                max_iter=200, random_state=0) 
document_topics2 = lda2.fit_transform(matrix2)
# Set n to your desired number of tokens 
n = 8
# Find top n tokens
topics2 = dict()
for idx, component in enumerate(lda2.components_): 
    top_n_indices2 = component.argsort()[:-(n + 1): -1] 
    topic_tokens2 = [feature_names2[i] for i in top_n_indices2] 
    topics2[idx] = topic_tokens2

topics2
data.head()
data1 = data.drop(['Age', 'Positive Feedback Count', 'Division Name', 'Department Name'], axis = 1)

data1.head()
data1['Rating'] = data1['Rating'].apply(lambda x: 1 if(x>=4) else 0)

X = data1.drop(['Rating'], axis = 1)
y = data1['Rating']

X['Title'] = X['Title'].apply(lambda x: str(x))
X['Review Text'] = X['Review Text'].apply(lambda x: str(x))
X.head()
X['Title_Review'] = X['Title'] + " " + X['Review Text']
X['Title_Review'] = X['Title_Review'].apply(lambda x: x.replace("nan","").strip())
X = X.drop(['Review Text', "Title"], axis = 1)
X.head()
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob

word_net_lemmatizer = WordNetLemmatizer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

stopwords1 = STOP_WORDS
st2 = string.digits+string.punctuation

def lemmatize_text(text):
    lis=[]
    words = text.split()
    for word in words:
        if(word in stopwords1):
            continue
        lis.append(word_net_lemmatizer.lemmatize(word, pos='v'))
    return lis


X['Title_Review'] = X['Title_Review'].apply(lambda x: " ".join(lemmatize_text(x)))
X['Title_Review'] = X['Title_Review'].apply(lambda x: x.translate(str.maketrans("","", st2)))
X['Title_Review'] = X['Title_Review'].apply(lambda x: x.lower())

from textblob import TextBlob

X['Sentiment_Polarity'] = X['Title_Review'].apply(lambda x: TextBlob(x).sentiment[0])
X['Sentiment_Subjectivity'] = X['Title_Review'].apply(lambda x: TextBlob(x).sentiment[1])

from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(stop_words='english')
df_text1 = vect.fit_transform(X['Title_Review'])
df_text1 = pd.DataFrame(df_text1.toarray(), columns= vect.get_feature_names())

X1 = X.drop(['Title_Review'], axis = 1)
X1 = pd.concat([X1, df_text1], axis = 1)

X1 = pd.get_dummies(columns = ['Class Name'], data = X1)
X1 = X1.iloc[:, :-1]
X1
X_train, X_test, y_train, y_test = train_test_split(X1, y, random_state = 0, test_size = 0.3)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


rf = RandomForestClassifier()

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("accuracy score is: {}".format(accuracy_score(y_test, y_pred)))
print("Confusion matrix is:")
print((confusion_matrix(y_test, y_pred)))
print("Classification report is:")
print((classification_report(y_test, y_pred)))
X2 = X1.copy()
X2['Rating'] = data['Rating']
y2 = data['Recommended IND']
X2= X2.drop(['Recommended IND'], axis = 1)
X2
X_train, X_test, y_train, y_test = train_test_split(X2, y2, random_state = 0, test_size = 0.3)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


rf = RandomForestClassifier()

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("accuracy score is: {}".format(accuracy_score(y_test, y_pred)))
print("Confusion matrix is:")
print((confusion_matrix(y_test, y_pred)))
print("Classification report is:")
print((classification_report(y_test, y_pred)))




