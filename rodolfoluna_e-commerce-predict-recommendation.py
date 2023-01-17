import pandas as pd

import matplotlib.pyplot as plt

from wordcloud import WordCloud
#Importing dataset

df = pd.read_csv("../input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv", index_col=0)
df.head(10)
pd.DataFrame.info(df)
reviewsDf = df[["Rating","Review Text"]]
reviewsDf.head()
pd.DataFrame.info(reviewsDf)
reviewsDf = reviewsDf.dropna(subset=['Review Text'])
reviewsDf.index = pd.Series(list(range(reviewsDf.shape[0])))
reviewsDf.head()
rev = reviewsDf['Review Text']



plt.subplots(figsize=(15,4))

wordcloud = WordCloud(background_color='white', width=900, height=300).generate(" ".join(rev))

plt.imshow(wordcloud)

plt.title('Words from Reviews\n',size=20)

plt.axis('off')

plt.show()
#Removing stop words.

from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer

import nltk

from nltk.corpus import wordnet

from nltk.stem import WordNetLemmatizer

!pip install nltk --upgrade

nltk.download('wordnet')



sw = set(stopwords.words('english'))



def preproc(data):

    #converting all to lowercase

    data = data.lower() 

    #Tokenize

    words = RegexpTokenizer(r'[a-z]+').tokenize(data)

    #Deleting stopwords

    words = [w for w in words if not w in sw]

    

    #Lemmatizing

    for pos in [wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV]:

        words = [WordNetLemmatizer().lemmatize(x, pos) for x in words]

    return " ".join(words)
reviewsDf['New Text'] = reviewsDf['Review Text'].apply(preproc)
reviewsDf.head()
def polarity (row):

  if row['Rating'] >= 4:

    return 'Positive'

  if row['Rating'] == 3:

    return 'Neutral'

  if row['Rating'] <= 2:

    return 'Negative'



reviewsDf['Class'] = reviewsDf.apply(lambda row: polarity(row), axis=1)
reviewsDf.head()
text, classe = reviewsDf["New Text"], reviewsDf["Class"]
text
classe
train_text = text[:16980]

test_text = text[16981:22640]

train_classe = classe[:16980]

test_classe = classe[16981:22640]
#Feature extraction



from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(encoding='latin-1')

X_train_counts = count_vect.fit_transform(train_text)

X_train_counts.shape
count_vect.vocabulary_.get('dress')
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer(use_idf=True)

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_train_tfidf.shape
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(X_train_tfidf, train_classe)
from sklearn.metrics import accuracy_score

X_test_counts = count_vect.transform(test_text)

X_test_tfidf = tfidf_transformer.transform(X_test_counts)

predito = clf.predict(X_test_tfidf)

gaussian_acc = accuracy_score(test_classe, predito)

print(gaussian_acc)
from sklearn.ensemble import RandomForestClassifier



ran = RandomForestClassifier(n_estimators=50)

ran.fit(X_train_tfidf, train_classe)
#Accuracy score

X_test_counts = count_vect.transform(test_text)

X_test_tfidf = tfidf_transformer.transform(X_test_counts)

predito = ran.predict(X_test_tfidf)

ran_acc = accuracy_score(test_classe, predito)

print(ran_acc)
from sklearn.svm import SVC



svm = SVC()

svm.fit(X_train_tfidf, train_classe)
#Accuracy score



X_test_counts = count_vect.transform(test_text)

X_test_tfidf = tfidf_transformer.transform(X_test_counts)

predito = svm.predict(X_test_tfidf)

svm_acc = accuracy_score(test_classe, predito)

print(svm_acc)
from sklearn.neural_network import MLPClassifier



nn = MLPClassifier()

nn.fit(X_train_tfidf, train_classe)
#Accuracy score



X_test_counts = count_vect.transform(test_text)

X_test_tfidf = tfidf_transformer.transform(X_test_counts)

predito = nn.predict(X_test_tfidf)

nn_acc = accuracy_score(test_classe, predito)

print(nn_acc)
from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(X_train_tfidf, train_classe)
#Accuracy score



X_test_counts = count_vect.transform(test_text)

X_test_tfidf = tfidf_transformer.transform(X_test_counts)

predito = lr.predict(X_test_tfidf)

lr_acc = accuracy_score(test_classe, predito)

print(lr_acc)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Neural Network'],

    'Score': [svm_acc, lr_acc, 

              ran_acc, gaussian_acc, nn_acc]})

models.sort_values(by='Score', ascending=False)