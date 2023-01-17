import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import re



import nltk

from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.tokenize import RegexpTokenizer



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression
irish_data = pd.read_csv('../input/ireland-historical-news/irishtimes-date-text.csv')
print("The shape of the data is ",irish_data.shape)

irish_data.head()
stop_words = set(stopwords.words('english')) 

lem = WordNetLemmatizer()

lem.lemmatize('Ready')
def remove_stopwords(line):

    word_tokens = word_tokenize(line)  

    filtered_words = [re.sub('[^A-Za-z]+', '', w.lower()) for w in word_tokens if not w in stop_words]

    return filtered_words
irish_data['tokenized'] = irish_data['headline_text'].apply(lambda x: remove_stopwords(x))
get_words = []



def lemmatize(line):

    lem_words = []

    

    for word in line:

        lem_word = lem.lemmatize(word,"v")

        if len(lem_word) > 1:

            lem_words.append(lem_word)

            get_words.append(lem_word)

    

    return lem_words
irish_data['lemmatized'] = irish_data['tokenized'].apply(lambda x: lemmatize(x))
freq_words = nltk.FreqDist(get_words)

less_words = []



for word in freq_words:

    if freq_words[word]<=2:

        less_words.append(word)
def remove_less_words(lists):

    filters = []

    

    for word in lists:

        if word not in less_words:

            filters.append(word)

            

    return ' '.join(filters)
irish_data['filtered'] = irish_data['lemmatized'].apply(lambda x: remove_less_words(x))
irish_data.head()
#token = RegexpTokenizer(r'[a-zA-Z0-9]+')

#CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,2),tokenizer = token.tokenize)

cv = CountVectorizer(ngram_range = (1,1))

text_counts= cv.fit_transform(irish_data['filtered'])
X_train, X_test, y_train, y_test = train_test_split(text_counts, irish_data['headline_category'], test_size=0.3, random_state=101)
logistic_model = LogisticRegression()

logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)
accuracy_per = accuracy_score(y_test, y_pred)



print("Accuracy on the dataset: {:.2f}".format(accuracy_per*100))