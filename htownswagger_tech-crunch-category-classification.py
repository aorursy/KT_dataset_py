#Import pertinent libraries

import re

import pandas as pd # CSV file I/O (pd.read_csv)

from nltk.corpus import stopwords

import numpy as np #linear algebra

import sklearn #sci-kit ML

import nltk #natural language processing

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score ,confusion_matrix

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/techcrunch_posts.csv')

#cleaning the data

df["content"] = df["content"].str.replace("\\n", " ")

#dropping 'id', 'img_src', 'url'

df.drop(['id', 'img_src', 'url', 'date'], axis = 1, inplace =True)

df.dropna(subset=['authors'], inplace=True)
#create new column concatenating title and content 

df['info'] = df[['title', 'content']].apply(lambda x: ' '.join(str(value) for value in x), axis=1)

#remove content

df.drop(['content'], axis = 1, inplace =True)

#remove articles in which there are no authors

#display first row

pd.set_option('display.max_colwidth', -1)

df.head(1)
# pdf = df

# pdf['category']=pdf['category'].astype('category').cat.codes

# pdf['section']=pdf['section'].astype('category').cat.codes

# pdf['authors']=pdf['authors'].astype('category').cat.codes

# pdf.corr(method='pearson')
df.groupby(by="category", sort = False).size()
newdf = df.groupby('category').filter(lambda x : len(x)>150)
fig, ax = plt.subplots(1, 1, figsize=(35,10))

sns.countplot(x = "category", data = newdf)
fig, ax = plt.subplots(1, 1, figsize=(12,12))

newdf['category'].value_counts().plot.pie( autopct = '%1.1f%%')
total_authors = newdf.authors.nunique()

article_cnt = newdf.shape[0]

print('Total Number of authors : ', total_authors)

print('avg articles written by per author: ' + str(article_cnt//total_authors))

print('Total news counts : ' + str(article_cnt))
authors_article_cnt = newdf.authors.value_counts()

sum_articles = 0

author_cnt = 0

for author_articles in authors_article_cnt:

    author_cnt += 1

    if author_articles < 80:

        break

    sum_articles += author_articles

print('{} authors write {} articles, so {} % of authors produce {} % of Tech Crunch articles'.

      format(author_cnt, sum_articles, format((author_cnt*100/total_authors), '.2f'), format((sum_articles*100/article_cnt), '.2f')))
newdf.authors.value_counts()[0:5]
#Investigating the content of Tech Crunch's top author and filtering low category counts

author_name = 'Natasha Lomas'

author_articles_instance = newdf[newdf['authors'] == author_name]

authordf = author_articles_instance.groupby(by='category').filter(lambda x: len(x) > 10)

authordf = authordf.groupby(by='category').size()

authordf
#Observing the most popular genres for Sarah Perez, we note that author likely corresponds to category

fig, ax = plt.subplots(1, 1, figsize=(10,10))

authordf.plot.pie( autopct = '%1.1f%%')
# Split the data into train and test.

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(newdf[['info', 'authors']], newdf['category'], test_size=0.33)
# Convert pandas series into numpy array

X_train = np.array(X_train);

X_test = np.array(X_test);

Y_train = np.array(Y_train);

Y_test = np.array(Y_test);

cleanTitles_train = [] #To append processed titles

cleanTitles_test = [] #To append processed titles

number_reviews_train = len(X_train) #Calculating the number of reviews

number_reviews_test = len(X_test) #Calculating the number of reviews
from nltk.stem import PorterStemmer, WordNetLemmatizer

lemmetizer = WordNetLemmatizer()

stemmer = PorterStemmer()

def get_words(titles_list):

    titles = titles_list[0]   

    author_names = [x for x in titles_list[1].lower().replace('and',',').replace(' ', '').split(',') if x != '']

    titles_only_letters = re.sub('[^a-zA-Z]', ' ', titles)

    words = nltk.word_tokenize(titles_only_letters.lower())

    stops = set(stopwords.words('english'))

    meaningful_words = [lemmetizer.lemmatize(w) for w in words if w not in stops]

    return ' '.join(meaningful_words + author_names)
#cleaning excess data in X_train

for i in range(0,1063):

    np.delete(X_train, len(X_train)-i)
for i in range(0,number_reviews_train):

    cleanTitle = get_words(X_train[i]) #Processing the data and getting words with no special characters, numbers or html tags

    cleanTitles_train.append( cleanTitle )
for i in range(0,number_reviews_test):

    cleanTitle = get_words(X_test[i]) #Processing the data and getting words with no special characters, numbers or html tags

    cleanTitles_test.append( cleanTitle )
vectorize = sklearn.feature_extraction.text.TfidfVectorizer(analyzer = "word", max_features=1000)

tfidwords_train = vectorize.fit_transform(cleanTitles_train)

X_train = tfidwords_train.toarray()



tfidwords_test = vectorize.transform(cleanTitles_test)

X_test = tfidwords_test.toarray()
from sklearn.svm import LinearSVC



model = LinearSVC()

model.fit(X_train,Y_train)

Y_predict = model.predict(X_test)

accuracy = accuracy_score(Y_test,Y_predict)*100

print(format(accuracy, '.2f'))
logistic_Regression = LogisticRegression()

logistic_Regression.fit(X_train,Y_train)

Y_predict = logistic_Regression.predict(X_test)

accuracy = accuracy_score(Y_test,Y_predict)*100

print(format(accuracy, '.2f'))
from sklearn.ensemble import BaggingClassifier

model = BaggingClassifier(random_state=0, n_estimators=10)

model.fit(X_train, Y_train)

prediction = model.predict(X_test)

print('Accuracy of bagged KNN is :', accuracy_score(prediction, Y_test)*100, '%')
from sklearn.naive_bayes import MultinomialNB



model = MultinomialNB(alpha=0.1)

model.fit(X_train,Y_train)

Y_predict = model.predict(X_test)

accuracy = accuracy_score(Y_test,Y_predict)*100

print(format(accuracy, '.2f'))