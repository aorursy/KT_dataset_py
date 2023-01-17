# Import libararies

import re
import pandas as pd # CSV file I/O (pd.read_csv)
from nltk.corpus import stopwords
import numpy as np
import sklearn
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score ,confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
import warnings
warnings.filterwarnings('ignore')
news = pd.read_json('../input/News_Category_Dataset.json', lines=True)
#remove_columns_list = ['authors', 'date', 'link', 'short_description', 'headline']
news['information'] = news[['headline', 'short_description']].apply(lambda x: ' '.join(x), axis=1)
# Dataset dimension(row, columns)
news.shape
# To display entire text
pd.set_option('display.max_colwidth', -1)
news.head(1)
#news[['information', 'category']].head(5)
#news[news['authors'] == ''].groupby(by='category').size()
#news[(news['authors'] == '') & (news['short_description'] == '' )].index
# Drop those rows which has authors and short_description column as empty.
news.drop(news[(news['authors'] == '') & (news['short_description'] == '' )].index, inplace=True)
news.groupby(by='category').size()
fig, ax = plt.subplots(1, 1, figsize=(35,7))
sns.countplot(x = 'category', data = news)

fig, ax = plt.subplots(1, 1, figsize=(15,15))
news['category'].value_counts().plot.pie( autopct = '%1.1f%%')
#count the number of author in the dataset
#news.authors.value_counts()
total_authors = news.authors.nunique()
news_counts = news.shape[0]
print('Total Number of authors : ', total_authors)
print('avg articles written by per author: ' + str(news_counts//total_authors))
print('Total news counts : ' + str(news_counts))
authors_news_counts = news.authors.value_counts()
sum_contribution = 0
author_count = 0
for author_contribution in authors_news_counts:
    author_count += 1
    if author_contribution < 80:
        break
    sum_contribution += author_contribution
print('{} of news is contributed by {} authors i.e  {} % of news is contributed by {} % of authors'.
      format(sum_contribution, author_count, format((sum_contribution*100/news_counts), '.2f'), format((author_count*100/total_authors), '.2f')))
news.authors.value_counts()[0:10]
author_name = 'Lee Moran'
#author_name = 'Ed Mazza'
particular_author_news = news[news['authors'] == author_name]
df = particular_author_news.groupby(by='category')['information'].count()
df
fig, ax = plt.subplots(1, 1, figsize=(20,20))
df.plot.pie( autopct = '%1.1f%%')

# Split the data into train and test.
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(news[['information', 'authors']], news['category'], test_size=0.33)
# Convert pandas series into numpy array
X_train = np.array(X_train);
X_test = np.array(X_test);
Y_train = np.array(Y_train);
Y_test = np.array(Y_test);
cleanHeadlines_train = [] #To append processed headlines
cleanHeadlines_test = [] #To append processed headlines
number_reviews_train = len(X_train) #Calculating the number of reviews
number_reviews_test = len(X_test) #Calculating the number of reviews
from nltk.stem import PorterStemmer, WordNetLemmatizer
lemmetizer = WordNetLemmatizer()
stemmer = PorterStemmer()
def get_words(headlines_list):
    headlines = headlines_list[0]   
    author_names = [x for x in headlines_list[1].lower().replace('and',',').replace(' ', '').split(',') if x != '']
    headlines_only_letters = re.sub('[^a-zA-Z]', ' ', headlines)
    words = nltk.word_tokenize(headlines_only_letters.lower())
    stops = set(stopwords.words('english'))
    meaningful_words = [lemmetizer.lemmatize(w) for w in words if w not in stops]
    return ' '.join(meaningful_words + author_names)
for i in range(0,number_reviews_train):
    cleanHeadline = get_words(X_train[i]) #Processing the data and getting words with no special characters, numbers or html tags
    cleanHeadlines_train.append( cleanHeadline )
for i in range(0,number_reviews_test):
    cleanHeadline = get_words(X_test[i]) #Processing the data and getting words with no special characters, numbers or html tags
    cleanHeadlines_test.append( cleanHeadline )
vectorize = sklearn.feature_extraction.text.TfidfVectorizer(analyzer = "word", max_features=30000)
tfidwords_train = vectorize.fit_transform(cleanHeadlines_train)
X_train = tfidwords_train.toarray()

tfidwords_test = vectorize.transform(cleanHeadlines_test)
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
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB(alpha=0.1)
model.fit(X_train,Y_train)
Y_predict = model.predict(X_test)
accuracy = accuracy_score(Y_test,Y_predict)*100
print(format(accuracy, '.2f'))
# from sklearn.ensemble import BaggingClassifier
# model = BaggingClassifier(random_state=0, n_estimators=10)
# model.fit(X_train, Y_train)
# prediction = model.predict(X_test)
# print('Accuracy of bagged KNN is :',accuracy_score(prediction, Y_test))
# from sklearn.tree import DecisionTreeClassifier

# model = DecisionTreeClassifier()
# model.fit(X_train, Y_train)
# prediction_decision_tree = model.predict(X_test)
# print('The accuracy of Decision Tree is', accuracy_score(prediction_decision_tree, Y_test))
# from sklearn.neighbors import KNeighborsClassifier

# model = KNeighborsClassifier()
# model.fit(X_train, Y_train)
# prediction_knn = model.predict(X_test)
# print('The accuracy of the KNN is', metrics.accuracy_score(prediction_knn, Y_test))


# from sklearn.svm import SVC
# model = SVC(kernel='rbf',C=1,gamma=0.1)
# model.fit(X_train, Y_train)
# predict_rsvm = model.predict(X_test)
# print('Predict accuracy is ',accuracy_score(predict_rsvm,Y_test))
