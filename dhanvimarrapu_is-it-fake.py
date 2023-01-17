import pandas as pd
true_df = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')

fake_df = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
true_df.head()
fake_df.head()
true_df['check'] = 'TRUE'

fake_df['check'] = 'FAKE'
true_df.head()
fake_df.head()
df_news = pd.concat([true_df, fake_df])
df_news['article'] = df_news['title']+""+df_news['text']+""+['subject']
df = df_news[['article','check']]
df['article'] = df['article'].apply(lambda x: x.lower())
import string



def punctuation_removal(messy_str):

    clean_list = [char for char in messy_str if char not in string.punctuation]

    clean_str = ''.join(clean_list)

    return clean_str
df['article'] = df['article'].apply(punctuation_removal)

df['article'].head()
from nltk.corpus import stopwords

stop = stopwords.words('english')



df['article'].apply(lambda x: [item for item in x if item not in stop])
from sklearn.feature_extraction.text import CountVectorizer



bow_article = CountVectorizer().fit(df['article'])



article_vect = bow_article.transform(df['article'])
from sklearn.feature_extraction.text import TfidfTransformer



tfidf_transformer = TfidfTransformer().fit(article_vect)

news_tfidf = tfidf_transformer.transform(article_vect)

print(news_tfidf.shape)
from sklearn.model_selection import train_test_split

X = news_tfidf

y = df['check']







X_train, X_test, Y_train,Y_test= train_test_split(X, y, test_size=0.2)
from sklearn.naive_bayes import MultinomialNB



fakenews_detector = MultinomialNB().fit(X_train, Y_train)
predictions = fakenews_detector.predict(X_test)

print(predictions)
from sklearn.metrics import classification_report

print (classification_report(Y_test, predictions))
from sklearn.linear_model import SGDClassifier



fake_detector_svc = SGDClassifier().fit(X_train, Y_train)
prediction_svc = fake_detector_svc.predict(X_test)
print (classification_report(Y_test, prediction_svc))
from sklearn.linear_model import LogisticRegression



fake_detector_logistic = LogisticRegression().fit(X_train, Y_train)
predictions_log_reg = fake_detector_logistic.predict(X_test)

print (classification_report(Y_test, predictions_log_reg))