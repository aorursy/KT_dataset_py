import pandas as pd
#Loading the datasets



true_df = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')

fake_df = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
true_df.head()
fake_df.head()
#Creating 'check' on both dfs that will be the target feature.



true_df['check'] = 'TRUE'

fake_df['check'] = 'FAKE'
true_df.head()
fake_df.head()
true_df.describe()
fake_df.describe()
#We will combine both dfs.



df_news = pd.concat([true_df, fake_df])
df_news.head(30)
df_news.info()
#Shuffling to see some Fakes



df_news.sample(frac = 1)
#Searching for null values.



df_news.isna().sum()
#We will join title, text and subject to create the article feature

df_news['article'] = df_news['title']+""+df_news['text']+""+['subject']
#Creating the final Dataframe with article and check.



df = df_news[['article','check']]
#Converting to lower case



df['article'] = df['article'].apply(lambda x: x.lower())
df['article'].head()
#Removing punctuation



import string



def punctuation_removal(messy_str):

    clean_list = [char for char in messy_str if char not in string.punctuation]

    clean_str = ''.join(clean_list)

    return clean_str
df['article'] = df['article'].apply(punctuation_removal)

df['article'].head()
#Removing stopwords



from nltk.corpus import stopwords

stop = stopwords.words('english')



df['article'].apply(lambda x: [item for item in x if item not in stop])
df['article']
%matplotlib inline



from wordcloud import WordCloud



all_words = ' '.join([text for text in df.article])



wordcloud = WordCloud(width= 800, height= 500,

                          max_font_size = 110,

                          collocations = False).generate(all_words)
import matplotlib.pyplot as plt



plt.figure(figsize=(10,7))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
#Function to generate wordcloud to True news.



def wordcloud_true(text, column_text):

    true_text = text.query("check == 'TRUE'")

    all_words = ' '.join([text for text in true_text[column_text]])



    wordcloud = WordCloud(width= 800, height= 500,

                              max_font_size = 110,

                              collocations = False).generate(all_words)

    plt.figure(figsize=(10,7))

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.show()
#Function to generate wordcloud to Fake news.



def wordcloud_fake(text, column_text):

    fake_text = text.query("check == 'FAKE'")

    all_words = ' '.join([text for text in fake_text[column_text]])



    wordcloud = WordCloud(width= 800, height= 500,

                              max_font_size = 110,

                              collocations = False).generate(all_words)

    plt.figure(figsize=(10,7))

    plt.imshow(wordcloud, interpolation='bilinear')

    plt.axis("off")

    plt.show()
#Wordcloud of the true news.



wordcloud_true(df, "article")
#Wordcloud of the fake news.



wordcloud_fake(df, "article")
from nltk import tokenize



token_space = tokenize.WhitespaceTokenizer()
import seaborn as sns

import nltk

    

def pareto(text, column_text, quantity):

    all_words = ' '.join([text for text in text[column_text]])

    token_phrase = token_space.tokenize(all_words)

    frequency = nltk.FreqDist(token_phrase)

    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),

                                   "Frequency": list(frequency.values())})

    df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity)

    plt.figure(figsize=(12,8))

    ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'blue')

    ax.set(ylabel = "Count")

    plt.show()
#The 20 more frequent words.



pareto(df, "article", 20)
#Lemmatization



'''from nltk.stem import WordNetLemmatizer 



lemmatizer = WordNetLemmatizer()



def lemmatize_text(text):

    return [lemmatizer.lemmatize(w) for w in df["article"]]



df['article'] = df["article"].apply(lemmatize_text)'''
from sklearn.feature_extraction.text import CountVectorizer



#Creating the bag of words

bow_article = CountVectorizer().fit(df['article'])



article_vect = bow_article.transform(df['article'])
#TF-IDF



from sklearn.feature_extraction.text import TfidfTransformer



tfidf_transformer = TfidfTransformer().fit(article_vect)

news_tfidf = tfidf_transformer.transform(article_vect)

print(news_tfidf.shape)
#We will use 20% of the data to train the models.



from sklearn.model_selection import train_test_split

X = news_tfidf

y = df['check']







X_train, X_test, Y_train,Y_test= train_test_split(X, y, test_size=0.2)
#Naive Bayes model

from sklearn.naive_bayes import MultinomialNB



fakenews_detector = MultinomialNB().fit(X_train, Y_train)
#Model Evaluation

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