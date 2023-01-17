#Basic libraries

import pandas as pd 

import numpy as np 



#Visualization libraries

import matplotlib.pyplot as plt 

from matplotlib import rcParams

import seaborn as sns

from textblob import TextBlob

from plotly import tools

import plotly.graph_objs as go

from plotly.offline import iplot

%matplotlib inline

plt.rcParams['figure.figsize'] = [10, 5]

import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)



#NLTK libraries

import nltk

import re

import string

from nltk.corpus import stopwords

from wordcloud import WordCloud,STOPWORDS

from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

# Machine Learning libraries

import sklearn 

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import MultinomialNB 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

 



#Metrics libraries

from sklearn import metrics

from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve





#Miscellanous libraries

from collections import Counter



#Ignore warnings

import warnings

warnings.filterwarnings('ignore')
#reading the fake and true datasets

fake_news = pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv')

true_news = pd.read_csv('../input/fake-and-real-news-dataset/True.csv')



# print shape of fake dataset with rows and columns and information 

print ("The shape of the  data is (row, column):"+ str(fake_news.shape))

print (fake_news.info())

print("\n --------------------------------------- \n")



# print shape of true dataset with rows and columns and information

print ("The shape of the  data is (row, column):"+ str(true_news.shape))

print (true_news.info())
#Target variable for fake news

fake_news['output']=0



#Target variable for true news

true_news['output']=1
#Concatenating and dropping for fake news

fake_news['news']=fake_news['title']+fake_news['text']

fake_news=fake_news.drop(['title', 'text'], axis=1)



#Concatenating and dropping for true news

true_news['news']=true_news['title']+true_news['text']

true_news=true_news.drop(['title', 'text'], axis=1)



#Rearranging the columns

fake_news = fake_news[['subject', 'date', 'news','output']]

true_news = true_news[['subject', 'date', 'news','output']]
fake_news['date'].value_counts()
#Removing links and the headline from the date column

fake_news=fake_news[~fake_news.date.str.contains("http")]

fake_news=fake_news[~fake_news.date.str.contains("HOST")]



'''You can also execute the below code to get the result 

which allows only string which has the months and rest are filtered'''

#fake_news=fake_news[fake_news.date.str.contains("Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec")]

#Converting the date to datetime format

fake_news['date'] = pd.to_datetime(fake_news['date'])

true_news['date'] = pd.to_datetime(true_news['date'])
frames = [fake_news, true_news]

news_dataset = pd.concat(frames)

news_dataset
#Creating a copy 

clean_news=news_dataset.copy()
def review_cleaning(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = str(text).lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text
clean_news['news']=clean_news['news'].apply(lambda x:review_cleaning(x))

clean_news.head()
stop = stopwords.words('english')

clean_news['news'] = clean_news['news'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

clean_news.head()
#Plotting the frequency plot

ax = sns.countplot(x="subject", data=clean_news,

                   facecolor=(0, 0, 0, 0),

                   linewidth=5,

                   edgecolor=sns.color_palette("dark", 3))



#Setting labels and font size

ax.set(xlabel='Type of news', ylabel='Number of news',title='Count of news type')

ax.xaxis.get_label().set_fontsize(15)

ax.yaxis.get_label().set_fontsize(15)
g = sns.catplot(x="subject", col="output",

                data=clean_news, kind="count",

                height=4, aspect=2)



#Rotating the xlabels

g.set_xticklabels(rotation=45)
ax=sns.countplot(x="output", data=clean_news)



#Setting labels and font size

ax.set(xlabel='Output', ylabel='Count of fake/true',title='Count of fake and true news')

ax.xaxis.get_label().set_fontsize(15)

ax.yaxis.get_label().set_fontsize(15)
#Extracting the features from the news

clean_news['polarity'] = clean_news['news'].map(lambda text: TextBlob(text).sentiment.polarity)

clean_news['review_len'] = clean_news['news'].astype(str).apply(len)

clean_news['word_count'] = clean_news['news'].apply(lambda x: len(str(x).split()))



#Plotting the distribution of the extracted feature

plt.figure(figsize = (20, 5))

plt.style.use('seaborn-white')

plt.subplot(131)

sns.distplot(clean_news['polarity'])

fig = plt.gcf()

plt.subplot(132)

sns.distplot(clean_news['review_len'])

fig = plt.gcf()

plt.subplot(133)

sns.distplot(clean_news['word_count'])

fig = plt.gcf()
#Function to get top n words

def get_top_n_words(corpus, n=None):

    vec = CountVectorizer().fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]



#Calling function and return only top 20 words

common_words = get_top_n_words(clean_news['news'], 20)



#Printing the word and frequency

for word, freq in common_words:

    print(word, freq)



#Creating the dataframe of word and frequency

df1 = pd.DataFrame(common_words, columns = ['news' , 'count'])



#Group by words and plot the sum

df1.groupby('news').sum()['count'].sort_values(ascending=False).iplot(

    kind='bar', yTitle='Count', linecolor='black', title='Top 20 words in news')

#Function to get top bigram words

def get_top_n_bigram(corpus, n=None):

    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]



#Calling function and return only top 20 words

common_words = get_top_n_bigram(clean_news['news'], 20)



#Printing the word and frequency

for word, freq in common_words:

    print(word, freq)

    

#Creating the dataframe of word and frequency

df3 = pd.DataFrame(common_words, columns = ['news' , 'count'])



#Group by words and plot the sum

df3.groupby('news').sum()['count'].sort_values(ascending=False).iplot(

    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in news')

#Function to get top trigram words

def get_top_n_trigram(corpus, n=None):

    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]



#Calling function and return only top 20 words

common_words = get_top_n_trigram(clean_news['news'], 20)



#Printing word and their respective frequencies

for word, freq in common_words:

    print(word, freq)



#Creating a dataframe with words and count

df6 = pd.DataFrame(common_words, columns = ['news' , 'count'])



#Grouping the words and plotting their frequencies

df6.groupby('news').sum()['count'].sort_values(ascending=False).iplot(

    kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams in news')

text = fake_news["news"]

wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(text))

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
text = true_news["news"]

wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    stopwords = STOPWORDS).generate(str(text))

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
#Creating the count of output based on date

fake=fake_news.groupby(['date'])['output'].count()

fake=pd.DataFrame(fake)



true=true_news.groupby(['date'])['output'].count()

true=pd.DataFrame(true)



#Plotting the time series graph

fig = go.Figure()

fig.add_trace(go.Scatter(

         x=true.index,

         y=true['output'],

         name='True',

    line=dict(color='blue'),

    opacity=0.8))



fig.add_trace(go.Scatter(

         x=fake.index,

         y=fake['output'],

         name='Fake',

    line=dict(color='red'),

    opacity=0.8))



fig.update_xaxes(

    rangeslider_visible=True,

    rangeselector=dict(

        buttons=list([

            dict(count=1, label="1m", step="month", stepmode="backward"),

            dict(count=6, label="6m", step="month", stepmode="backward"),

            dict(count=1, label="YTD", step="year", stepmode="todate"),

            dict(count=1, label="1y", step="year", stepmode="backward"),

            dict(step="all")

        ])

    )

)

        

    

fig.update_layout(title_text='True and Fake News',plot_bgcolor='rgb(248, 248, 255)',yaxis_title='Value')



fig.show()
#Extracting 'reviews' for processing

news_features=clean_news.copy()

news_features=news_features[['news']].reset_index(drop=True)

news_features.head()
stop_words = set(stopwords.words("english"))

#Performing stemming on the review dataframe

ps = PorterStemmer()



#splitting and adding the stemmed words except stopwords

corpus = []

for i in range(0, len(news_features)):

    news = re.sub('[^a-zA-Z]', ' ', news_features['news'][i])

    news = news.split()

    news = [ps.stem(word) for word in news if not word in stop_words]

    news = ' '.join(news)

    corpus.append(news)   
corpus[3]
tfidf_vectorizer = TfidfVectorizer(max_features=5000,ngram_range=(2,2))

# TF-IDF feature matrix

X= tfidf_vectorizer.fit_transform(news_features['news'])

X.shape
#Getting the target variable

y=clean_news['output']
print(f'Original dataset shape : {Counter(y)}')
## Divide the dataset into Train and Test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    thresh = cm.max() / 2.

    for i in range (cm.shape[0]):

        for j in range (cm.shape[1]):

            plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
#creating the objects

logreg_cv = LogisticRegression(random_state=0)

dt_cv=DecisionTreeClassifier()

knn_cv=KNeighborsClassifier()

nb_cv=MultinomialNB(alpha=0.1) 

cv_dict = {0: 'Logistic Regression', 1: 'Decision Tree',2:'KNN',3:'Naive Bayes'}

cv_models=[logreg_cv,dt_cv,knn_cv,nb_cv]



#Printing the accuracy

for i,model in enumerate(cv_models):

    print("{} Test Accuracy: {}".format(cv_dict[i],cross_val_score(model, X, y, cv=10, scoring ='accuracy').mean()))
param_grid = {'C': np.logspace(-4, 4, 50),

             'penalty':['l1', 'l2']}

clf = GridSearchCV(LogisticRegression(random_state=0), param_grid,cv=5, verbose=0,n_jobs=-1)

best_model = clf.fit(X_train,y_train)

print(best_model.best_estimator_)

print("The mean accuracy of the model is:",best_model.score(X_test,y_test))
logreg = LogisticRegression(C=24.420530945486497, random_state=0)

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
cm = metrics.confusion_matrix(y_test, y_pred)

plot_confusion_matrix(cm, classes=['Negative','Neutral','Positive'])
print("Classification Report:\n",classification_report(y_test, y_pred))
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([-0.01, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.show()