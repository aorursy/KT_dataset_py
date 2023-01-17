# Loading data to Dataframe

import pandas as pd

data=pd.read_csv('../input/Reviews.csv')
data[['Text', 'Score']].head()
# Return the first 5 rows of dataset

data.head()
# Showing basic description of data

data.info()

data.describe()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.barplot(x=data['Score'].value_counts().keys(),y=data['Score'].value_counts())  # create barplot of Score

plt.xlabel('Score')

plt.ylabel('Count')
print(data.shape)                                # return shape of data

data.drop_duplicates(subset=['Text'])            # drop duplicates from Text column

print(data.shape)                                # return shape of data after dropping duplicates

data=data[data['Score']!=3]                      # remove comments with score equal 3 (the neutral one)

print(data.shape)                                # return shape of data after removing neutral comments
def labelling_data(data):

    if data['Score']>3:

        data['Sentiment']=1                # positive review

    else:

        data['Sentiment']=0                # negative review

    return data



data=data.apply(labelling_data,axis=1)

print(data.head())
data=data[['Text', 'Sentiment']]    # choose Text and Sentiment columns in Dataframe

data.head()
sns.barplot(x=data['Sentiment'].value_counts().keys(),y=data['Sentiment'].value_counts())      # create barplot of Score

plt.xlabel('Sentiment')

plt.ylabel('Count')
#BALANCING DATA



data_positive=data[data['Sentiment']==1].reset_index()                # creating dataframe with positive reviews 

print('Number of positive comments: ',data_positive.shape[0])         # number of positive comments



data_negative=data[data['Sentiment']==0].reset_index()                # creating dataframe with negative reviews

print('Number of negative comments: ',data_negative.shape[0])         # number of negative comments



#take a random sample of positive comments with number of comments equaled to number of negative comments

data_positive_sample=data_positive.sample(n=data_negative.shape[0])



#Merging sample of positive comments with negative comments

data=pd.concat([data_positive_sample, data_negative]).sample(frac=1).reset_index()

data=data[['Text', 'Sentiment']]

print('Total number of comments: ',data.shape)

print(data.head())

import re

import nltk

from nltk.corpus import stopwords



def cleaning_text(data):

    # extracting text from data

    text=data['Text']

    

    # lowering text

    text = text.lower() 

    

    # removing html tags

    cleanr = re.compile('<.*?>')

    text = re.sub(cleanr, ' ', text)

    

    # removing punctuations

    text = re.sub(r'[?|!|\'|"|#]',r'',text)

    text = re.sub(r'[.|,|)|(|\|/]',r' ',text)

    

    # stemming and removing stopwords

    snow = nltk.stem.SnowballStemmer('english')

    my_stopwords=stopwords.words('english')

    my_stopwords.append('i\'m')

    words = [snow.stem(word) for word in text.split() if word not in my_stopwords]

    

    # creating new column with cleaned text

    data['Cleaned text']=' '.join(words)

    

    return data

    



data_cleaned=data.apply(cleaning_text,axis=1)

print(data_cleaned.head())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_cleaned['Cleaned text'],data_cleaned['Sentiment'], test_size=0.2)



print('Number of comments in train data: ', X_train.shape[0])

print('Number of comments in test data: ', X_test.shape[0])
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#Countvectorizer

count_vect = CountVectorizer()

X_train_counts = count_vect.fit_transform(X_train)

X_test_counts = count_vect.transform(X_test)
print(X_train_counts.shape)
#TFIDF

tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

X_test_tfidf = tfidf_transformer.transform(X_test_counts)
print(X_train_tfidf.shape)
#Countvectorizer with n-grams

count_vect_ngrams = CountVectorizer(ngram_range=(1,3))

X_train_counts_ngrams = count_vect_ngrams.fit_transform(X_train)

X_test_counts_ngrams = count_vect_ngrams.transform(X_test)
print(X_train_counts_ngrams.shape)
#TFIDF with n-grams

tfidf_transformer = TfidfTransformer()

X_train_tfidf_ngrams = tfidf_transformer.fit_transform(X_train_counts_ngrams)

X_test_tfidf_ngrams = tfidf_transformer.fit_transform(X_test_counts_ngrams)
print(X_train_tfidf_ngrams.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
#Function which fit different models to the data and returns prediction made by these models

def models(X_train, y_train, X_test, y_test):

    prediction=dict()

    logreg=LogisticRegression(solver='liblinear').fit(X_train, y_train)

    prediction['Logistic'] = logreg.predict(X_test)

    

    KNN_3=KNN(n_neighbors=3).fit(X_train,y_train)

    prediction['KNN_3']=KNN_3.predict(X_test)

    

    KNN_5=KNN(n_neighbors=5).fit(X_train,y_train)

    prediction['KNN_5']=KNN_5.predict(X_test)

    

    MultiNB=MultinomialNB().fit(X_train,y_train)

    prediction['MultinomialNB']=MultiNB.predict(X_test)

    

    BernNB=BernoulliNB().fit(X_train,y_train)

    prediction['BernoulliNB']=BernNB.predict(X_test)

    

    return prediction
prediction_counts=models(X_train_counts, y_train, X_test_counts, y_test)
prediction_counts_ngrams=models(X_train_counts_ngrams, y_train, X_test_counts_ngrams, y_test)
prediction_tfidf=models(X_train_tfidf, y_train, X_test_tfidf, y_test)
prediction_tfidf_ngrams=models(X_train_tfidf_ngrams, y_train, X_test_tfidf_ngrams, y_test)
#evaluation - ROC

import pandas as pd

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

import numpy as np

import matplotlib.pyplot as plt



def roc(prediction,title):

    cmp = 0

    colors = ['b', 'g', 'y', 'm', 'k']

    for model, predicted in prediction.items():

        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predicted)

        roc_auc = auc(false_positive_rate, true_positive_rate)

        plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f'% (model,roc_auc))

        cmp += 1



    plt.title(title)

    plt.legend(loc='lower right')

    plt.plot([0,1],[0,1],'r--')

    plt.xlim([-0.1,1.2])

    plt.ylim([-0.1,1.2])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()
#Evaluating models which were trained on data countvectorized

roc(prediction_counts, 'Classifiers comparison with ROC for CountVectorizer')
#Evaluating models which were trained on data countvectorized with ngrams

roc(prediction_counts_ngrams, 'Classifiers comparison with ROC for CountVectorizer with ngrams')
#Evaluating models which were trained on tfidf representation of data

roc(prediction_tfidf, 'Classifiers comparison with ROC for TFIDF')
#Evaluating models which were trained on tfidf representation with ngrams of data

roc(prediction_tfidf_ngrams, 'Classifiers comparison with ROC for TFIDF with ngrams')
import matplotlib as mpl

import numpy as np

import matplotlib.pyplot as plt

import wordcloud

from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



mpl.rcParams['font.size']=12                

mpl.rcParams['savefig.dpi']=100             

mpl.rcParams['figure.subplot.bottom']=.1 



#functions taking colors from given ranges

def red_color_function(word, font_size, position,orientation,random_state=None, **kwargs):

    return("hsl(0,100%%, %d%%)" % np.random.randint(20,60))



def green_color_function(word, font_size, position,orientation,random_state=None, **kwargs):

    return("hsl(106,58%%, %d%%)" % np.random.randint(20,60))



#creating and plotting the wordcloud

def show_wordcloud(data, title = None):

    #Creating wordcloud from given text

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=200,

        max_font_size=50, 

        scale=3,

        random_state=1).generate(str(data))

    

    if title == 'negative reviews':   # use red scale of colour if opinion is negative

        wordcloud.recolor(color_func = red_color_function)

    elif title == 'positive reviews': # use green scale of colour if opinion is positive

        wordcloud.recolor(color_func = green_color_function)

        

    #plotting the wordcloud    

    fig = plt.figure(1, figsize=(15, 15))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)

    

    plt.imshow(wordcloud)

    plt.show()
#divide the text for 'positive' and 'nagetive'

data_words_positive = data_cleaned[data_cleaned['Sentiment']==1]['Cleaned text'].tolist() 

data_words_negative = data_cleaned[data_cleaned['Sentiment']==0]['Cleaned text'].tolist()



positive = ' '.join(data_words_positive)  #make vector from single words

negative = ' '.join(data_words_negative)
show_wordcloud(positive, title = 'positive reviews')
show_wordcloud(negative, title = 'negative reviews')