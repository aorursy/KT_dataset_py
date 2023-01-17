# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
hotel= pd.read_csv("/kaggle/input/hotel-reviews/7282_1.csv")

hotel.head()
hotel = hotel.loc[hotel['categories']=='Hotels']

# Drop rows with empty review rating

hotel = hotel[hotel['reviews.rating'].notna()] 
#Review text related to the 0.0 score

hotel.loc[hotel['reviews.rating']==0.0].iloc[0,14]
#I get rid of the 0.0 score since the reviews in this scores are only promts to write reviews.

hotel = hotel.loc[hotel['reviews.rating']!=0.0]

hotel = hotel.reset_index(drop = True)

hotel.sort_values(by = 'reviews.rating')
hotel_s = hotel[['name','reviews.rating','reviews.text']]

hotel_s["reviews.text"] = hotel_s["reviews.text"].apply(str)

#Review rating range from (1.0 - 10.0) 

hotel_s['reviews.rating'].unique()





#Create the label

#If a review rating is lower than 5, it is considered as a negative review; vice versa

hotel_s['is_bad_review'] = hotel_s['reviews.rating'].apply(lambda x:1 if x<5 else 0)

hotel_s
# return the wordnet object value corresponding to the POS tag

from nltk.corpus import wordnet

    

import string

from nltk import pos_tag

from nltk.corpus import stopwords

from nltk.tokenize import WhitespaceTokenizer

from nltk.stem import WordNetLemmatizer



def get_wordnet_pos(pos_tag):

    if pos_tag.startswith('J'):

        return wordnet.ADJ

    elif pos_tag.startswith('V'):

        return wordnet.VERB

    elif pos_tag.startswith('N'):

        return wordnet.NOUN

    elif pos_tag.startswith('R'):

        return wordnet.ADV

    else:

        return wordnet.NOUN
def clean_text(text):

    # lower text

    text = text.lower()

    # tokenize text and remove puncutation

    text = [word.strip(string.punctuation) for word in text.split(" ")]

    # remove words that contain numbers

    text = [word for word in text if not any(c.isdigit() for c in word)]

    # remove stop words

    stop = stopwords.words('english')

    text = [x for x in text if x not in stop]

    #remove empty tokens

    text = [t for t in text if len(t) > 0]

    #pos tag text

    pos_tags = pos_tag(text)

    # lemmatize text

    

    # the 'post_tag'function will return turples with the first word t[0] is the orginal word, and the second t[1] is the speech tag

    # for example, ('fly', 'NN')

    

    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]

    # remove words with only one letter

    text = [t for t in text if len(t) > 1]

    # join all

    text = " ".join(text)

    return(text)



# clean text data



hotel_s["review_clean"] = hotel_s["reviews.text"].apply(lambda x: clean_text(x))

hotel_s.dtypes

hotel_s
# Add sediment analysis columns (neg,neu,pos,compond)

from nltk.sentiment.vader import SentimentIntensityAnalyzer



#something in the reviews.text 



sid = SentimentIntensityAnalyzer()

hotel_s["sentiments"] = hotel_s["review_clean"].apply(lambda x: sid.polarity_scores(x))

#It turns out the hotel_s["sentiments"] values are dictionaries. Thus, I will need to transform it into 4 seperated column using pd.concat 

#(I have no idea how this function work)

hotel_s = pd.concat([hotel_s.drop(['sentiments'], axis=1), hotel_s['sentiments'].apply(pd.Series)], axis=1)



hotel_s
# Number of words in a review

hotel_s['num_words'] = hotel_s['reviews.text'].apply(lambda x: len(x.split()))



# Number of characters in a review

hotel_s['num_chars'] = hotel_s['reviews.text'].apply(lambda x: len(x))

#create doc2vec vector columns



from gensim.test.utils import common_texts

from gensim.models.doc2vec import Doc2Vec, TaggedDocument



# I think the tag here is for the whole review sentence.

documents =[TaggedDocument(doc,[i]) for i,doc in enumerate(hotel_s['reviews.text'].apply(lambda x: x.split()))]



# train a Doc2Vec model with our text data

model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)



# transform each document into a vector data

doc2vec_df = hotel_s["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)

#Rename doc2vec dataframe columns

doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]

hotel_s = pd.concat([hotel_s, doc2vec_df], axis=1)

hotel_s
# add tf-idfs columns



from sklearn.feature_extraction.text import TfidfVectorizer

# Ignore terms that have a document frequency strictly lower than 10

tfidf = TfidfVectorizer(min_df = 10)

# Learn vocabulary and idf, return term-document matrix (What is this)

tfidf_result = tfidf.fit_transform(hotel_s["review_clean"]).toarray()

# Named columns the names of the words

tfidf_df = pd.DataFrame(tfidf_result, columns = tfidf.get_feature_names())

tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]

tfidf_df.index = hotel_s.index

# Merge the hotel_s dataframe and tfidf dataframe

hotel_s = pd.concat([hotel_s, tfidf_df], axis=1)

hotel_s
pd.options.display.max_colwidth = 9999



# Compound score ranks from high-low, top 3 positive reviews



hotel_s[['reviews.text','review_clean','compound','is_bad_review']].sort_values('compound',ascending = False).head(3)
# Compound score ranks from low-high, top 3 negative reviews



hotel_s[['reviews.text','review_clean','compound','is_bad_review']].sort_values('compound',ascending = True).head(3)
#Set the stype

plt.style.use('seaborn')
# Hotels with highest compound scores



pos_review = hotel_s[hotel_s['compound']>=0.05]

top_pos_com = pos_review[['name','compound']].groupby(['name']).mean().sort_values('compound',ascending = True).tail(10).reset_index() 

# top_pos_com



plt.figure(figsize = (10,5))

plt.barh(top_pos_com['name'],top_pos_com['compound'])

plt.title('Top 10 Hotels with the Best Reviews',fontSize = 18)

plt.ylabel('Hotel Name',fontSize = 14)

plt.xlabel('Mean Compound Sentiment Score',fontSize = 14)

plt.figure(figsize = (10,2))
# Hotels with lowest compound scores



neg_review = hotel_s[hotel_s['compound']<= -0.05]

top_neg_com = neg_review[['name','compound']].groupby(['name']).mean().sort_values('compound',ascending = False).tail(10).reset_index() 



plt.figure(figsize = (10,5))

plt.barh(top_neg_com['name'],top_neg_com['compound'])

plt.title('Top 10 Hotels with the Worst Reviews',fontSize = 18)

plt.ylabel('Hotel Name',fontSize = 14)

plt.xlabel('Mean Compound Sentiment Score(negative numbers)',fontSize = 14)

plt.figure(figsize = (10,2))
from wordcloud import WordCloud



def show_wordCloud(text,title):

    wordcloud = WordCloud(

                    background_color ='white', 

                    max_words = 200,

                    min_font_size = 10,

                    min_word_length = 3).generate(str(text))



    plt.figure(figsize = (10,10)) 

    plt.title(title,fontsize =30, pad = 30)

    plt.imshow(wordcloud) 

    plt.axis("off") 

    plt.tight_layout(pad = 0) 



    plt.show()





# Wordcloud in positive reviews



show_wordCloud(pos_review['review_clean'],'WordCloud in Positive Reviews')
# Wordcloud in Negative reviews



show_wordCloud(neg_review['review_clean'],'WordCloud in Negative Reviews')
hotel_s[['compound']].hist(figsize = (10,8))

plt.xlabel('Compound Sentimemt Score')

plt.ylabel('Number of Reviews')

plt.title("Reviews' Compound Sentiment Score Distribution",fontsize =20)
label = 'is_bad_review'

ignore_columns = [label,'name','reviews.rating','reviews.text','review_clean']

features = [c for c in hotel_s.columns if c not in ignore_columns]



#Split the data into train and test

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split



# 80% train and 20% test

x_train, x_test, y_train, y_test = train_test_split(hotel_s[features], hotel_s[label], test_size = 0.20, random_state = 42) 



# Train model M1

rf = RandomForestClassifier(n_estimators = 100, random_state = 42)

rf.fit(x_train, y_train)

feature_importances_df = pd.DataFrame({"feature": features, "importance": rf.feature_importances_}).sort_values("importance", ascending = False)

feature_importances_df = feature_importances_df.head(100)



imp_feature = [f for f in feature_importances_df['feature']]



feature_importances_df
label = 'is_bad_review'



# Split train data and test data

# 80% train and 20% test

x2_train, x2_test, y2_train, y2_test = train_test_split(hotel_s[imp_feature], hotel_s[label], test_size = 0.20, random_state = 42) 



# Train model M2

rf2 = RandomForestClassifier(n_estimators = 100, random_state = 42)

rf2.fit(x2_train, y2_train)

# Use the M1 to Predict x_test

y_pred= rf.predict(x_test)



#Import scikit-learn metrics module for accuracy calculation



from sklearn import metrics



# Model Accuracy, how often is the classifier correct?



print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
y2_pred= rf2.predict(x2_test)

print("Accuracy:",metrics.accuracy_score(y2_test, y2_pred))
from sklearn.metrics import roc_curve, auc, roc_auc_score



# print(rf.predict_proba(x_test))



# x[0] means the probability that the testing sample is not classified as the first class 

# x[1] means the probability that the testing sample is classified as the first class



y_pred = [x[1] for x in rf.predict_proba(x_test)]

y2_pred = [a[1] for a in rf2.predict_proba(x2_test)]



fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label = 1)

fpr2, tpr2, thresholds2 = roc_curve(y2_test, y2_pred, pos_label = 1)



# print(thresholds)



roc_auc = auc(fpr, tpr)

roc_auc2 = auc(fpr2,tpr2)



plt.figure(1, figsize = (15, 10))

lw = 2



plt.plot(fpr, tpr, color='darkorange',

         lw=lw, label='ROC curve M1 (area = %0.2f)' % roc_auc) #0.2f means only print 2 digit of the float 

plt.plot(fpr2, tpr2, color='green',

         lw=lw, label='ROC curve M2 (area = %0.2f)' % roc_auc2)

plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic (ROC) Curve')

plt.legend(loc="lower right")

plt.show()