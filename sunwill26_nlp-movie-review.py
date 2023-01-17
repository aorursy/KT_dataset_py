import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# classic libraries

import pandas as pd

import numpy as np



# Charts

from wordcloud import WordCloud

import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter

import seaborn as sns



# cleaning text 

import nltk

from string import punctuation

from nltk.stem import SnowballStemmer

from sklearn.model_selection import train_test_split

from nltk import tokenize



# vectorizer for model

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer



# model

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB



# performance

from sklearn.metrics import confusion_matrix,accuracy_score



reviews=pd.read_csv("../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")

print(reviews.shape)

reviews.head()
# % of positive and negative reviews

rev_sent=reviews.groupby(['sentiment']).sentiment.count().to_frame('Count').reset_index()

plt.pie(rev_sent['Count'],labels=rev_sent['sentiment'],autopct='%1.1f%%',colors=('#e64040','#40a1e6'))

plt.title('Reviews %')

plt.show()
# Adding a column with binary classification positive (1) and negative (0)

reviews['sentiment_bin']=reviews['sentiment'].replace(['positive','negative'],[1,0])
# Cleaning text - 1



# stopwords

irrelevant_stuff=nltk.corpus.stopwords.words("english")

# stopwords + punctuation

punct=['br','/><','/>','10','15','20','30','80','.<']

for p in punctuation:

    punct.append(p)

irrelevant_stuff=irrelevant_stuff+punct



# function to split a sentence into a list of words

split_token=tokenize.WordPunctTokenizer()



# changes are: lower case / remove punctuation and stopwords

clean_text=[]

for opinion in reviews['review']:

    clean_list_words=[]

    list_words=split_token.tokenize(opinion)

    for word in list_words:

        if word.lower() not in irrelevant_stuff:

            clean_list_words.append(word.lower())

    clean_text.append(' '.join(clean_list_words))

reviews['clean_text_1']=clean_text

        
# Cleaning text - 2 



# Stemming - reducing word inflections to the root or origin

stemmer=SnowballStemmer("english")



# Stemming words

clean_text=[]

for opinion in reviews['clean_text_1']:

    clean_list_words=[]

    list_words=split_token.tokenize(opinion)

    for word in list_words:

        clean_list_words.append(stemmer.stem(word))

    clean_text.append(' '.join(clean_list_words))

reviews['clean_text_2']=clean_text
# comparing sentences

print(' '.join(reviews['review'][0].split()[0:10]))

print(' '.join(reviews['clean_text_1'][0].split()[0:10]))

print(' '.join(reviews['clean_text_2'][0].split()[0:10]))
# splitting data for training and testing

train,test,class_train,class_test =train_test_split(reviews['clean_text_2'],reviews['sentiment_bin'],random_state=100,train_size=0.6)
# Transforming reviews into vectors, results in a sparse matrix with words as columns and "sentence ID" in rows

# Bag of words

vectorize=CountVectorizer()

bag_of_words_train=vectorize.fit_transform(train)

bag_of_words_test=vectorize.transform(test)



# Term Frequency * Inverse Document Frequency model (TDIDF)

tfidf=TfidfVectorizer()

tf_train=tfidf.fit_transform(train)

tf_test=tfidf.transform(test)





# Just to have a rough idea what the vectorizer is doing

sparse_matrix=pd.DataFrame.sparse.from_spmatrix(bag_of_words_train,columns=vectorize.get_feature_names())

sparse_matrix



# model_1_1 Multinominal Naive Bayes Classifier

NBC=MultinomialNB()

model_NBC=NBC.fit(bag_of_words_train,class_train)

NBC_predict=model_NBC.predict(bag_of_words_test)

acc_NBC=accuracy_score(class_test,NBC_predict)

print(f'The accuary for NBC was {acc_NBC*100:.2f}%')
cm=confusion_matrix(class_test,NBC_predict)

print(cm)

print(f'There are {cm[0,0]} true negatives,{cm[1,0]} false negatives, {cm[1,1]} true positives and {cm[0,1]} false positives,')
# model_1_2 Multinominal Naive Bayes Classifier + tfidf

model_tf=NBC.fit(tf_train,class_train)

tf_predict=model_tf.predict(tf_test)

acc_NBC_tf=accuracy_score(class_test,tf_predict)

print(f'The accuary for NBC with TFIDF was {acc_NBC_tf*100:.2f}%')
cm=confusion_matrix(class_test,tf_predict)

print(cm)

print(f'There are {cm[0,0]} true negatives,{cm[1,0]} false negatives, {cm[1,1]} true positives and {cm[0,1]} false positives,')
# model_2_1 Logistic Regression

reg_log=LogisticRegression(solver="lbfgs")

model=reg_log.fit(bag_of_words_train, class_train)

predict_lr=model.predict(bag_of_words_test)

acc_lr=reg_log.score(bag_of_words_test,class_test)

print(f'The accuary for Log Reg was {acc_lr*100:.2f}%')
cm=confusion_matrix(class_test,predict_lr)

print(cm)

print(f'There are {cm[0,0]} true negatives,{cm[1,0]} false negatives, {cm[1,1]} true positives and {cm[0,1]} false positives,')
# model_2_2 Logistic Regression + tfidf

reg_log=LogisticRegression(solver="lbfgs")

model=reg_log.fit(tf_train, class_train)

predict_lr_tf=model.predict(tf_test)

acc_lr_tf=reg_log.score(tf_test,class_test)

print(f'The accuary for Log Reg with TFIDF was {acc_lr_tf*100:.2f}%')
cm=confusion_matrix(class_test,predict_lr_tf)

print(cm)

print(f'There are {cm[0,0]} true negatives,{cm[1,0]} false negatives, {cm[1,1]} true positives and {cm[0,1]} false positives,')
# I got his from matplotlib documentation

labels =['NBC','LR']



vec=[100*acc_NBC,100*acc_lr]

tf=[100*acc_NBC_tf,100*acc_lr_tf]



# label locations

x = np.arange(len(labels)) 

# bar width

width = 0.35  



fig, axs = plt.subplots()

axs.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))



m1 =axs.bar(x - width/2, vec, width, label='Vec')

m2 = axs.bar(x + width/2, tf, width, label='TF IDF')



axs.set_ylabel('Accuracy (%)')

axs.set_ylim([40, 100])

axs.set_title('Accuracy by Model and Vector Type')

axs.set_xticks(x)

axs.set_xticklabels(labels) 

axs.legend(loc=8)



def autolabel(ms):

    """Attach a text label above each bar in *rects*, displaying its height."""

    for m in ms:

        height = m.get_height()

        axs.annotate(f'{height:.3f}%',

                    xy=(m.get_x() + m.get_width() / 2, height),

                    xytext=(0, 3),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom')

autolabel(m1)

autolabel(m2)



fig.tight_layout()

plt.show()
# positive reviews

pos_rev=reviews.query("sentiment=='positive'")

pos_all_words=" ".join([text for text in pos_rev['clean_text_2']])

    

pos_rev_word_cloud=WordCloud(width=800, height=500,

                         max_font_size=110,

                         collocations=False).generate(pos_all_words)



# negative reviews

neg_rev=reviews.query("sentiment=='negative'")

neg_all_words=" ".join([text for text in neg_rev['clean_text_2']])   

neg_rev_word_cloud=WordCloud(width=800, height=500,

                         max_font_size=110,

                         collocations=False).generate(neg_all_words)



# Plotting

fig, axs=plt.subplots(1,2,figsize=(20,7))



# Pos

axs[0].imshow(pos_rev_word_cloud,interpolation='bilinear')

axs[0].set_title('Positive Reviews',size=15)

# Neg 

axs[1].imshow(neg_rev_word_cloud,interpolation='bilinear')

axs[1].set_title('Negative Reviews',size=15)



fig.suptitle('Wordcloud by Sentiment',size=20)

plt.show()
# Tokenizer and n most frequent words

split_token=tokenize.WhitespaceTokenizer()

n=10



# Positive review

pos_token=split_token.tokenize(pos_all_words)

pos_freq=nltk.FreqDist(pos_token)

pos_df_freq=pd.DataFrame({"Words":list(pos_freq.keys()),"Frequency":list(pos_freq.values())})

pos_df_freq=pos_df_freq.nlargest(n,'Frequency')



# Negative review

neg_token=split_token.tokenize(neg_all_words)

neg_freq=nltk.FreqDist(neg_token)

neg_df_freq=pd.DataFrame({"Words":list(neg_freq.keys()),"Frequency":list(neg_freq.values())})

neg_df_freq=neg_df_freq.nlargest(n,'Frequency')



# Plotting charts

fig, axs=plt.subplots(1,2,figsize=(10,5))



# pos

axs[0].bar(pos_df_freq['Words'],pos_df_freq['Frequency'])

axs[0].set_xticklabels(pos_df_freq['Words'], rotation=45)

axs[0].set_title('Positive Reviews',size=10)



# neg

axs[1].bar(neg_df_freq['Words'],neg_df_freq['Frequency'])

axs[1].set_xticklabels(neg_df_freq['Words'], rotation=45)

axs[1].set_title('Negative Reviews',size=10)





fig.suptitle('Word Frequency by Sentiment',size=15)

plt.show()
from PIL import Image

from IPython.display import Image
Image('../input/nlp-equations/NBC.JPG')
Image('../input/nlp-equations/Logistic_Regression.JPG')
Image('../input/nlp-equations/TFIDF.JPG')