
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt
import string

data  = pd.read_csv("../input/boardgamegeek-reviews/bgg-13m-reviews.csv")
data=data.drop(["Unnamed: 0"], axis = 1)
data.head(5)
print("Total No of reviews ",data.shape[0] )
df = data.dropna()
data=df.reset_index()
data=data.drop(["index"], axis = 1)
data.head(5)
decimals = pd.Series([0], index=['rating'])
data = data.round(decimals)
#Ratings wise distribution
data["rating"].value_counts()
plot_imbal_labels = data.groupby(["rating"]).size()
plot_imbal_labels = plot_imbal_labels / plot_imbal_labels.sum()
fig, ax = plt.subplots(figsize=(12,8))
sns.barplot(plot_imbal_labels.keys(), plot_imbal_labels.values).set_title("Before Down Sampling")
ax.set_ylabel('Number of samples')
import seaborn as sns
import numpy as np 
import warnings
warnings.filterwarnings("ignore")
sns.set_style("darkgrid")

data['text length'] = data.comment.apply(len)
stars_10 = data[data["rating"]==10.0]
print(stars_10.shape)
stars_10_downsample = resample(stars_10, replace=False, n_samples=20000, random_state=0)
stars_10_downsample.shape
stars_9 = data[data["rating"]==9.0]
print(stars_9.shape)
stars_9_downsample = resample(stars_9, replace=False, n_samples=20000, random_state=0)
stars_9_downsample.shape
stars_8 = data[data["rating"]==8.0]
print(stars_8.shape)
stars_8_downsample = resample(stars_8, replace=False, n_samples=20000, random_state=0)
stars_8_downsample.shape
stars_7 = data[data["rating"]==7.0]
print(stars_7.shape)
stars_7_downsample = resample(stars_7, replace=False, n_samples=20000, random_state=0)
stars_7_downsample.shape
stars_6 = data[data["rating"]==6.0]
print(stars_6.shape)
stars_6_downsample = resample(stars_6, replace=False, n_samples=20000, random_state=0)
stars_6_downsample.shape
stars_5 = data[data["rating"]==5.0]
print(stars_5.shape)
stars_5_downsample = resample(stars_5, replace=False, n_samples=20000, random_state=0)
stars_5_downsample.shape
stars_4 = data[data["rating"]==4.0]
print(stars_4.shape)
stars_4_downsample = resample(stars_4, replace=False, n_samples=20000, random_state=0)
stars_4_downsample.shape
stars_3 = data[data["rating"]==3.0]
print(stars_5.shape)
stars_3_downsample = resample(stars_3, replace=False, n_samples=20000, random_state=0)
stars_3_downsample.shape
stars_2 = data[data["rating"]==2.0]
print(stars_2.shape)
stars_2_downsample = resample(stars_2, replace=False, n_samples=20000, random_state=0)
stars_2_downsample.shape
stars_1 = data[data["rating"]==1.0]
print(stars_1.shape)
stars_1_downsample = resample(stars_1, replace=False, n_samples=20000, random_state=0)
stars_1_downsample.shape
#Concatening the samples and defining the dataframe
reviews_ds = pd.concat([stars_1_downsample,stars_2_downsample,stars_3_downsample,stars_4_downsample,stars_5_downsample,stars_6_downsample,stars_7_downsample,stars_8_downsample,stars_9_downsample,stars_10_downsample])
reviews_ds.shape
reviews_ds['rating'].value_counts()
plot_bal_labels = reviews_ds.groupby(["rating"]).size()
plot_bal_labels = plot_bal_labels / plot_bal_labels.sum()
fig, ax = plt.subplots(figsize=(12,8))
sns.barplot(plot_bal_labels.keys(), plot_bal_labels.values).set_title("After Down Sampling")
ax.set_ylabel('Proportion of samples')
reviews_ds['comment']
d = {'whom', "you're", 'but', 'at', 'more', 'themselves', 'couldn', 'why', 'few', "you've", 'your',
        'doesn', 'before', 'him', 'she', 'don', 'what', 'are', 'doing', 'theirs', 'all', 've', 'into', 'player',
        'himself', 'same', 'has', 'above', 'was', 'where', "wasn't", 'the', 'just', 'again', 'm', 'isn',  'de',
        'did', 'does', 'or', "won't", 'yourself', 'here', 'll', 'through', 'because', 'about', 'which', 'watching',
        'ours', 'herself', 'my', 'this', 'how', 'during', 'until', 'between', 'aren', 'ma', 'be', 'of', 'going',
        'some', 'wasn', 'too', "didn't", "isn't", "doesn't", 'then', 'itself', "mightn't", 'd', 'from',
        'them', 'can', 'each', 'no', 'ourselves', 'won', 'ain', 'yours', 'myself', 'such', 'to', "needn't", 
         'mustn', "should've", 'when', 'weren', 'shouldn', 'haven', "mustn't", 'in', 'we', 'down', 'me',
        'against', 'both', 'needn', 'those', 'an', 'only', "that'll", 'hers', 'with', 'by', 'will', 'people',
        'wouldn', 'had', 'while', 'out', "you'll", 'not', 't', "weren't", "hadn't", 'hadn', 'if', 's', 'bgg',
        "hasn't", 'mightn', 'any', 'their', 'were', 'having', 'now', 'hasn', 'it', 'so', 'its', 'he',
        'y', 'should', 'you', 'me', 'yourselves', 'a', 'off', "haven't", "you'd", 'i', 'our', 'who', "shan't", 
         'further', 'is', 'very', 'her', "shouldn't", 'am', 'o', 're', 'over', 'shan', 'once', 'for', 'also',
        'been', 'there', 'own', "shan't", 'on', 'down', 'do', "wouldn't", 'his', 'these', 'most', 'that',
        "it's", 'and', 'nor', "don't", 'other', "she's", 'after', 'below', 'didn', "aren't", 'they',
        'being', "couldn't", 'have', 'than', 'up', 'as', 'under', 'film','movie','one','time','see','story','well',
         'like','even','good','also''first','get','much','first','point', 'box', 'others', 'mechanic', 'felt', 'tile',
        'plot','films','many','movies','made','acting','thing','way','think','character','did', 'such', 'doing', 
       'just', 'very', 'shan', 'against', 't', "you're", 'who', 'than', 'br','music','however','must','take','big',
          'only', "haven't", 'yours', 'you', 'its', 'other', 'we', 'where', 'then', 'they', 'won', "you've",
          'some', 've', 'y', 'each', "you'll", 'them', 'to', 'was', 'once', 'and', 'ain', 'under', 'through',
          'for', "won't", 'mustn', 'a', 'are', 'that', 'at', 'why', 'any', 'nor', 'these', 'yourselves', 'board',
          'has', 'here', "needn't", 'm', 'above', 'up', 'more', 'if', 'ma', 'didn', 'whom', 'can', 'have',
          'an', 'should', 'there', 'couldn', 'her', 'how', 'of', 'doesn', "shouldn't", 'further', 'rule',
          "wasn't", 'between', 'd', 'wouldn', 'his', 'being', 'do', 'when', 'hasn', "she's", 'by', "should've",
          'into', 'aren', 'weren', 'as', 'needn', 'what', "it's", 'hadn', 'with', 'after', 'he', 'off', 'not',
          'does', 'own', "weren't", "isn't", 'my', 'too', "wouldn't", 'been', 'again', 'same', 'few', "don't",
          'our', 'myself', 'your', 'before', 'about', 'most', 'during', 'll', 'on', 'shouldn', 'is', 'out',
         'below', 'which', 'from', 'she', 'were', 'those', 'over', 'until', 'theirs', 'mightn', 'random', 'nostar',
          'yourself', 'i', 'am', 'so', 'himself', 'it', 'had', 'or', 'all', 'while', "aren't", 'ours', 'strategy',
          "that'll", 'but', 'because', 'in', 'now', 'themselves', 'him', "doesn't", 'both', 're', 'wasn', 'theme', 
          's', "hasn't", "didn't", 'their', "mustn't", 'herself', 'the', 'this', 'will', 'isn', "you'd", 'game', 
          'haven', 'itself', "couldn't", 'o', 'be', 'don', 'hers', "mightn't", 'having', "hadn't", 'ourselves',
        'characters','watch','could','would','really','two','man','show','seen','still','never','make','little',
        'life','years','know','say','end','ever','scene','real','back','though','world','go','new','something',
       'scenes','nothing','makes','work','young','old','find','us','funny','actually','another','actors','director',
       'series','quite','cast','part','always','lot','look','love','horror','want','minutes','pretty','better','great',
       'best','family','may','role','every', 'performance','bad','things','times','bad','great','best','script','every',
       'seems','least','enough','original','action','bit','comedy','saw', 'long','right','fun','fact','around','guy', 'got',
       'anything','point', 'give','thought', 'whole', 'gets', 'making','without','day', 'feel', 'come','played','almost',
      'might', 'money', 'far', 'without', 'come', 'almost','kind', 'done','especially', 'yet', 'last', 'since', 'different',
       'although','true','interesting', 'reason', 'looks', 'done', 'someone', 'trying','job', 'shows', 'woman', 'tv', 
       'probably', 'father', 'girl', 'plays', 'instead', 'away', 'girl', 'probably', 'believe', 'sure', 'course', 'john', 
       'rather', 'later', 'dvd', 'war', 'found', 'looking', 'anyone', 'maybe', 'rather', 'let', 'screen', 'year', 'hard', 
       'together', 'set', 'place','comes', 'half', 'idea', 'american', 'play', 'takes', 'performances', 'everyone','actor',
     'wife', 'goes','sense', 'book', 'ending', 'version', 'star', 'everything', 'put', 'said', 'seeing', 'house', 'main'
     , 'three' ,'watched', 'high', 'else', 'men', 'night','need', 'try', 'kid','prefer', 'group', 'system', 'game','card'
        ,'playing','turn','dice','roll','move','hour','draw', 'le', 'deck', 'component', 'gameplay', 'choice', 'design'
        ,'size', 'hand', 'number', 'add', 'keep', 'chance', 'add', 'ok', 'gave', 'round', 'win', 'decision', 'experience'
         ,'oh','used','type','basically','next', 'update', 'url', 'seem', 'building','child','completely','either',
         'simply','easy', 'second', 'rolling','opponent', 'start','guess','space', 'understand','tried', 'artwork',
         'quality', 'randomness', 'scoring', 'map', 'element', 'remember', 'bought', 'gaming', 'totally','light','rating',
         'nice', 'piece', 'buy', 'art', 'use', 'wanted','combat','based', 'read', 'word', 'order','use', 'getting','control'
        ,'monopoly','party','friend','table','value','interaction','question','mechanism','small','puzzle', 'rate', 'amount'
         , 'single', 'resource', 'score', 'care', 'able', 'designer', 'took', 'level','seemed', 'help', 'non', 'given'
        ,'person', 'avoid', 'copy', 'taking','edition', 'rulebook', 'color','quickly', 'change', 'sold','complete'
        ,'top','sort', 'event', 'mind', 'http', 'couple', 'ship', 'com', 'unit', 'concept', 'due', 'aspect','battle','often',
         'victory','short', 'special', 'huge','early', 'attack','option','que', 'unplayable', 'winning', 'review', 'run','low',
         'cannot', 'unless', 'designed','comment', 'pure', 'la','name', 'answer','except', 'ability', 'hate', 'absolutely',
     'abstract', 'die', 'adult', 'figure', 'actual', 'age', 'area', 'came'}
# to clear text from jargon words for better accuraccy

from nltk.corpus import stopwords
def clear_text(text):

    stop = set(stopwords.words('english'))
    
    stop.update(d)
    
    clean_tokens = [tok for tok in text if len(tok.lower())>1 and (tok.lower() not in stop)]

    pos_list = clean_tokens
    
    return pos_list


#Basic Data Cleaning fucntion

c = 0

def text_preprocess(given_review):
    review = given_review
    review = re.sub('[^a-zA-Z]', ' ',review)
    review = review.lower()
    review = review.split()
    lemmatizer = WordNetLemmatizer()
    stop = set(stopwords.words('english'))
    stop.update(d)
    review = [lemmatizer.lemmatize(w) for w in review if not w in stop]
    global c
    c+=1
    print(c)
    return (' '.join(review))
 
reviews_ds['cleaned_text'] = reviews_ds['comment'].apply(text_preprocess)

reviews_clean = reviews_ds.copy()
reviews_clean = reviews_clean.dropna(axis=0, subset=['cleaned_text'])

from collections import Counter
l1= reviews_ds[reviews_ds["rating"]==1.0]
l1=l1.cleaned_text
a = " ".join(l1).split()
list1= clear_text(a)
str_1=set(Counter(list1).most_common(100))
star_1=[]
for e in str_1:
    star_1.append(e)
l2= reviews_ds[reviews_ds["rating"]==2.0]
l2=l2.cleaned_text
a = " ".join(l2).split()
list2= clear_text(a)
str_2=sorted(set(Counter(list2).most_common(100)))
star_2=[]
for e in str_2:
    star_2.append(e)
l3= reviews_ds[reviews_ds["rating"]==3.0]
l3=l3.cleaned_text
a = " ".join(l3).split()
list3= clear_text(a)
str_3=sorted(set(Counter(list3).most_common(100)))
star_3=[]
for e in str_3:
    star_3.append(e)
l4= reviews_ds[reviews_ds["rating"]==4.0]
l4=l4.cleaned_text
a = " ".join(l4).split()
list4= clear_text(a)
str_4=sorted(set(Counter(list4).most_common(100)))
star_4=[]
for e in str_4:
    star_4.append(e)
l5= reviews_ds[reviews_ds["rating"]==5.0]
l5=l5.cleaned_text
a = " ".join(l5).split()
list5= clear_text(a)
str_5=sorted(set(Counter(list5).most_common(100)))
star_5=[]
for e in str_5:
    star_5.append(e)
l6= reviews_ds[reviews_ds["rating"]==6.0]
l6=l6.cleaned_text
a = " ".join(l6).split()
list6= clear_text(a)
str_6=sorted(set(Counter(list6).most_common(100)))
star_6=[]
for e in str_6:
    star_6.append(e)
l7= reviews_ds[reviews_ds["rating"]==7.0]
l7=l7.cleaned_text
a = " ".join(l7).split()
list7= clear_text(a)
str_7=sorted(set(Counter(list7).most_common(100)))
star_7=[]
for e in str_7:
    star_7.append(e)
l8= reviews_ds[reviews_ds["rating"]==8.0]
l8=l8.cleaned_text
a = " ".join(l8).split()
list8= clear_text(a)
str_8=sorted(set(Counter(list2).most_common(100)))
star_8=[]
for e in str_8:
    star_8.append(e)
l9= reviews_ds[reviews_ds["rating"]==9.0]
l9=l9.cleaned_text
a = " ".join(l9).split()
list9= clear_text(a)
str_9=sorted(set(Counter(list9).most_common(100)))
star_9=[]
for e in str_9:
    star_9.append(e)
l10= reviews_ds[reviews_ds["rating"]==10.0]
l10=l10.cleaned_text
a = " ".join(l10).split()
list10= clear_text(a)
str_10=sorted(set(Counter(list10).most_common(100)))
star_10=[]
for e in str_10:
    star_10.append(e)
neg_words=[]
for i in range(len(star_1)):
    neg_words.append(star_1[i][0])
    neg_words.append(star_2[i][0])
    neg_words.append(star_3[i][0])
    neg_words.append(star_4[i][0])
    neg_words.append(star_5[i][0])
# convert the set to the list 
unique_neg_list = [] 
      
# traverse for all elements 
for x in neg_words: 
# check if exists in unique_list or not 
    if x not in unique_neg_list: 
        unique_neg_list.append(x) 

    
print(unique_neg_list)
pos_words=[]
for i in range(len(star_1)):
    pos_words.append(star_10[i][0])
    pos_words.append(star_9[i][0])
    pos_words.append(star_8[i][0])
    pos_words.append(star_7[i][0])
    pos_words.append(star_6[i][0])
# convert the set to the list 
unique_pos_list = [] 
      
# traverse for all elements 
for x in pos_words: 
# check if exists in unique_list or not 
    if x not in unique_pos_list: 
        unique_pos_list.append(x) 

    
print(unique_pos_list)
"""
EDA Before Data Cleaning 

"""

import numpy as np
from nltk.util import ngrams
from collections import Counter


print('Average word length of phrases in corpus is:',np.mean(reviews_ds['comment'].apply(lambda x: len(x.split()))))


def MostCommonPostEDA(star, no):
    text = ' '.join(reviews_ds.loc[reviews_ds.rating == star, 'comment'].values)
    text_unigrams = [i for i in ngrams(text.split(), 1)]
    text_bigrams = [i for i in ngrams(text.split(), 2)]
    text_trigrams = [i for i in ngrams(text.split(), 3)]
    print("The most common words in rating",star,"\n", Counter(text_unigrams).most_common(no))
    print("The most common bigrams in rating",star, "\n",Counter(text_bigrams).most_common(no))
    print("The most common trigrams in rating",star, "\n",Counter(text_trigrams).most_common(no))
    return Counter(text_unigrams).most_common(no),Counter(text_bigrams).most_common(no),Counter(text_trigrams).most_common(no)

'''
Citation: https://stackoverflow.com/questions/13925251/python-bar-plot-from-list-of-tuples

'''

def plot_grams(title,ylab,lis):
  # sort in-place from highest to lowest
  lis.sort(key=lambda x: x[1], reverse=True) 

  # save the names and their respective scores separately
  # reverse the tuples to go from most frequent to least frequent 
  grams = list(zip(*lis))[0]
  count = list(zip(*lis))[1]
  x_pos = np.arange(len(grams)) 

  # calculate slope and intercept for the linear trend line
  slope, intercept = np.polyfit(x_pos, count, 1)
  trendline = intercept + (slope * x_pos)
  plt.plot(x_pos, trendline, color='red', linestyle='--')    
  plt.bar(x_pos, count,align='center')
  plt.xticks(x_pos, grams, rotation=70) 
  plt.ylabel(ylab)
  plt.title(title)
  plt.show()


all_x,all_y,all_z = [],[],[]
stars = ["Star 1","Star 2","Star 3","Star 4"," Star 5"," Star 6"," Star 7"," Star 8"," Star 9"," Star 10"]
#All the common words, unigrams and bigrams in all the sentiments
for i in [1,2,3,4,5]:
    x,y,z = MostCommonPostEDA(i,3)
    all_x.append(x)
    all_y.append(y)
    all_z.append(z)

for x,i in zip(all_x,stars):
  plot_grams(i,"Most Common Words",x)

for y,i in zip(all_y,stars):
  plot_grams(i,"Bi-grams",y)

for z,i in zip(all_z,stars):
  plot_grams(i,"Tri-grams",z)

'''
Post Cleaning EDA

'''


print('Average word length of phrases in corpus is:',np.mean(reviews_clean['cleaned_text'].apply(lambda x: len(x.split()))))


def MostCommonPostEDA(star, no):
    text = ' '.join(reviews_clean.loc[reviews_clean.rating == star, 'cleaned_text'].values)
    text_unigrams = [i for i in ngrams(text.split(), 1)]
    text_bigrams = [i for i in ngrams(text.split(), 2)]
    text_trigrams = [i for i in ngrams(text.split(), 3)]
    print("The most common words in star",star,"\n", Counter(text_unigrams).most_common(no))
    print("The most common bigrams in star",star, "\n",Counter(text_bigrams).most_common(no))
    print("The most common trigrams in star",star, "\n",Counter(text_trigrams).most_common(no))
    return Counter(text_unigrams).most_common(no),Counter(text_bigrams).most_common(no),Counter(text_trigrams).most_common(no)




all_x,all_y,all_z = [],[],[]
stars = ["Cleaned Star 1","Cleaned Star 2","Cleaned Star 3","Cleaned Star 4","Cleaned Star 5","Cleaned Star 6","Cleaned Star 7","Cleaned Star 8","Cleaned Star 9","Cleaned Star 10"]
#All the common words, unigrams and bigrams in all the sentiments
for i in [1,2,3,4,5,6,7,8,9,10]:
    x,y,z = MostCommonPostEDA(i,3)
    all_x.append(x)
    all_y.append(y)
    all_z.append(z)

for x,i in zip(all_x,stars):
  plot_grams(i,"Most Common Words",x)

for y,i in zip(all_y,stars):
  plot_grams(i,"Bi-grams",y)

for z,i in zip(all_z,stars):
  plot_grams(i,"Tri-grams",z)



all_words = ' '.join([text for text in reviews_clean['cleaned_text']])

from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')

positive_words = ' '.join(unique_pos_list)
negative_words = ' '.join(unique_neg_list)
pos_wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(positive_words)
plt.figure(figsize=(10, 7))
plt.imshow(pos_wordcloud, interpolation="bilinear")
plt.axis('off')
neg_wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(neg_wordcloud, interpolation="bilinear")
plt.axis('off')
#Splitting into train and test sets for the downsampled data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(reviews_ds['cleaned_text'],reviews_ds['rating'], test_size=0.2)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
y_test

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

cv = CountVectorizer(max_features = 50)
X_train_bow = cv.fit_transform(X_train).toarray()
X_test_bow = cv.fit_transform(X_test).toarray()

#Naive Bayes 
from sklearn.naive_bayes import MultinomialNB
classifier_mulnb = MultinomialNB()
classifier_mulnb.fit(X_train_bow, y_train)
y_test_pred_nulnb = classifier_mulnb.predict(X_test_bow)
mulnb_score = accuracy_score(y_test,y_test_pred_nulnb)
print("Multinomial Naive Bayes score", mulnb_score)

#Logistic Regression

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train_bow,  y_train)
y_test_predicted_logreg = log_reg.predict(X_test_bow)
score_test_logreg = accuracy_score(y_test,y_test_predicted_logreg)
print("Logististic Regression score", score_test_logreg)

#Random Forests
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train_bow,  y_train)
y_test_predicted_rf = rf.predict(X_test_bow)
score_test_rf = metrics.accuracy_score(y_test, y_test_predicted_rf)
print("Random Forest score",score_test_rf)
#Linear SVC
from sklearn.svm import LinearSVC
clf_svm = LinearSVC()
clf_svm.fit(X_train_bow,  y_train)
y_test_predicted_svm = clf_svm.predict(X_test_bow)
score_test_svm = metrics.accuracy_score(y_test, y_test_predicted_svm)
print("Linear SVM score",score_test_svm)
#Baseline plot
[mulnb_score,score_test_rf,score_test_logreg,score_test_svm]

#LogisticRegressionclassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import numpy as np

#Contribution 2: to help ML models to fit over different features 
num_fea = [10000,20000,50000,60000,80000,100000,200000,500000]
train_score_log_l = []
test_score_log_l = []
cnf_logreg = []


for i in num_fea:

  log_reg = LogisticRegression(C=1, penalty='l1', solver='liblinear')
  
  tfidf = TfidfVectorizer(
      input='content',
      encoding='utf-8',
      decode_error='strict',
      strip_accents=None,
      lowercase=True,
      preprocessor=None,
      tokenizer=None,
      stop_words=None,
      ngram_range=(1, 3),
      analyzer='word',
      max_df=1.0,
      min_df=1,
      max_features= i,
      vocabulary=None,
      binary=False,
  )

  pipeline_log_reg = Pipeline([
      ('tfidf', tfidf),
      ('logreg', log_reg)    ])



  pipeline_log_reg.fit(X_train, y_train)
  train_score_log = pipeline_log_reg.score(X_train, y_train)
  test_score_log = pipeline_log_reg.score(X_test, y_test)
  y_pred_pipeline_log_reg = pipeline_log_reg.predict(X_test)
  cnf = confusion_matrix(y_pred_pipeline_log_reg,y_test)
  print(classification_report(y_pred_pipeline_log_reg,y_test))
  cnf_logreg.append(cnf)
  train_score_log_l.append(train_score_log)
  test_score_log_l.append(test_score_log)
  
  print(i,"Train score",train_score_log)
  print(i,"Test Score ",test_score_log)
  
#MultinomialNBClassifier

train_score_mulnb_l = []
test_score_mulnb_l = []
cnf_mulnb = []

for i in num_fea:

  classifier_mulnb = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)

  tfidf = TfidfVectorizer(
      input='content',
      encoding='utf-8',
      decode_error='strict',
      strip_accents=None,
      lowercase=True,
      preprocessor=None,
      tokenizer= None,
      stop_words=None,
      ngram_range=(1, 3),
      analyzer='word',
      max_df=1.0,
      min_df=1,
      max_features= i,
      vocabulary=None,
      binary=False,
  )

  pipeline_mulnb = Pipeline([
      ('tfidf', tfidf),
      ('classifier_mulnb', classifier_mulnb)  ])



  pipeline_mulnb.fit(X_train, y_train)
  train_score_mulnb = pipeline_mulnb.score(X_train, y_train)
  test_score_mulnb = pipeline_mulnb.score(X_test, y_test)
  y_pred_pipeline_mulnb = pipeline_mulnb.predict(X_test)
  cnf = confusion_matrix(y_pred_pipeline_mulnb,y_test)
  print(classification_report(y_pred_pipeline_mulnb,y_test))
  cnf_mulnb.append(cnf)
  train_score_mulnb_l.append(train_score_mulnb)
  test_score_mulnb_l.append(test_score_mulnb)
  print(i,"Train score",train_score_mulnb)
  print(i,"Test Score ",test_score_mulnb)

#RandomForestClassifier
train_score_rf_l = []
test_score_rf_l = []
cnf_rf = []

for i in num_fea:

  rf = RandomForestClassifier(max_depth=2, random_state=0, n_estimators = 1000)

  tfidf = TfidfVectorizer(
      input='content',
      encoding='utf-8',
      decode_error='strict',
      strip_accents=None,
      lowercase=True,
      preprocessor=None,
      tokenizer= None,
      stop_words=None,
      ngram_range=(1, 3),
      analyzer='word',
      max_df=1.0,
      min_df=1,
      max_features= i,
      vocabulary=None,
      binary=False,
  )

  pipeline_rf = Pipeline([
      ('tfidf', tfidf),
      ('classifier_rf', rf)  ])

  pipeline_rf.fit(X_train, y_train)
  train_score_rf = pipeline_rf.score(X_train, y_train)
  test_score_rf = pipeline_rf.score(X_test, y_test)
  y_pred_pipeline_rf = pipeline_rf.predict(X_test)
  cnf = confusion_matrix(y_pred_pipeline_rf,y_test)
  print(classification_report(y_pred_pipeline_rf,y_test))
  cnf_rf.append(cnf)
  train_score_rf_l.append(train_score_rf)
  test_score_rf_l.append(test_score_rf)
  print(i,"Train score",train_score_rf)
  print(i,"Test Score ",test_score_rf)
  print(i,"Confusion Matrix: \n",cnf_rf)
  

#LinearSVC

train_score_svc_l = []
test_score_svc_l = []
cnf_svc = []


for i in num_fea:

  svc = LinearSVC(
      C=1.0,
      class_weight='balanced',
      dual=False,
      fit_intercept=True,
      intercept_scaling=1,
      loss='squared_hinge',
      max_iter=2000,
      multi_class='ovr',
      penalty='l2',
      random_state=0,
      tol=1e-05, 
      verbose=0
  )

  tfidf = TfidfVectorizer(
      input='content',
      encoding='utf-8',
      decode_error='strict',
      strip_accents=None,
      lowercase=True,
      preprocessor=None,
      tokenizer=None,
      stop_words=None,
      ngram_range=(1, 3),
      analyzer='word',
      max_df=1.0,
      min_df=1,
      max_features= i,
      vocabulary=None,
      binary=False,
  )

  pipeline_svc = Pipeline([
      ('tfidf', tfidf),
      ('svc', svc)    ])



  pipeline_svc.fit(X_train, y_train)
  train_score_svc = pipeline_svc.score(X_train, y_train)
  test_score_svc = pipeline_svc.score(X_test, y_test)
  y_pred_pipeline_svc = pipeline_svc.predict(X_test)
  cnf = confusion_matrix(y_pred_pipeline_svc,y_test)
  print(classification_report(y_pred_pipeline_svc,y_test))
  cnf_svc.append(cnf)
  train_score_svc_l.append(train_score_svc)
  test_score_svc_l.append(test_score_svc)
  print(i,"Train score",train_score_svc)
  print(i,"Test Score ",test_score_svc)
  
#Best accuracy comparision
#Values Taken from the above models and rounded to 2 decimals to plot
baseline_scores = [('Logistic Regression',11.2),('Naive Bayes',12.05),('Random Forests',12.19),('Linear SVC',12.9)]
best_scores_tfidf = [('Logistic Regression',33.76),('Naive Bayes',68.86),('Random Forests',18),('Linear SVC',83)]

plot_grams("Baseline (BOW) Comparision","Accuracy",baseline_scores)
plot_grams("TFIDF Pipelines Comparision","Accuracy",best_scores_tfidf)

import matplotlib.pyplot as plt 



fig, ax = plt.subplots(1, 1 ,figsize=(7,7))
plt.plot(num_fea, train_score_log_l, label='Logistic regression') 
plt.plot(num_fea, train_score_mulnb_l,label='Naive Bayes') 
plt.plot(num_fea, train_score_svc_l,label='Linear SVM Classifier') 
plt.plot(num_fea, train_score_rf_l,label='Random Forests') 


plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1)
 
plt.xlabel('# of features') 

plt.ylabel('Accuracy') 
 
plt.title('Test Accuracy for different model') 
  
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import itertools

"""

Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
"""



def plot_confusion_matrix(cm,target_names,title= 'Normalized Confusion matrix',
                          cmap=None,
                          normalize=True):
   

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "gray")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "gray")


    plt.tight_layout()
    
    plt.show()
star = ['1', '2','3','4','5','6','7','8','9','10']

for i in range(len(cnf_svc)): 
  
    if i == (len(cnf_svc)-1): 
            plot_confusion_matrix(cnf_svc[i],star)
pred = pipeline_svc.predict(["it was perfect but it can be better"])[0]
print(pred)
##############---Creating test set to predict final daset rating ----########################

from sklearn.utils import shuffle
train = shuffle(data)
X = train['comment'].values
y = train['rating'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
test_score_svc = pipeline_svc.score(X_test, y_test)
y_pred_pipeline_svc = pipeline_svc.predict(X_test)
cnf = confusion_matrix(y_pred_pipeline_svc,y_test)
print(classification_report(y_pred_pipeline_svc,y_test))
print("Final Test Dataset Accuracy: " +str(test_score_svc))
