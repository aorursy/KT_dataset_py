# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # for linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re #regular expressions
import matplotlib.pyplot as plt  #graphes
import seaborn as sns #graphes
import string 
import nltk #traitement de language humain
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline

#On importe les bibliotheques necessaires
#On lit et copy les data sets pour le pre-processing

#train = pd.read_csv('../input/twitter-sentiment-analysis-hatred-speech/train.csv')
#train_original=train.copy()

trainb = pd.read_csv('../input/sentiment-analys-dataset/Sentiment Analysis Dataset.csv',error_bad_lines=False)
trainb_original=trainb.copy()

test = pd.read_csv('../input/twitter-sentiment-analysis-hatred-speech/test.csv')
test_original=test.copy()

trainb.columns = ['id','label','todelete','tweet']
trainb = trainb.drop("todelete", axis=1)
trainb['label'].replace({1: 0, 0: 1}, inplace=True)

train=trainb[:31962]

trainb.head()
combine = train.append(test,ignore_index=True,sort=True)
#on peut visualiser le nouveau fichier avec ces deux commandes
#combine.tail()
combine.head()
#cette fonction est pour enlever le @user des tweets
def remove_pattern(text,pattern):
    
    # re.findall() trouve tout le mots pattern dans un texte et les mets dans une liste r
    r = re.findall(pattern,text)
    
    # re.sub() enleve la pattern des phrases de la data set
    for i in r:
        text = re.sub(i,"",text)
    
    return text
combine['Tidy_Tweets'] = np.vectorize(remove_pattern)(combine['tweet'], "@[\w]*")
#on ajoute une colonne avec les tweets pre-processed , dans cette exemple ils manquent le @user
combine.head()
#Maintenant on enleve la ponctuation , les chiffres et les characteres speciaux 
combine['Tidy_Tweets'] = combine['Tidy_Tweets'].str.replace("[^a-zA-Z#]", " ")

#combine.head(10)
#On enleve les petits mots insignifiant comme 'the' 'oh' 'and'  etc ...
combine['Tidy_Tweets'] = combine['Tidy_Tweets'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

combine.head(10)
#on creer une liste a partir des tweets ordonnés

tokenized_tweet = combine['Tidy_Tweets'].apply(lambda x: x.split())

#tokenized_tweet.head()
#on regroupe les mots de la meme famille 

from nltk import PorterStemmer

ps = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [ps.stem(i) for i in x])

tokenized_tweet.head()
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

combine['Tidy_Tweets'] = tokenized_tweet
combine.head()
# fonction pour extraire les hashtags : 

def Hashtags_Extract(x):
    hashtags=[]
    
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r'#(\w+)',i)
        hashtags.append(ht)
    
    return hashtags

#on extrait les hashtags positifs
ht_positive = Hashtags_Extract(combine['tweet'][combine['label']==0])

ht_positive[:10]
#on extrait les hashtags negatifs 

ht_negative = Hashtags_Extract(combine['tweet'][combine['label']==1])

ht_negative[:10]
#on combine la list en une list 1xn (1dimension) 
ht_negative_unnest = sum(ht_negative,[])
ht_positive_unnest = sum(ht_positive,[])


#une representation des hashtags les plus utilisés 
word_freq_positive = nltk.FreqDist(ht_positive_unnest)

word_freq_positive
#creating d'un dataframe qui compte le redondance de chaque hashtag positif

df_positive = pd.DataFrame({'Hashtags':list(word_freq_positive.keys()),'Count':list(word_freq_positive.values())})

df_positive.head(10)


#affichage de cette dataframe en forme de graphe 

df_positive_plot = df_positive.nlargest(20,columns='Count')

sns.barplot(data=df_positive_plot,y='Hashtags',x='Count')
sns.despine()
#meme chose pour les negatifs
word_freq_negative = nltk.FreqDist(ht_negative_unnest)

word_freq_negative

df_negative = pd.DataFrame({'Hashtags':list(word_freq_negative.keys()),'Count':list(word_freq_negative.values())})

df_negative.head(10)

df_negative_plot = df_negative.nlargest(20,columns='Count')

sns.barplot(data=df_negative_plot,y='Hashtags',x='Count')
sns.despine()
#Tfidf

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(max_df=0.90, min_df=2,max_features=1000,stop_words='english')

tfidf_matrix=tfidf.fit_transform(combine['Tidy_Tweets'])

df_tfidf = pd.DataFrame(tfidf_matrix.todense())

df_tfidf
train_tfidf_matrix = tfidf_matrix[:31962]

train_tfidf_matrix.todense()
from sklearn.model_selection import train_test_split
x_train_tfidf, x_valid_tfidf, y_train_tfidf, y_valid_tfidf = train_test_split(train_tfidf_matrix,train['label'],test_size=0.3,random_state=17)

#on divise la data set sous des matrices d'entrainement et de test randomly
from sklearn.linear_model import LogisticRegression

Log_Reg = LogisticRegression(random_state=0,solver='lbfgs')

Log_Reg.fit(x_train_tfidf,y_train_tfidf)

prediction_tfidf = Log_Reg.predict_proba(x_valid_tfidf)

prediction_tfidf

#log_reg prends ces matrices pour pouvoir classifier leur positivité selon la probabilité
#tf
test_tfidf = tfidf_matrix[31962:]
test_pred = Log_Reg.predict_proba(test_tfidf)
#on predit le reste de la matrice non labeled 
test_pred_int = test_pred[:,1] >= 0.3
#kanchdo les valeurs li kber mn 0.3 f probabilités bach kan7sbohom homa proba dominante
test_pred_int = test_pred_int.astype(np.int)

test['label'] = test_pred_int

submission = test[['id','label','tweet']]


submission.to_csv('resulttf.csv', index=False)

restf = pd.read_csv('resulttf.csv')
restf
restf['Tidy_Tweets'] = np.vectorize(remove_pattern)(restf['tweet'], "@[\w]*")
restf.head()
#Data vizualization wordcloud 
from wordcloud import WordCloud,ImageColorGenerator
from PIL import Image
import urllib
import requests



all_words_positive = ' '.join(text for text in restf['Tidy_Tweets'][restf['label']==0])


# combining the image with the dataset
Mask = np.array(Image.open(requests.get('http://clipart-library.com/image_gallery2/Twitter-PNG-Image.png', stream=True).raw))

# We use the ImageColorGenerator library from Wordcloud 
# Here we take the color of the image and impose it over our wordcloud
image_colors = ImageColorGenerator(Mask)

# Now we use the WordCloud function from the wordcloud library 
wc = WordCloud(background_color='white', height=1500, width=4000,mask=Mask).generate(all_words_positive)

# Size of the image generated 
plt.figure(figsize=(10,20))

# Here we recolor the words from the dataset to the image's color
# recolor just recolors the default colors to the image's blue color
# interpolation is used to smooth the image generated 
plt.imshow(wc.recolor(color_func=image_colors),interpolation="hamming")

plt.axis('off')
plt.show()
all_words_negative = ' '.join(text for text in restf['Tidy_Tweets'][restf['label']==1])


# combining the image with the dataset
Mask = np.array(Image.open('../input/rdtweet/redtweet.png'))

# We use the ImageColorGenerator library from Wordcloud 
# Here we take the color of the image and impose it over our wordcloud
image_colors = ImageColorGenerator(Mask)

# Now we use the WordCloud function from the wordcloud library 
wc = WordCloud(background_color='white', height=1500, width=4000,mask=Mask).generate(all_words_negative)

# Size of the image generated 
plt.figure(figsize=(10,20))


# Here we recolor the words from the dataset to the image's color
# recolor just recolors the default colors to the image's blue color
# interpolation is used to smooth the image generated 
plt.imshow(wc.recolor(color_func=image_colors),interpolation="hamming")

plt.axis('off')
plt.show()
#combine Res with train 
total = train.append(restf,ignore_index=True,sort=True)
#on peut visualiser le nouveau fichier avec ces deux commandes
#combine.tail()
total.head()
#count neg
countneg = len(total['tweet'][total['label']==1])
print(countneg)
countpos = len(total['tweet'][total['label']==0])
print(countpos)


#Pie charts 
# Data to plot
labels = 'Positive', 'Negative'
sizes = [countpos, countneg]
colors = ['aqua', 'lightsalmon']


# Plot
plt.pie(sizes, labels=labels, colors=colors,autopct='%1.1f%%', shadow=True,radius=3, startangle=100)


plt.axis('off')
plt.show()

from sklearn.metrics import f1_score

# Si la prediction est sup ou egale à 0.3 on lui donne 1 sinon 0
#  0 positive  1  negative 
prediction_int = prediction_tfidf[:,1]>=0.3

prediction_int = prediction_int.astype(np.int)
prediction_int

# calcule f1 score
log_tfidf = f1_score(y_valid_tfidf, prediction_int)

log_tfidf