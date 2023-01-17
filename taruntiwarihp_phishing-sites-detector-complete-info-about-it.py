# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install selenium
import pandas as pd # use for data manipulation and analysis

import numpy as np # use for multi-dimensional array and matrix



import seaborn as sns # use for high-level interface for drawing attractive and informative statistical graphics 

import matplotlib.pyplot as plt # It provides an object-oriented API for embedding plots into applications

%matplotlib inline 

# It sets the backend of matplotlib to the 'inline' backend:

import plotly.express as px

import time # calculate time 



from sklearn.linear_model import LogisticRegression # algo use to predict good or bad

from sklearn.naive_bayes import MultinomialNB # nlp algo use to predict good or bad



from sklearn.model_selection import train_test_split # spliting the data between feature and target

from sklearn.metrics import classification_report # gives whole report about metrics (e.g, recall,precision,f1_score,c_m)

from sklearn.metrics import confusion_matrix # gives info about actual and predict

from nltk.tokenize import RegexpTokenizer # regexp tokenizers use to split words from text  

from nltk.stem.snowball import SnowballStemmer # stemmes words

from sklearn.feature_extraction.text import CountVectorizer # create sparse matrix of words using regexptokenizes  

from sklearn.pipeline import make_pipeline # use for combining all prerocessors techniuqes and algos



from PIL import Image # getting images in notebook

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator# creates words colud



from bs4 import BeautifulSoup # use for scraping the data from website

from selenium import webdriver # use for automation chrome 

import networkx as nx # for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.



import pickle# use to dump model 



import warnings # ignores pink warnings 

warnings.filterwarnings('ignore')
# Loading the dataset

phish_data = pd.read_csv('/kaggle/input/phishing-site-urls/phishing_site_urls.csv')
phish_data.head()
phish_data.tail()
phish_data.info()
phish_data.isnull().sum() # there is no missing values
#create a dataframe of classes counts

label_counts = pd.DataFrame(phish_data.Label.value_counts())
#visualizing target_col

fig = px.bar(label_counts, x=label_counts.index, y=label_counts.Label)

fig.show()
tokenizer = RegexpTokenizer(r'[A-Za-z]+')#to getting alpha only
phish_data.URL[0]
# this will be pull letter which matches to expression

tokenizer.tokenize(phish_data.URL[0]) # using first row
print('Getting words tokenized ...')

t0= time.perf_counter()

phish_data['text_tokenized'] = phish_data.URL.map(lambda t: tokenizer.tokenize(t)) # doing with all rows

t1 = time.perf_counter() - t0

print('Time taken',t1 ,'sec')
phish_data.sample(5)
stemmer = SnowballStemmer("english") # choose a language
print('Getting words stemmed ...')

t0= time.perf_counter()

phish_data['text_stemmed'] = phish_data['text_tokenized'].map(lambda l: [stemmer.stem(word) for word in l])

t1= time.perf_counter() - t0

print('Time taken',t1 ,'sec')
phish_data.sample(5)
print('Getting joiningwords ...')

t0= time.perf_counter()

phish_data['text_sent'] = phish_data['text_stemmed'].map(lambda l: ' '.join(l))

t1= time.perf_counter() - t0

print('Time taken',t1 ,'sec')
phish_data.sample(5)
#sliceing classes

bad_sites = phish_data[phish_data.Label == 'bad']

good_sites = phish_data[phish_data.Label == 'good']
bad_sites.head()
good_sites.head()
def plot_wordcloud(text, mask=None, max_words=400, max_font_size=120, figure_size=(24.0,16.0), 

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    more_stopwords = {'com','http'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color='white',

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    mask = mask)

    wordcloud.generate(text)

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'green', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  

d = '../input/masks/masks-wordclouds/'
data = good_sites.text_sent

data.reset_index(drop=True, inplace=True)
common_text = str(data)

common_mask = np.array(Image.open(d+'star.png'))

plot_wordcloud(common_text, common_mask, max_words=400, max_font_size=120, 

               title = 'Most common words use in good urls', title_size=15)
data = bad_sites.text_sent

data.reset_index(drop=True, inplace=True)
common_text = str(data)

common_mask = np.array(Image.open(d+'comment.png'))

plot_wordcloud(common_text, common_mask, max_words=400, max_font_size=120, 

               title = 'Most common words use in bad urls', title_size=15)
#create cv object

cv = CountVectorizer()
help(CountVectorizer())
feature = cv.fit_transform(phish_data.text_sent) #transform all text which we tokenize and stemed
feature[:5].toarray() # convert sparse matrix into array to print transformed features
trainX, testX, trainY, testY = train_test_split(feature, phish_data.Label)
# create lr object

lr = LogisticRegression()
lr.fit(trainX,trainY)
lr.score(testX,testY)
Scores_ml = {}

Scores_ml['Logistic Regression'] = np.round(lr.score(testX,testY),2)
print('Training Accuracy :',lr.score(trainX,trainY))

print('Testing Accuracy :',lr.score(testX,testY))

con_mat = pd.DataFrame(confusion_matrix(lr.predict(testX), testY),

            columns = ['Predicted:Bad', 'Predicted:Good'],

            index = ['Actual:Bad', 'Actual:Good'])





print('\nCLASSIFICATION REPORT\n')

print(classification_report(lr.predict(testX), testY,

                            target_names =['Bad','Good']))



print('\nCONFUSION MATRIX')

plt.figure(figsize= (6,4))

sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")
# create mnb object

mnb = MultinomialNB()
mnb.fit(trainX,trainY)
mnb.score(testX,testY)
Scores_ml['MultinomialNB'] = np.round(mnb.score(testX,testY),2)
print('Training Accuracy :',mnb.score(trainX,trainY))

print('Testing Accuracy :',mnb.score(testX,testY))

con_mat = pd.DataFrame(confusion_matrix(mnb.predict(testX), testY),

            columns = ['Predicted:Bad', 'Predicted:Good'],

            index = ['Actual:Bad', 'Actual:Good'])





print('\nCLASSIFICATION REPORT\n')

print(classification_report(mnb.predict(testX), testY,

                            target_names =['Bad','Good']))



print('\nCONFUSION MATRIX')

plt.figure(figsize= (6,4))

sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")
acc = pd.DataFrame.from_dict(Scores_ml,orient = 'index',columns=['Accuracy'])

sns.set_style('darkgrid')

sns.barplot(acc.index,acc.Accuracy)
pipeline_ls = make_pipeline(CountVectorizer(tokenizer = RegexpTokenizer(r'[A-Za-z]+').tokenize,stop_words='english'), LogisticRegression())

##(r'\b(?:http|ftp)s?://\S*\w|\w+|[^\w\s]+') ([a-zA-Z]+)([0-9]+)  -- these tolenizers giving me low accuray 
trainX, testX, trainY, testY = train_test_split(phish_data.URL, phish_data.Label)
pipeline_ls.fit(trainX,trainY)
pipeline_ls.score(testX,testY) 
print('Training Accuracy :',pipeline_ls.score(trainX,trainY))

print('Testing Accuracy :',pipeline_ls.score(testX,testY))

con_mat = pd.DataFrame(confusion_matrix(pipeline_ls.predict(testX), testY),

            columns = ['Predicted:Bad', 'Predicted:Good'],

            index = ['Actual:Bad', 'Actual:Good'])





print('\nCLASSIFICATION REPORT\n')

print(classification_report(pipeline_ls.predict(testX), testY,

                            target_names =['Bad','Good']))



print('\nCONFUSION MATRIX')

plt.figure(figsize= (6,4))

sns.heatmap(con_mat, annot = True,fmt='d',cmap="YlGnBu")
pickle.dump(pipeline_ls,open('phishing.pkl','wb'))
loaded_model = pickle.load(open('phishing.pkl', 'rb'))

result = loaded_model.score(testX,testY)

print(result)
predict_bad = ['yeniik.com.tr/wp-admin/js/login.alibaba.com/login.jsp.php','fazan-pacir.rs/temp/libraries/ipad','tubemoviez.exe','svision-online.de/mgfi/administrator/components/com_babackup/classes/fx29id1.txt']

predict_good = ['youtube.com/','youtube.com/watch?v=qI0TQJI3vdU','retailhellunderground.com/','restorevisioncenters.com/html/technology.html']

loaded_model = pickle.load(open('phishing.pkl', 'rb'))

#predict_bad = vectorizers.transform(predict_bad)

# predict_good = vectorizer.transform(predict_good)

result = loaded_model.predict(predict_bad)

result2 = loaded_model.predict(predict_good)

print(result)

print("*"*30)

print(result2)