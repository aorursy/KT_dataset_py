# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sb
import statsmodels.api as sm
from sklearn import linear_model
#load in data
df = pd.read_csv('../input/winemag-data-130k-v2.csv')
# see data contents
df.head()
# Start EDA process
# clean data (drop NaN in price, variety and country)
df = df.dropna(subset=['price','country','variety'])
df.count()
# Check how many variety of wine are
df['variety'].value_counts()
# Since so many variety of wine available in this dataset,
# Therefore, I reduce the data by filtering more than 1000 reviews 
# The reason why I reduce the dataset is that the minor wine type will not be avaliable in Australian local liquor shop
df2=df.groupby('variety').filter(lambda x: len(x) >1000) # New dataset
#df2 = pd.DataFrame({col:vals['points'] for col,vals in variety.groupby('variety')})
df2
#print df2['variety'].value_counts()
df2['variety'].value_counts()
# To check my dataset looks like
# plot data (price and points)
x = df2['points']
y = df2['price']

plt.plot(x,y,'bo')
#Above figure indicates that three over priced wines
#Therefore, I remove those
df3 = df2.drop(df2[df2.price > 1000].index)
x = df3['points']
y = df3['price']
plt.plot(x,y,'bo');
#Showing the example of the result of the initial data processing.

import matplotlib.pyplot as plt
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
fig, ax = plt.subplots(figsize = (20,10))
chart = sb.boxplot(x='variety',y='points', data=df3, ax = ax)
plt.title('Distribution of Point Scores by major 24 wine type',size=14) 
plt.xlabel('Wine type') 
plt.xticks(rotation = 90)

# For this process, Ordinary Least Square method (OLS) used. 
from scipy.stats import pearsonr #Pearson correlation coefficient calculator
import statsmodels.api as sm
# First plot result
sb.lmplot(y = 'price', x='points', data=df3)
# Pearson correltion between price and point
pearsonr(df3.price, df3.points)
# Showing the table for the OLS calculation results
sm.OLS(df3.points, df3.price).fit().summary()
df_sorted = df3.sort_values(by='points', ascending=True)  # sort by points

num_of_wines = df_sorted.shape[0]  # number of wines
worse = df_sorted.head(int(0.3*num_of_wines))  # 30 % of worst wines listed
better = df_sorted.tail(int(0.3*num_of_wines))  # 30 % of best wines listed
# Show the data that used in this process

plt.hist(df3['points'], color='grey', label='All')
plt.hist(worse['points'], color='blue', label='Worse')
plt.hist(better['points'], color='red', label='Better')
plt.legend()
plt.show()
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names
 
def word_feats(words):
    return dict([(word, True) for word in words])
# just select few words to express sweet and dry
sweet_vocab = [ 'sweet', 'Fruit', 'port', 'muscat', 'semi sweet', 'medium sweet', 'very sweet', 'sauterves',':)' ]
dry_vocab = [ 'dry', 'off dry', 'medium dry','secco', 'extra', ':(' ]
unknown_vocab = [ 'the','are','were', 'was','is','did','words','not' ]
 
sweet_features = [(word_feats(sweet), 'sweet') for sweet in sweet_vocab]
dry_features = [(word_feats(dry), 'dry') for dry in dry_vocab]
unknown_features = [(word_feats(unk), 'unk') for unk in unknown_vocab]
 
train_set = dry_features + sweet_features + unknown_features
 
classifier = NaiveBayesClassifier.train(train_set) 

# define function for classifying sweet and dry
def dry_sweet(sentence):
    sentence= sentence.lower()
    words = sentence.split(' ')
    dry=0
    sweet=0
    #words
    for word in words:
        classResult = classifier.classify( word_feats(word))
        if classResult == 'dry':
            dry = dry + 1
        if classResult == 'sweet':
            sweet = sweet + 1
    s = str(float(sweet)/len(words))
    d = str(float(dry)/len(words))
    return(s, d)
# Converting description into the point (numerical numbers)
worse['s & d']= worse['description'].apply(lambda x: dry_sweet(x))
better['s & d']= better['description'].apply(lambda x: dry_sweet(x))
type(worse['s & d'][1])
# Add new columuns into dataframes
worse[['sweet', 'dry']] = worse['s & d'].apply(pd.Series)
better[['sweet', 'dry']] = better['s & d'].apply(pd.Series)
type(worse['s & d'][1])
# Add new columuns into dataframes
worse[['sweet', 'dry']] = worse['s & d'].apply(pd.Series)
better[['sweet', 'dry']] = better['s & d'].apply(pd.Series)
better.info()
# convert object into numeric
worse['sweet']=worse['sweet'].astype('float64')
better['sweet']=better['sweet'].astype('float64')
'Worse =', worse.sweet.mean(), 'Better =', better.sweet.mean()
#Load requisite packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
import nltk
from nltk.stem.porter import PorterStemmer
# Start EDA process
# clean data (drop NaN in price, variety and country)
df = df.dropna(subset=['price','country','variety'])
df.count()
# Check how many country produce wine?
df['country'].value_counts()
# This time only US wine will be used,
df_US = df[(df.country == 'US') ]
df_US.info()
# extract all words in descriptions for US wine
us_descs = df_US['description'].values
us_descs = " ".join(us_descs)

def tokenize(text):
    text = text.lower()
    text = re.sub('[' + string.punctuation + '0-9\\r\\t\\n]', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if len(w) > 2]
    tokens = [w for w in tokens if not w in ENGLISH_STOP_WORDS]
    return tokens 

def stemwords(words):
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words] # stem words 
    return words
# create counter of best words 
us_words = stemwords(tokenize(us_descs))
us_ctr = Counter(us_words)
# cloud for best words
wordcloud = WordCloud()
wordcloud.fit_words(us_ctr)

fig=plt.figure(figsize=(10, 6))   # Prepare a plot 5x3 inches
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# Again clean data
from sklearn.model_selection import train_test_split
data = df3.drop(['Unnamed: 0','country','designation','points','province','region_1','region_2','variety','winery'], axis = 1)
y = df3.variety

data_train, data_test, y_train, y_test = train_test_split(data, y, random_state=1)
print(data_train.shape, data_test.shape, y_train.shape, y_test.shape)
# Make list for thr wine-type

wine =df3.variety.unique().tolist()
wine.sort()
wine[:24]
# Make lower case of every wine-type
output = set()
for x in df3.variety:
    x = x.lower()
    x = x.split()
    for y in x:
        output.add(y)

variety_list =sorted(output)
variety_list[:24]
extras = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', 'cab',"%"]
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
stop.update(variety_list)
stop.update(extras)
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer 
vect = CountVectorizer(stop_words = stop)
data_train_dtm = vect.fit_transform(data_train.description)
price = data_train.price.values[:,None]
data_train_dtm = hstack((data_train_dtm, price))
data_train_dtm
data_test_dtm = vect.transform(data_test.description)
price_test = data_test.price.values[:,None]
data_test_dtm = hstack((data_test_dtm, price_test))
data_test_dtm
# Create predicting model using Logistic regression
# Step 1 creating regression model using logistic regression
from sklearn.linear_model import LogisticRegression
models = {}
for z in wine:
    model = LogisticRegression(random_state=0)
    y = y_train == z
    model.fit(data_train_dtm, y)
    models[z] = model

testing_model = pd.DataFrame(columns = wine)
#Show the test model
testing_model
# Create predicting model using Logistic regression
# Step 2 making predicted model
for variety in wine:
    testing_model[variety] = models[variety].predict_proba(data_test_dtm)[:,1]
    
predicted_wine = testing_model.idxmax(axis=1)

comparison = pd.DataFrame({'actual':y_test.values, 'predicted':predicted_wine.values})   

from sklearn.metrics import accuracy_score
#Show the results (accuracy of the predicted model)
'Accuracy Score:',accuracy_score(comparison.actual, comparison.predicted)*100,"%"
# Show results
comparison.head(24)
