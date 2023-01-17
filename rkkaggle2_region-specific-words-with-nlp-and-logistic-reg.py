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
#Importing the required modules for EDA and ETL

import os

import glob #glob is a tool to index and list multiple files for convenient reading

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.style as style



style.use('Solarize_Light2')

gl = glob.glob('../input/twitter-dateset-collected-during-covid19-pandemic/data_twitter/*.csv')

print('There are {} files total'.format(len(gl)))
#Reading in and concatenating tweet data into a single dataframe

li = [pd.read_csv(t) for t in gl]

li = pd.concat(li)

li = li.iloc[:,1:4]

li.columns = ['time','content','location']



li.info()
from nltk.tokenize import TweetTokenizer 

from nltk.corpus import stopwords



twe = TweetTokenizer()

#Filtering the data to two large scale indian Cities Chennai and Bangalore

che = li[li['location'].isin(["Chennai","Bangalore"])]

che.time =  pd.to_datetime(che.time)



#Creating a new list of tokenized tweets for cleaning

chetoken = [twe.tokenize(t) for t in che.content]

stop = stopwords.words('english') #Removing Stop Words

chetoken = [[t for t in g if t not in stop] for g in chetoken]

chetoken = [[t for t in g if t.isalpha()] for g in chetoken] #Removing non alpha numeric characters

chetoken = [' '.join(t) for t in chetoken]





chetoken[:5] #Previewing the first five cleaned tweets
#Plotting the number of daily tweets over observation period by city

vis1 = che[['time','location','content']].groupby(['time','location']).agg('count')

vis1 = vis1.reset_index()

plt.figure(figsize=(25,15))



ax1 = sns.relplot(data=vis1,x='time',y='content',hue='location',kind='line',style='location')

ax1.set_xticklabels( rotation=40, ha="right")

ax1.fig.suptitle('Number of Daity Tweets by City')

ax1.set_xlabels("Date")

ax1.set_ylabels('Number of COVID-19 tweets')





#Importing the packages necessary to build a wordcloud

from PIL import Image

from wordcloud import WordCloud





#Consilidating our corpus into a single text string for processing

singletext = ''.join(map(str, chetoken))



#Creating the mask(wordcloud shape) in the form of a COVID protien spike

covidmask =  np.array(Image.open('../input/covidimg/covidspike.jpg'))



#Putting the wordcloud together

wordcloud1 = WordCloud(width = 700,height=700,colormap='GnBu',

                       mask=covidmask,max_words=400,background_color = 'white')

wordcloud1.generate(singletext)

plt.figure(figsize=(35,23))

plt.imshow(wordcloud1, interpolation="bilinear")

plt.axis("off")

plt.margins(x=0, y=0)

plt.show()
#Building a binomial classifier with the following packages for Logistic Regression

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer



vectorizer = CountVectorizer(strip_accents='ascii',lowercase=True,stop_words='english')

X_train, X_test, y_train, y_test = train_test_split(chetoken, che.location, test_size=0.25)

trvect = vectorizer.fit_transform(X_train)

tevect = vectorizer.transform(X_test)



trvect
lr = LogisticRegression(C=10,max_iter=3000)

lr.fit(trvect,y_train)

y_pred = lr.predict(tevect)



print('The Accuracy Score is : {0:2.2%}'.format(accuracy_score(y_test,y_pred)))

confusion_matrix(y_test,y_pred)
classification_report(y_test,y_pred,output_dict=True)
#Building a dataframe of top words associated by city of origin

features = dict(zip(vectorizer.get_feature_names(),np.exp(lr.coef_).reshape(-1,1)))

features = pd.DataFrame(features).T

features.columns = ['Odds']

features = features.sort_values('Odds')



#Streamlining the features to the top 25 region specific words per city

bow = vectorizer.vocabulary_

topwords = pd.concat([features.head(25), features.tail(25)])

topwords['frequency'] = [trvect[bow.get(t)].sum() for t in topwords.index]

topwords['city'] = np.where(topwords.Odds < 1,'Bangalore','Chennai')



#Examining Last 10 words of DataFrame

topwords.tail(10)
#Similarly examining First 10 words of DataFrame

topwords.head(10)
topwords['logodds'] = np.log(topwords.Odds)



plt.figure(figsize=(20,10))

ax2 = sns.scatterplot(x='frequency',y = 'logodds',hue='city',data=topwords, palette="Set2")

ax2.set_title('Most Region specific words used by Chennaites and Bangaloreans')

for line in range(0,topwords.shape[0]):

     ax2.text(topwords.frequency[line]+0.2, topwords.logodds[line], topwords.index[line],

              horizontalalignment='left', size='small', color='black')