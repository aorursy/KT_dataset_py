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
import re

from nltk.corpus import stopwords

import nltk

from nltk.tokenize import RegexpTokenizer

from bs4 import BeautifulSoup

from string import punctuation

import string

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.ensemble import voting,StackingClassifier,AdaBoostClassifier,RandomForestClassifier
df = pd.read_csv('../input/clickbait-dataset/clickbait_data.csv')

df
nltk.download('stopwords')
stop = set(string.punctuation)
tokenizer = RegexpTokenizer(r'\w+')
def remhtml(text):

  soup = BeautifulSoup(text,"html.parser")

  return soup.get_text()



def rem_bw_sq_bracks(text):

  return re.sub('\[[^]]*\]','',text)



def remhttplink(text):

  return re.sub(r'http\S+','',text)



def remAT(text):

  return re.sub(r'@','',text)



def remstopwords(text):

  word_list = tokenizer.tokenize(text)

  clean_list = [w for w in word_list if w not in stop]

  clean_text = ' '.join(clean_list)

  return clean_text



def clean_sent(text):

  text = text.lower()

  text = remhtml(text)

  text = rem_bw_sq_bracks(text)

  text = remhttplink(text)

  text = remAT(text)

  text = remstopwords(text)

  return text



df['headline'] = df['headline'].apply(clean_sent)
df
from wordcloud import WordCloud,STOPWORDS
plt.figure(figsize = (20,20)) # Text that is ClickBait

wc = WordCloud(max_words = 3000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(df[df.clickbait == 1].headline))

plt.imshow(wc , interpolation = 'bilinear')

plt.grid(False)
plt.figure(figsize = (20,20)) # Text that is not ClickBait

wc = WordCloud(max_words = 3000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(df[df.clickbait == 0].headline))

plt.imshow(wc , interpolation = 'bilinear')

plt.grid(False)
sns.countplot(df.clickbait)
import tensorflow_hub as hub
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
x_train,x_test,y_train,y_test = train_test_split(df.headline.values,df.clickbait.values,random_state = 0,stratify=df.clickbait.values,test_size=0.20)
embed_train = embed(x_train)

x = embed_train.numpy()
embed_test = embed(x_test)

xt = embed_test.numpy()
y_train=y_train.astype('int')

y_test=y_test.astype('int')
from time import time
def sigmoid(x):

  return (1/(1+np.exp(-1*x)))
dt = DecisionTreeClassifier(random_state=2)

def pso(X, Y, popu_size, maxItr=100):

  particles = []

  for i in range(popu_size):

      particles.append(np.random.choice([0, 1], size=512))

  particles = np.array(particles,dtype=float)

  velocities = np.random.rand(popu_size,512)

  itr = 0

  pbest = np.zeros((popu_size,512))

  pbest_score = np.ones((popu_size,),dtype=float)*0

  gbest_score = 0

  gbest = np.zeros((512,))

  while(itr < maxItr):

    start = time()

    for i in range(popu_size):

      X_togive = np.multiply(X, particles[i])

      dt.fit(X_togive, Y)

      Xt = np.multiply(xt, particles[i])

      score = dt.score(Xt,y_test)

      if score > pbest_score[i]:

        pbest_score[i] = score

        pbest[i] = particles[i]

      if score > gbest_score:

        gbest_score = score

        gbest = pbest[i]

    for i in range(popu_size):

      velocities[i] = 0.4*velocities[i] + 2*np.random.rand()*(pbest[i]-particles[i]) +2*np.random.rand()*(gbest-particles[i])

      for kk in range(512):

        if velocities[i][kk]>4.0:

          velocities[i][kk] =  4.0

        elif velocities[i][kk]<-4.0:

          velocities[i][kk] = -4.0

      velo_sig = sigmoid(velocities[i])

      for kk in range(512):

        particles[i][kk] = (np.random.randn()<velo_sig[kk])*1

    end = time()

    print("Epochs:",itr+1,"Time Taken:",(end-start),"secs","Best Score:",gbest_score)

    itr+=1

  return gbest
dt.fit(x,y_train)

dt.score(xt,y_test)
new_x = pso(x,y_train,5,100)
X_togive = np.multiply(x,new_x)

dt.fit(X_togive, y_train)

dt.score(xt,y_test)
rf = RandomForestClassifier(random_state=2)

rf.fit(X_togive,y_train)

rf.score(xt,y_test)