import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data_set=pd.read_csv("../input/fake-and-real-news-dataset/True.csv")
data_set.head()
dataset_complete=pd.DataFrame(data_set)
import numpy as np
dataset_complete.insert(loc=0,value=np.ones(len(data_set)),column="Label" )
dataset_complete.head()
data_set_fake=pd.read_csv("../input/fake-and-real-news-dataset/Fake.csv")
dataset_completed=dataset_complete.append(data_set_fake,ignore_index=True)
dataset_completed.Label.fillna(0,inplace=True)
dataset_completed.shape
#checking the null value
data_set.isnull().sum()
#shuffling the dataset rows so that we get mix of postive and negative not sequentially
from sklearn.utils import shuffle
dataset_completed=shuffle(dataset_completed)
dataset_completed.head(10)
labels=dataset_completed.iloc[:,0]
features=dataset_completed.iloc[:,1:]
#reseting the index to get the index in order
labels=labels.reset_index()
labels=labels.Label
features.reset_index(inplace=True)
features=features.iloc[:,1:]
labels.head()
features.head()
import nltk
nltk.download()
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from nltk.stem import PorterStemmer
lemmitizer=WordNetLemmatizer()
stemmer=PorterStemmer()
corpus=[]
for i in range(len(features)):
  news_text=features.text[i]
  news_text=re.sub('[^a-zA-Z]'," ",news_text) #taking only the word made of A-Z and a-z
  news_text=news_text.lower()#lowering the sentences
  news_text=news_text.split()#spiltings the sentences into words
  word_wt_sw=[]
  for word in news_text:
    if word not in set(stopwords.words('english')):
      word_wt_sw.append(lemmitizer.lemmatize(word)) #lemmatize the word
  news_text=" ".join(word_wt_sw)
  corpus.append(news_text)
#applying CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=7000)
X=cv.fit_transform(corpus).toarray()
X.shape
labels=np.array(labels)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,labels,test_size=0.2,random_state=0)
from sklearn.naive_bayes import MultinomialNB
nv=MultinomialNB(alpha=0.1)
nv.fit(X_train,y_train)
y_pred=nv.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
conf=confusion_matrix(y_pred,y_test)
acc=accuracy_score(y_pred,y_test)
print("accuarcy is ", acc)
conf
string=""
i=0
for sent in (features.text):
  for word in sent.split():
    string=string+" "+word
  i=i+1
  if(i>=1200):
    break
#wordcloud = WordCloud().generate(string)

from wordcloud import WordCloud
wordcloud = WordCloud(width = 1200, height = 800, 
                background_color ='white', 
                stopwords = stopwords.words("english"), 
                min_font_size = 10).generate(string)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
