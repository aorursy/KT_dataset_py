import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import pandas_profiling

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import roc_curve, auc

from sklearn.neural_network import MLPClassifier

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

import sklearn.metrics as mt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv')
df.head()
df.shape
df.info()
df.describe()
df.isnull().sum()
df.groupby(['Rating', 'Recommended IND'])['Recommended IND'].count()
df1=df.drop(['Unnamed: 0','Division Name','Department Name',],axis=1)

df1.columns
df1.head()
df1[df1['Review Text'].isnull()]
df1 = df1[~df1['Review Text'].isnull()]
df1.shape
df1.head(77)
import plotly.offline as py

import plotly.graph_objs as go

x=df1['Recommended IND'].value_counts()

colors = ['#FEBFB3', '#E1396C']



trace=go.Pie(labels=x.index,values=x,textinfo="value",

            marker=dict(colors=colors, 

                           line=dict(color='#000000', width=2)))

layout=go.Layout(title="Cloths are Recommended or not",width=500,height=500)

fig=go.Figure(data=[trace],layout=layout)

py.iplot(fig, filename='pie_chart_subplots')
import plotly.express as px

fig = px.histogram(df1, x=df1['Rating'], nbins=10)

fig.show()


fig = px.histogram(df1, x = df1['Class Name'])

fig.show()
df1['review_len'] = df1['Review Text'].astype(str).apply(len)
px.histogram(df1, x = 'review_len')
df1['token_count'] = df1['Review Text'].apply(lambda x: len(str(x).split()))
px.histogram(df1, x = 'token_count')
!pip install TextBlob

from textblob import *
df1['polarity'] = df1['Review Text'].map(lambda text: TextBlob(text).sentiment.polarity)

df1['polarity']
fig = px.histogram(df1, x = df1['polarity'])

fig.show()
pop = df1.loc[df1.polarity == 1,['Review Text']].sample(3).values

for i in pop:

    print(i[0])
pop = df1.loc[df1.polarity == 0.5,['Review Text']].sample(3).values

for i in pop:

    print(i[0])
pop = df1.loc[df1.polarity < 0,['Review Text']].sample(3).values

for i in pop:

    print(i[0])
negative = (len(df1.loc[df1.polarity <0,['Review Text']].values)/len(df1))*100

positive = (len(df1.loc[df1.polarity >0.5,['Review Text']].values)/len(df1))*100

neutral  = len(df1.loc[df1.polarity >0 ,['Review Text']].values) - len(df1.loc[df1.polarity >0.5 ,['Review Text']].values)

neutral = neutral/len(df1)*100 

plt.figure(figsize =(10, 7)) 

plt.pie([positive,negative,neutral], labels = ['Positive','Negative','Neutral'],colors = [ 'blue','#E1396C','#FEBFB3'])
plt.figure(figsize=(8,8))

Age = df1['Age']

fx=sns.boxplot(x='Rating',y='Age',data=df1)

plt.title("Distribution of age with respect to rating")

plt.xlabel("Rating")

plt.ylabel("Age")
y = df1['Recommended IND']

X = df1.drop(columns = 'Recommended IND')
plt.figure(figsize=(14,7))

sns.heatmap(df1.corr(method='kendall'), annot=True )
set1 =set()

cor = df1.corr()

for i in cor.columns:

    for j in cor.columns:

        if cor[i][j]>0.8 and i!=j:

            set1.add(i)

print(set1)
X = X.drop(labels = ['token_count'],axis = 1)
X.corr()
import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords

stop=stopwords.words('english')
from wordcloud import WordCloud

positivedata = df1[ df1['Recommended IND'] == 1]

positivedata =positivedata['Review Text']

negdata = df1[df1['Recommended IND'] == 0]

negdata= negdata['Review Text']



def wordcloud_draw(df1, color = 'white'):

    words = ' '.join(df1)

    cleaned_word = " ".join([word for word in words.split()

                              if(word!='clothes' and word!='shop')

                            ])

    wordcloud = WordCloud(stopwords=stop,

                      background_color=color,

                      width=2500,

                      height=2000

                     ).generate(cleaned_word)

    plt.figure(1,figsize=(10, 7))

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()

    

print("Positive words are as follows")

wordcloud_draw(positivedata,'white')

print("Negative words are as follows")

wordcloud_draw(negdata)
corpus =[]

X.index = np.arange(len(X))
for i in range(len(X)):

    review = re.sub('[^a-zA-z]',' ',X['Review Text'][i])

    review = review.lower()

    review = review.split()

    ps = PorterStemmer()

    review =[ps.stem(i) for i in review if not i in set(stopwords.words('english'))]

    review =' '.join(review)

    corpus.append(review)
from sklearn.feature_extraction.text import CountVectorizer as CV

cv  = CV(max_features = 3000,ngram_range=(1,1))

X_cv = cv.fit_transform(corpus).toarray()

y = y.values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_cv, y, test_size = 0.20, random_state = 0)
from sklearn.naive_bayes import BernoulliNB

classifier = BernoulliNB()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)

print('accuracy:',acc)
from sklearn.feature_extraction.text import TfidfVectorizer as TV

tv  = TV(ngram_range =(1,1),max_features = 3000)

X_tv = tv.fit_transform(corpus).toarray()
X_train, X_test, y_train, y_test = train_test_split(X_tv, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("accuracy:" , acc)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier()

classifier.fit(X_train,y_train)

preds=classifier.predict(X_test)

rf_accuracy=accuracy_score(preds,y_test)

print("Random Forest Model accuracy",rf_accuracy)
import xgboost as xgb

xgb=xgb.XGBClassifier()

xgb.fit(X_train,y_train)

preds2=xgb.predict(X_test)

xgb_accuracy=accuracy_score(preds2,y_test)

print("XGBoost Model accuracy",xgb_accuracy)
from sklearn.linear_model import LogisticRegressionCV

classifier=LogisticRegressionCV(cv=6,scoring='accuracy',random_state=0,n_jobs=-1,verbose=3,max_iter=500).fit(X_train,y_train)

y_pred1 = classifier.predict(X_test)
from sklearn import metrics

print("Logistic Regression Accuracy:",metrics.accuracy_score(y_test, y_pred1))
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = 3000)

tokenizer.fit_on_texts(corpus)

sequences = tokenizer.texts_to_sequences(corpus)

padded = pad_sequences(sequences, padding='post')

word_index = tokenizer.word_index

count = 0

for i,j in word_index.items():

    if count == 11:

        break

    print(i,j)

    count = count+1
embedding_dim = 64

model = tf.keras.Sequential([

    tf.keras.layers.Embedding(3000, embedding_dim),

    tf.keras.layers.GlobalAveragePooling1D(),

    tf.keras.layers.Dense(6, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')

])



model.summary()
num_epochs = 10



model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(padded,y,epochs= num_epochs,validation_split= 0.39)
loss = model.history.history

loss = pd.DataFrame(loss)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

t = f.suptitle('Basic ANN Performance', fontsize=12)

f.subplots_adjust(top=0.85, wspace=0.3)



epoch_list = range(1,11)

ax1.plot(epoch_list, loss['accuracy'], label='Train Accuracy')

ax1.plot(epoch_list, loss['val_accuracy'], label='Validation Accuracy')

ax1.set_xticks(np.arange(0, 11, 1))

ax1.set_ylabel('Accuracy Value')

ax1.set_xlabel('Epoch')

ax1.set_title('Accuracy')

l1 = ax1.legend(loc="best")



ax2.plot(epoch_list, loss['loss'], label='Train Loss')

ax2.plot(epoch_list, loss['val_loss'], label='Validation Loss')

ax2.set_xticks(np.arange(0, 11, 1))

ax2.set_ylabel('Loss Value')

ax2.set_xlabel('Epoch')

ax2.set_title('Loss')

l2 = ax2.legend(loc="best")
sample_string = "I hate this dress"

sample = tokenizer.texts_to_sequences(sample_string)

padded_sample = pad_sequences(sample, padding='post')

print("Padded sample", padded_sample.T)

print("Probabilty of a person recommending :",model.predict(padded_sample.T)[0][0]*100,"%")
sample_string = "i love the fabric"

sample = tokenizer.texts_to_sequences(sample_string)

padded_sample = pad_sequences(sample, padding='post')

print("Padded sample", padded_sample.T)

print("Probabilty of a person recommending :",model.predict(padded_sample.T)[0][0]*100,"%")