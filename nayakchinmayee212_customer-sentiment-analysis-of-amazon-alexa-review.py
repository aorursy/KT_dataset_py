# Importing the packages



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline
import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Loading the dataset (Here .tsv means tabs separated value)



df_reviews=pd.read_csv("/kaggle/input/amazon-alexa-reviews/amazon_alexa.tsv",sep='\t')

df_reviews.head()
df_reviews.shape             # Checking the number of rows and columns.
df_reviews.info()
df_reviews.describe()                  # There are 2 numeric columns 'rating' and 'feedback'
df_reviews['date']=pd.to_datetime(df_reviews['date'])
df_reviews['date'].dt.year.value_counts()
df_reviews['date'].min()
df_reviews['date'].max()
# Lets have a look at the time series plot.

plt.figure(figsize=(15,6))

sns.countplot(x='date',data=df_reviews)

plt.xticks(rotation=90)

plt.show();
# Lets look at the purchases on each of the three months



sns.countplot(df_reviews['date'].dt.month)
# This is the count of products sold or the reviews recieved in each month.



df_reviews['date'].dt.month.value_counts()
sns.countplot(x='rating',data=df_reviews)
df_reviews.rating.value_counts()
sns.countplot(x='feedback',data=df_reviews)
# Now lets have a look at the word count of the reviews.



df_reviews['length']=df_reviews['verified_reviews'].apply(lambda x:len(x.split(" ")))

df_reviews.head()
# Lets have a look at the histogram of the length of reviews.

plt.hist(x='length',data=df_reviews,bins=30);
df_reviews.length.describe()
# Is there a relation between review word count and the rating

sns.lmplot(x='length',y='rating',data=df_reviews)
neg=df_reviews[df_reviews['feedback']== 0]
from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

text = neg['verified_reviews'].values

wordcloud = WordCloud(

width=3000,

height=2000,

background_color = 'black',

stopwords = STOPWORDS).generate(str(text))

fig = plt.figure(

figsize=(40,30),

facecolor = 'k',

edgecolor = 'k')

plt.imshow(wordcloud,interpolation='bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
sns.countplot(x='rating',data=neg)
N = df_reviews.shape[0]

N
corpus = []



import re

import nltk                                                 # For text preprocessing we use nltk

from nltk.corpus import stopwords                           # Removing all the stopwords

from nltk.stem.porter import PorterStemmer                  # Reducing words to base form
nltk.set_proxy('SYSTEM PROXY')



##nltk.download()

nltk.download('stopwords')
ps=PorterStemmer()



for i in range (0,N):

    review = re.sub('[^a-zA-Z ]',' ',df_reviews['verified_reviews'][i]) # Removing special symbols like...,! and kkeeping only 

    review = review.lower()                                             # Lower case

    review = review.split()                                             # String split into words

    review = [ps.stem(word) for word in review                         # Reducing words to base form

    if not word in set(stopwords.words('english'))]

    review = " ".join(review)

    corpus.append(review)
corpus
#TF-IDF Vectorizer



from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(use_idf=True, strip_accents='ascii')
y = df_reviews['feedback']
X = vectorizer.fit_transform(corpus)
X.shape
y.shape
# Split the test and train



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42, test_size=0.2)
# Importing the decission tree classifier



from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier(class_weight='balanced')
dt_clf.fit(X_train, y_train)               # Model fitting on X_train,y_train

y_pred = dt_clf.predict(X_test)            # Model prediction on X_test
# Split the test and train

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Confusion Matrix

cm = confusion_matrix(y_test,y_pred)

print(cm)
# Classification Report

cr = classification_report(y_test ,y_pred)

print(cr)
# Accuracy Score

acc=accuracy_score(y_test,y_pred)

print(acc)