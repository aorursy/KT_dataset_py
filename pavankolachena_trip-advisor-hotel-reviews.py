# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
import nltk as nlp
from nltk.corpus import wordnet
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
#nltk.download()
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud,ImageColorGenerator

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/trip-advisor-hotel-reviews/tripadvisor_hotel_reviews.csv")
train.head()
train.tail()
train.shape
train.dtypes
cat = train.select_dtypes("object")
cat.dtypes
cat_data = cat.astype("category")
cat_data.dtypes
train.describe()
corr_df = train.corr()
corr_df
sns.heatmap(corr_df , annot = True)
plt.show()
train.isnull().sum().sort_values(ascending = False)
train.isnull().sum()
len(train['Rating'].value_counts())
train.rename(columns={"Review" :"review","Rating" :"rating" }, inplace = True)
review_list=[]
for review in train.review:
    review=re.sub("[^a-zA-Z]"," ",review)
    review=review.lower()
    review=nltk.word_tokenize(review)
    lemma  = nlp.WordNetLemmatizer()
    review=[lemma.lemmatize(word) for word in review]
    review=" ".join(review)
    review_list.append(review)
max_features =250
count_vectorizer =CountVectorizer(max_features=max_features,stop_words="english")
sparce_matrix = count_vectorizer.fit_transform(review_list).toarray()
print("The 150 most commonly used {} words: {} ".format(max_features,count_vectorizer.get_feature_names()))
df_review_list = pd.DataFrame(review_list, columns = ['review'])
text = " ".join(str(each) for each in df_review_list.review)
# Create and generate a word cloud image:
wordcloud = WordCloud(max_words=250, background_color="white").generate(text)
plt.figure(figsize=(15,10))
# Display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
punctuations = """!()-![]{};:,+'"\,<>./?@#$%^&*_~Â""" 


def reviewParse(review):
    splitReview = review.split() 
    parsedReview = "".join([word.translate(str.maketrans('', '', punctuations)) + " " for word in splitReview]) 
    return parsedReview 

train["CleanReview"] = train["review"].apply(reviewParse) 
train.head() 
review = train["CleanReview"].copy() 


print("Example Sentence: ") 
print(review[26])

token = Tokenizer() 
token.fit_on_texts(review) 
texts = token.texts_to_sequences(review) 


print("Into a Sequence: ")
print(texts[26])

texts = pad_sequences(texts, padding='post') 


print("After Padding: ")
print(texts[26])
def encodeLabel(label):
    if label == 5 or label == 4: 
        return 2 
    if label == 3: 
        return 1 
    return 0 

labels = ["Negative", "Neutral", "Positive"] 
train["EncodedRating"] = train["rating"].apply(encodeLabel) 
train.head() 
X_train, X_test, y_train, y_test = train_test_split(texts, train["EncodedRating"], test_size = 0.25, random_state = 0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

seaborn.heatmap(cm)
plt.show()
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
print(classification_report(y_test,predictions))
# set width of bar
barWidth = 0.25
 
# set height of bar
bars1 = [12, 30, 1, 8, 22]
bars2 = [28, 6, 16, 5, 10]
bars3 = [29, 3, 24, 25, 17]
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
# Make the plot
plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='macro accuracy')
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='precision')
plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='recall')
 
# Add xticks on the middle of the group bars
plt.xlabel('group', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], ['A', 'B', 'C', 'D', 'E'])
 
# Create legend & Show graphic
plt.legend()
plt.show()
