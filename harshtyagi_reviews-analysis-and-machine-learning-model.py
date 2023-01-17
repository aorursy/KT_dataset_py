import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.rcParams['figure.figsize'] = 13, 10
dataframe = pd.read_csv('../input/amazon_alexa.tsv', sep = '\t')
sns.countplot(dataframe['feedback'] )
print("Percentage of negative reviews: ", (len(dataframe[dataframe['feedback'] == 0]) * 100)/len(dataframe))
print("Percentage of Positive reviews: ", (len(dataframe[dataframe['feedback'] == 1]) * 100)/len(dataframe))
sns.countplot(dataframe['rating'])
ax = sns.countplot(x = 'variation', data = dataframe,palette="Blues_d", order = dataframe['variation'].value_counts().index)
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
ax = sns.barplot(x = 'variation', y = 'rating', data = dataframe)
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
dataframe.groupby('variation').mean().reset_index()
def wordclouds(x, label):
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    text = []
    dataframe_pos = x['feedback'] == label
    for i in range(0, len(x)):
        review = dataframe['verified_reviews'][i]
        text.append(review)
    text = " ".join(text for text in text)

    stopwords = set(STOPWORDS)
    stopwords.remove('not')
    wordcloud = WordCloud(stopwords=stopwords, background_color="black", max_font_size=100).generate(text)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    
wordclouds(dataframe, 1)
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
stop_words = set(stopwords.words('english'))
stop_words.remove('not')
for i in range(0, len(dataframe)):
    review = re.sub('[^a-zA-Z]', ' ', dataframe['verified_reviews'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in stop_words]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2500)
X = cv.fit_transform(corpus).toarray()
y = dataframe.iloc[:, 4].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn import tree
classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
print("The F1 score is: ", f1_score(y_test, y_pred, average="macro")*100)
print("The precision score is: ", precision_score(y_test, y_pred, average="macro")*(100))
print("The recall score is: ", recall_score(y_test, y_pred, average="macro")*100) 
print("The accuracy score is: ", accuracy_score(y_test, y_pred)*100)