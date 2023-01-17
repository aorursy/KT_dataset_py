#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

%config InlineBackend.figure_format = 'retina'
#import datasets
data = pd.read_csv("../input/spam.csv",encoding='latin-1')
data.head()
#drop unwanted columns and name change
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v1":"label", "v2":"text"})
# convert label to a numerical variable
data['label'] = data.label.map({'ham':0, 'spam':1})
data.head()
#count observations in each label
data.label.value_counts()
#text transformation (stopwords,lowering,stemming) and creating bag of words model using CountVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = data.iloc[:, 0].values
#showing first and last 20 features names
print(cv.get_feature_names()[0:20])
print(cv.get_feature_names()[-20:])
print(X.shape,y.shape)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
print(X_train.shape,X_test.shape)
#Visualisations
from wordcloud import WordCloud
ham_words = ''
spam_words = ''
spam = data[data.label == 1]
ham = data[data.label ==0]
for val in spam.text:
    text = re.sub('[^a-zA-Z]', ' ', val)
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    for words in text:
        spam_words = spam_words + words + ' '
    
for val in ham.text:
    text = re.sub('[^a-zA-Z]', ' ', val)
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
    for words in text:
          ham_words = ham_words + words + ' '
            
# Generate a word cloud image
spam_wordcloud = WordCloud(width=600, height=400).generate(spam_words)
ham_wordcloud = WordCloud(width=600, height=400).generate(ham_words)
#Spam Word cloud
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(spam_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
#Ham word cloud
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(ham_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
# Fitting Naive Bayes to the Training set (Gaussian NB)
prediction = dict()
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
classifier = GaussianNB()
classifier.fit(X_train, y_train)
# Predicting the Test set results
prediction["GaussianNB"] = classifier.predict(X_test)
accuracy_score(y_test,prediction["GaussianNB"])
# Fitting Naive Bayes to the Training set (Multinomial NB)
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
# Predicting the Test set results
prediction["MultinomialNB"] = classifier.predict(X_test)
accuracy_score(y_test,prediction["MultinomialNB"])
# Fitting Naive Bayes to the Training set (Multinomial NB)
from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(X_train, y_train)
# Predicting the Test set results
prediction["BernouliiNB"] = classifier.predict(X_test)
accuracy_score(y_test,prediction["BernouliiNB"])
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
# Predicting the Test set results
prediction["Logistic"] = classifier.predict(X_test)
accuracy_score(y_test,prediction["Logistic"])
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)
# Predicting the Test set results
prediction["KNN"] = classifier.predict(X_test)
accuracy_score(y_test,prediction["KNN"])
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train,y_train)
# Predicting the Test set results
prediction["RandomForrest"] = classifier.predict(X_test)
accuracy_score(y_test,prediction["RandomForrest"])
conf_mat = confusion_matrix(y_test, prediction['MultinomialNB'])
conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
sns.heatmap(conf_mat_normalized)
plt.ylabel('True label')
plt.xlabel('Predicted label')
print(conf_mat)
#model evaluation
print(classification_report(y_test, prediction['MultinomialNB'], target_names = ["Ham", "Spam"]))
