import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/restaurant-reviews/Restaurant_Reviews.tsv' , delimiter= '\t')
dataset.head()
dataset.isnull().sum() # Checking for any null values in the dataset
import re       # regular expression              
import nltk # natural language toolkit
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []                       

# Iterating through all the reviews
for i in range(0,1000):
    # Removing unnecessary punctuations and numbers except letters and replacing removed words with space.
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    # converting review to lowercase
    review = review.lower()
    # Converting review to list(of Strings)
    review  = review.split()
    ps = PorterStemmer()
    words = stopwords.words('english')
    words.remove('not')
    words.remove('but')
    words.remove('is')
    # Loop through all words and keep those which are not in stopwords list.
    # set is much faster than a list and is considered when the review is very large eg. an article,a book
    review = [ps.stem(word) for word in review if not word in set(words)]
    # Joining back the review list to a string with each word seperated by a space.
    review = ' '.join(review)
    corpus.append(review)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
cv = CountVectorizer(max_features= 1566)                
X = cv.fit_transform(corpus).toarray()                 # toarray() is used to convert into matrix
y = dataset.iloc[:,1].values
X[0:5]
y[0:5]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.1, random_state = 25)
from sklearn.decomposition import PCA
pca = PCA(n_components=560)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
from sklearn.svm import SVC
clf = SVC(kernel = 'linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
results = pd.DataFrame({
    'Actual': np.array(y_test).flatten(),
    'Predicted': np.array(y_pred).flatten(),
})
results[1:20]
from sklearn.metrics import plot_confusion_matrix, accuracy_score
plot_confusion_matrix(clf,X_test , y_test, cmap = plt.cm.Blues)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy Score: ',accuracy)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))