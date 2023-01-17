import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Load the iris dataset as an example
from sklearn.datasets import load_iris
iris= load_iris()
# Store the feature matrix (x) and response vector (y)
X = iris.data
y = iris.target
#Check the shapes of X and y
print (X.shape)
print (y.shape)
# Examine first 5 rows of feature matrix
pd.DataFrame(X, columns=iris.feature_names).head()
# Examine the response vector
print (y)
#import the class
from sklearn.neighbors import KNeighborsClassifier

#instantiate the model (with the default parameter)
knn = KNeighborsClassifier()

# fit the model with data (occurs in-place)
knn.fit(X,y)
#  Predict the response for a new observation
knn.predict([[3,5,4,1]])

#Note: Say if the features we specify above is 5 instead of 4, it will miserably fail. This is 
# beacuse the model is trained with only 4 features like Sepal length, Sepal Width, Petal length, and Petal width.

#Similarly, if the features mentioned are not in the specified order, it will also create an error. It has to follow 
#the order as mentioned.
#Example text for model training (SMS messaging)
simple_train = ['call you tonight', 'call me a cab', 'Please call me.... PLEASE!']
#import and instantiate CountVectorizer (With default parameters)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
# Learn the 'Vocabulary' of training data
vect.fit_transform(simple_train)
# Examine the fitted vocubulary
vect.get_feature_names()
# transform training data into a 'document-term-matrix'
simple_train_dtm = vect.transform(simple_train)
simple_train_dtm
# Convert Sparse matrix to dense matrix
simple_train_dtm.toarray()
# examine the vocabulary and document-term-matrix together
pd.DataFrame(simple_train_dtm.toarray(), columns=vect.get_feature_names())
# Check the type of the document-term-matrix
type(simple_train_dtm)
#examine the sparse matrix contents
print(simple_train_dtm)
#example text for model testing
simple_test= ["please don't call me"]
#transform testing data into a document-term-matrix
simple_test_dtm = vect.transform(simple_test)
simple_test_dtm.toarray()
# examine the vocabulary and document-term-matrix together
pd.DataFrame(simple_test_dtm.toarray(), columns=vect.get_feature_names())

#Note: So here what happend to the word "Dont" present in the text??
#####
##### The answer is, the model during the training has seen only 3 x 6 matrix, where it has not encountered a word 
#called 'dont'. So, if it predicts a new word during the testing, it drops the word. This means that the model must
#be trained in such way to add the word called 'dont', if it encounters it in future.
import pandas as pd
#read the file into pandas using the relative path
text= pd.read_csv("../input/traintsv/train.tsv",delimiter='\t',encoding='utf-8')
#examine the shape
text.shape
# examine the first 5 rows
text.head()
text.isnull().sum()
text.dropna(subset=['brand_name'],how='all', inplace=True)
text.dropna(subset=['item_description'],how='all', inplace=True)
text.isnull().sum()
text.head(5)
X =pd.DataFrame()
X['item_description']= text['item_description']
X['brand_name']= text['brand_name']
X.shape
X1= X.ix[:5000,:]
X1.shape
X1.loc[:, 'brand_name'] = 'Brand1'
X1.head(10)
X2= X.ix[9000:,:]
X2.loc[:,'brand_name'] = 'Brand2'
X2.shape
X2.tail(10)
Xtext = pd.concat([X1, X2], axis=0)
Xtext.head(5)
Xtext.shape
#convert brand_name to a numeric value
Xtext['brand_num'] = Xtext.brand_name.map({'Brand1': 0, 'Brand2': 1})
#how to define X and y from IRIS dataset for use with a model
X= iris.data
y=iris.target
print(X.shape)
print(y.shape)
#how to define X and y from the text data for use with COUNTVECTORIZE
X= Xtext.item_description
y= Xtext.brand_name
print(X.shape)
print(y.shape)
      
#Note: It must be one dimentional object, as we need to convert it into 2D by using countvectorizer
# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# instantiate the vectorizer
vect = CountVectorizer()
# learn training data vocabulary, then use it to create a document-term matrix
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
# equivalently: combine fit and transform into a single step
X_train_dtm = vect.fit_transform(X_train)
# examine the document-term matrix
X_train_dtm
# transform testing data (using fitted vocabulary) into a document-term matrix
X_test_dtm = vect.transform(X_test)
X_test_dtm
# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
# train the model using X_train_dtm (timing it with an IPython "magic command")
%time nb.fit(X_train_dtm, y_train)
# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test_dtm)
#calculate accuracy of class predictions from sklearn import metrics metrics.accuracy_score(y_test, y_pred_class)
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)
