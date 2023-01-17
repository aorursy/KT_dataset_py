# Import pandas and numpy
import pandas as pd  # for file loading and data manipulation
import numpy as np # for linear algebra

# Load dataset
# As dataset is tab separated we have to specify the separator '\t'
# As there are no lables we have to assign labels i.e.names=['label', 'text']

sms=pd.read_csv('SMSSpamCollection', sep='\t', names=['label', 'text'])

# Check the head of dataset

sms.head(2)

# Check size of dataset
sms.shape

# there are a total of 5572 SMS under column 'text' which are labelled as ham(not spam) and spam under column 'label'.

# Take the 1st message to do all the cleaning/manipulation and then loop the process for all messages
# messages are under laleb text and the 1st item can be accessed using index 0

#message=sms['text'][0]
#Check 1st message content

# Further imports

# import regular expression
import re
# import natural language toolkit
import nltk
# import stopwords
from nltk.corpus import stopwords
# import PorterStemmer for stemming
# wiki: stemming is the process of reducing inflected (or sometimes derived) words to their word stem, base or root form.
# Ex: liked to like 
from nltk.stem.porter import PorterStemmer

# Create an object for PorterStemmer
ps=PorterStemmer()

# Step 1
# Pattern to remove everything except a-z and A-Z
#message=re.sub('[^a-zA-Z]',' ', message) # run this separately for understanding purpose
# output: Go until jurong point  crazy   Available only in bugis n great world la e buffet    Cine there got amore wat

# Step 2
# covert to lower case to remove duplicate words ex: Like & like
#message=message.lower()
# output: go until jurong point  crazy   available only in bugis n great world la e buffet    cine there got amore wat

# Step 3
# Covert to list using split function
#message=message.split()
# output: ['go','until','jurong','point','crazy','available','only','in','bugis','n','great','world','la','e','buffet',
#'cine','there', 'got','amore','wat']

# Step 4
# 1.Apply stemming on each word before feeding it to for loop. 
# wiki: stemming is the process of reducing inflected (or sometimes derived) words to their word stem, base or root form
# 2.Use stopwords database to remove word which does not change sense of statement even if removed ex: this, that, there etc

#message=[ps.stem(word) for word in message if not word in set(stopwords.words('english'))]
# output: ['go','jurong','point','crazi','avail','bugi','n','great','world','la','e','buffet','cine','got','amor','wat']

# Step 5
# covert the list into a string
#message=' '.join(message)
# output: go jurong point crazi avail bugi n great world la e buffet cine got amor wat

# With above step we are done with cleaning part. Now we need to put all this code in a for loop and run it for all 5572 text messages.<br> Above 5 lines are for explanation purpose and you should remove them as for loop is going to take care of cleaning process for all the text messages.

allsms=[] # initialize empty list for adding all text messages
for i in range(0, 5572):
    message=sms['text'][i]
    message=re.sub('[^a-zA-Z]',' ', message)
    message=message.lower()
    message=message.split()
    message=[ps.stem(word) for word in message if not word in set(stopwords.words('english'))]
    message=' '.join(message)
    allsms.append(message) # add messages to allsms list.

# The CountVectorizer provides a simple way to both tokenize a collection of text documents and build a vocabulary of known words, but also to encode new documents using that vocabulary.
# The vectors returned from a call to transform() will be sparse vectors, and you can transform them back to numpy arrays to look and better understand what is going on by calling the toarray() function.
# ref: https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(allsms).toarray() # prepare feature valirable

y=sms.iloc[:,0] # prepare target variable

# y contains categorical value and we need to change this to numerical value by using label encoding
# LabelEncoder will simply take the categorical variable and label them form 0 till n.
# Here we have only two categories hence we dont need OneHotEncoder

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

# Split the data i.e. training set 80% and test set 20%

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Import all classification algorithms

# Algorithm and metrics imports
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, accuracy_score

# Prepare a list of all algorithms

models=[]
models.append(('LoR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM', SVC()))
models.append(('NB', GaussianNB()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))

# Using for loop fit all algorithms to training data, predict on test data and calculate accuracy using accuracy_score

results=[]
names=[]
for name, model in models:
    clf=model.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    accu=accuracy_score(y_test, y_pred)
    names.append(name)
    results.append(accu)
    print(name, accu)
    
# create a dataframe for accuracy score for each model

pd.DataFrame(results,index=names,columns=['Accuracy'])

# Thank you. Share your comments
