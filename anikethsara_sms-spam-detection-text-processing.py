import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud # to Visualize our dataset in the WordCloud  
data = pd.read_csv("../input/spam.csv", encoding = 'ISO-8859-1')
data.shape
data.head()
#Removing the NaN variables
data = data.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"], axis = 1)
data.head()
# Re-naming the columns for our convenience 
data.columns = ["labels", "data"]
data.columns
#Converting the labels to the binary format 
data["b_labels"] = data["labels"].map({'ham': 0, 'spam':1})
data.head()
Y = data["b_labels"].values
#Fitting CV to convert the text data to vectors
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

cv = CountVectorizer(decode_error = 'ignore')
X = cv.fit_transform(data["data"])
Xtrain, Xtest, Ytrain , Ytest = train_test_split(X,Y, test_size = 0.33)
from sklearn.naive_bayes import MultinomialNB
#fitting the model
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
#Printing the scores
print("train score:", model.score(Xtrain, Ytrain))
print("test score:", model.score(Xtest, Ytest))

#Predicting Xtest
Ypred = model.predict(Xtest)
"""tfidf = TfidfVectorizer(decode_error = "ignore")
Xt = tfidf.fit_transform(data['data'])

Xt_train, Xt_test, Y_train , Y_test = train_test_split(X,Y, test_size = 0.33)
model.fit(Xt_train, Ytrain)
print("train score:", model.score(Xt_train, Ytrain))
print("test score:", model.score(Xt_test, Ytest))"""
def visualize(label):
    words = ''
    for msg in data[data['labels']== label]['data']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width = 600, height = 400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
visualize('spam')
visualize('ham')
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Ypred,Ytest)
print(cm)
accuracy = (cm[0][0]+cm[1][1])/(cm[0][1]+cm[0][0]+cm[1][1]+cm[1][0])
print(accuracy)
data['predictions'] = model.predict(X)
data.head()
#Figuring out where we are predicitng not spam in the place of spam
sneaky_spam = data[(data['predictions'] == 0) & (data['b_labels'] == 1)]["data"]
for msg in sneaky_spam:
    print(msg)
#Figuring out where we are predicitng spam in the place of not spam
sneaky_not_spam = data[(data['predictions'] == 1) & (data['b_labels'] == 0)]["data"]
for msg in sneaky_not_spam:
    print(msg)
