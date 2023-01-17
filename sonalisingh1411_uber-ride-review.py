# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd





# Importing the dataset

dataset = pd.read_csv("../input/Uber_Ride_Reviews.csv")
#checking the dataset

dataset.head(30) 
#Checking the dimensions of dataset....

dataset.shape
#Checking the null values in the dataset

dataset.isnull().sum()
#Counting all unique values of column "sentiment" from dataset 



dataset["sentiment"].value_counts()

#Counting all unique values of column "ride_rating" from dataset 



dataset["ride_rating"].value_counts()

#visualize the "rating" of each customer with the help of Bar plot 

dataset["ride_rating"].value_counts().plot.bar(color='blue')

#Bar plot for sentiment column

dataset["sentiment"].value_counts().plot.bar(color='Pink')

#ride_rating column is nomore required for my prediction....hence removed

dataset = dataset.drop('ride_rating', 1)

dataset

# Cleaning the texts

import re

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

corpus = [] #Creating an empty list

for i in range(0, 1000):

    review = re.sub('[^a-zA-Z]', ' ', dataset['ride_review'][i])#Remove all special character like('',:,;,%,*,@,$,!,#) 

    #Convert complete sentences into lowercase. 

    review = review.lower() 

    #Spliting each sentences into different words.

    review = review.split()

    #Stemming example is love-loving,loved,lovely.etc.

    ps = PorterStemmer()

    #Removing unnecessary words like is,of,for,this...and only focusing on main meaning of the sentence. 

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

    review = ' '.join(review)

    corpus.append(review)

print(corpus)
print(review)
# Creating the Bag of Words model

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 1344)

X= cv.fit_transform(corpus).toarray()

X = X.transpose()#Transposing here just to match the dimensionsns with y-target variable



y = dataset.iloc[:, 1].values
print(X) #Separting every unique value and mapping it as 1 just like encode and decode..

print(y) # liked or not!

X.shape
y.shape


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer



import re

import string

import nltk



cleanup_re = re.compile('[^a-z]+')

def cleanup(sentence):

    sentence = sentence.lower()

    sentence = cleanup_re.sub(' ', sentence).strip()

    #sentence = " ".join(nltk.word_tokenize(sentence))

    return sentence



dataset["Reviews_Clean"] = dataset["ride_review"].apply(cleanup)
from wordcloud import WordCloud, STOPWORDS

import matplotlib as mpl

stopwords = set(STOPWORDS)





mpl.rcParams['font.size']=12                #10 

mpl.rcParams['savefig.dpi']=100             #72 

mpl.rcParams['figure.subplot.bottom']=0.1 







def show_wordcloud(dataset, title = None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=300,

        max_font_size=40, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

        

    ).generate(str(dataset))

    

    fig = plt.figure(1, figsize=(15, 15))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()

show_wordcloud(dataset["Reviews_Clean"])
show_wordcloud(dataset["Reviews_Clean"][dataset["sentiment"] == 1] , title="Postive Words")

show_wordcloud(dataset["Reviews_Clean"][dataset["sentiment"] == 0] , title="Negative Words")
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
# Fitting Naive Bayes to the Training set

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)







# Predicting the Test set results

predict_gaussianNB = classifier.predict(X_test)











# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predict_gaussianNB)

print(cm)

from sklearn import metrics #for checking the model accuracy

print('The accuracy of the Naive bayes is:',metrics.accuracy_score(y_test,predict_gaussianNB))

# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)



# Predicting the Test set results

Predict_Logistic = classifier.predict(X_test)





# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, Predict_Logistic)

print(cm)





from sklearn import metrics #for checking the model accuracy

print('The accuracy of the Logistic_regression is:',metrics.accuracy_score(y_test,Predict_Logistic))
# Fitting Decision Tree Regression to the dataset

from sklearn.tree import DecisionTreeClassifier

clr = DecisionTreeClassifier(random_state = 0)

clr.fit(X_train, y_train)



predict_Decisiontree = clr.predict(X_test)





# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predict_Decisiontree)

print(cm)



from sklearn import metrics #for checking the model accuracy

print('The accuracy of the Decisiontree is:',metrics.accuracy_score(y_test,predict_Decisiontree))
# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)





# Predicting the Test set results

predict_randomforest = classifier.predict(X_test)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predict_randomforest)

print(cm)



from sklearn import metrics #for checking the model accuracy

print('The accuracy of the Random forest is:',metrics.accuracy_score(y_test,predict_randomforest))