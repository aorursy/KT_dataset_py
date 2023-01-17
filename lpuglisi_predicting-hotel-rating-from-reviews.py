from nltk.tokenize import word_tokenize, sent_tokenize 

from nltk import pos_tag

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB, GaussianNB

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import pandas as pd

import numpy as np

from IPython.display import display

import matplotlib.pyplot as plt

import itertools

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode

%matplotlib inline

init_notebook_mode()
data = pd.read_csv("../input/7282_1.csv")

#Filter out ratings that are zero

data = data[data['reviews.rating']>0]

#Renaming column names that have a '.' in them

data = data.rename(index=str, columns={'reviews.date':'reviewsdate', 'reviews.dateAdded':'reviewsdateAdded',

       'reviews.doRecommend':'reviewsdoRecommend', 'reviews.id':'reviewsid', 'reviews.rating':'reviewsrating', 'reviews.text':'reviewstext',

       'reviews.title':'reviewstitle', 'reviews.userCity':'reviewsuserCity', 'reviews.username':'reviewsusername',

       'reviews.userProvince':'reviewsuserProvince'})

data.reviewstext = data.reviewstext.fillna('x')

#A few hundred ratings had a score above 5, filtering these out

data = data[data['reviewsrating']<=5]

#A few hundred ratings had decimals, rounding each of those down to an integer

data.reviewsrating = data.reviewsrating.astype(int)
#Creating a function that I will use to clean review strings

#Function makes the string 'txt' lowercase, removes stopwords, finds the length, and pulls out only adjectives

#Returns a list of the length, cleaned txt, and only adjective txt

def cleanme(txt):

    sent = txt.lower()

    wrds = word_tokenize(sent)

    clwrds = [w for w in wrds if not w in stopwords.words('english')]

    ln = len(clwrds)

    pos = pd.DataFrame(pos_tag(wrds))

    pos = " ".join(list(pos[pos[1].str.contains("JJ")].iloc[:,0]))

    rt = [ln, " ".join(clwrds), pos]

    return(rt)
data.country.unique()
plt1 = go.Scatter(x = data.longitude, y=data.latitude, mode = 'markers')

lyt1 = go.Layout(title="Locations of Hotel Reviews", xaxis=dict(title='Longitude'), yaxis=dict(title='Latitude'))

fig1 = go.Figure(data=[plt1], layout=lyt1)

iplot(fig1)
#Filter to only include datapoints within the US

data = data[((data['latitude']<=50.0) & (data['latitude']>=24.0)) & ((data['longitude']<=-65.0) & (data['longitude']>=-122.0))]
#Create a field that shows the length of each review

tmp = list()

for i in range(len(data)):

    tmp.append(cleanme(data.iloc[i,:]['reviewstext']))

tmp = pd.DataFrame(tmp)

tmp.columns = ['reviewlen', 'cleanrev', 'adjreview']
#Add calculated columns back to the dataset

data = data.reset_index()

data = pd.concat([data,tmp], axis=1)

data.head()
plt2 = go.Histogram(x = data.reviewlen)

lyt2 = go.Layout(title="Frequency of Review Length", xaxis=dict(title='Review Length', range=[0,400]), yaxis=dict(title='Frequency'))

fig2 = go.Figure(data=[plt2], layout=lyt2)

iplot(fig2)
data = data.sort_values(by='reviewlen')

plt3 = go.Scatter(x = data.reviewlen, y = data.reviewsrating, mode='markers')

lyt3 = go.Layout(title="Review Length vs. Star Rating", xaxis=dict(title='Review Length'),yaxis=dict(title='Rating'))

fig3 = go.Figure(data=[plt3], layout=lyt3)

iplot(fig3)

print("Review Length to Rating Correlation:",data.reviewlen.corr(data.reviewsrating))
#Setting up the X and Y data, where X is the review text and Y is the rating

#Three different inputs will be used: original review text, cleaned review text, and only adjectives review text

x1 = data.reviewstext

x2 = data.cleanrev

x3 = data.adjreview

y = data.reviewsrating
#Creating a vectorizer to split the text into unigrams and bigrams

vect = TfidfVectorizer(ngram_range = (1,2))

x_vect1 = vect.fit_transform(x1)

x_vect2 = vect.fit_transform(x2)

x_vect3 = vect.fit_transform(x3)
#Making some simple functions for linear svc, knn, and naive bayes

def linsvc(x,y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 10)

    classf = LinearSVC()

    classf.fit(x_train, y_train)

    pred = classf.predict(x_test)

    print("Linear SVC:",accuracy_score(y_test, pred))

    return(y_test, pred)



def revknn(x,y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 10)

    classf = KNeighborsClassifier(n_neighbors=2)

    classf.fit(x_train, y_train)

    pred = classf.predict(x_test)

    print("kNN:",accuracy_score(y_test, pred))

    return(y_test, pred)



def revnb(x,y):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state = 10)

    classf = MultinomialNB()

    classf.fit(x_train, y_train)

    pred = classf.predict(x_test)

    print("Naive Bayes:",accuracy_score(y_test, pred))

    return(y_test, pred)
svmy1,svmp1 = linsvc(x_vect1,y)

svmy2,svmp2 = linsvc(x_vect2,y)

svmy3,svmp3 = linsvc(x_vect3,y)



knny1,knnp1 = revknn(x_vect1,y)

knny2,knnp2 = revknn(x_vect2,y)

knny3,knnp3 = revknn(x_vect3,y)



nby1,nbp1 = revnb(x_vect1,y)

nby2,nbp2 = revnb(x_vect2,y)

nby3,nbp3 = revnb(x_vect3,y)
#This function will plot a confusion matrix and is taken from the sklearn documentation with just some minor tweaks

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = np.around((cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]),decimals=2)

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
c1 = confusion_matrix(svmy1,svmp1)

c2 = confusion_matrix(svmy2,svmp2)

c3 = confusion_matrix(nby2,nbp2)

class_names = ['1', '2', '3', '4', '5']

plt.figure()

plot_confusion_matrix(c1, classes=class_names,normalize=False,title='Confusion matrix - SVM Full Review')

plt.figure()

plot_confusion_matrix(c2, classes=class_names,normalize=False,title='Confusion matrix - SVM No Stopwords')

plt.figure()

plot_confusion_matrix(c3, classes=class_names,normalize=False,title='Confusion matrix - NB No Stopwords')