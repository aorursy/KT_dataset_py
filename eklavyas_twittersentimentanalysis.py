import re

import itertools

import numpy as np

import pandas as pd

from sklearn.svm import SVC

import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC

from stop_words import get_stop_words

from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer
# Get stop words

stop_words = get_stop_words('english')
# Tokenise and filter 

def tokenise(myStr):



    # r is for python raw string, to avoid backslashes issues

    cleanText = re.compile(r'[^0-9a-zA-Z:\-|\/\\_@#]')

    tokens = cleanText.split(myStr)



    wordList = []



    for t in tokens:

        if (t not in stop_words) and (len(t)>1):

            wordList.append(t)



    return wordList
# Read data, remove whitespace, and lower case

data = pd.read_csv('/kaggle/input/tweetsentiments/tweets_sentiment.csv', encoding = 'ISO-8859-1', usecols=[0, 1])

data['SentimentText'] = data['SentimentText'].str.strip()

data['SentimentText'] = data['SentimentText'].str.lower()
# Tokenise and filter each tweet

listOfLists = []

for i in range(len(data)):

    listOfLists.append(tokenise(data['SentimentText'][i]))

data['Tokens'] = pd.DataFrame({'Tokens': listOfLists})
# Counts to see the dataset's ratio balance

total_positive_count = (data['Sentiment']==1).sum()

total_negative_count = (data['Sentiment']==0).sum()



total_prob_positive_count = total_positive_count/len(data)

total_prob_negative_count = total_negative_count/len(data)
# Total number of tweets

len(data)
# Positive and negative tweets count

print("Positive:", total_positive_count, "\nNegative:", total_negative_count)
# We can see that the data is balanced between the positive and negative tweets

print("Positive:", total_prob_positive_count, "\nNegative:", total_prob_negative_count)
# The average length of tweets

print('The mean number of words in a tweet is',(data['Tokens'].str.len()).mean())
# The average length of tweets

print('The standard deviation of words in a tweet is',(data['Tokens'].str.len()).std())
# Show the longest tweetsk

data['Tokens'].str.len().sort_values(ascending=False).head()
# This converts the list of words into space-separated strings

data['Tokens'] = data['Tokens'].apply(lambda x: ' '.join(x))
# Transform the data into occurrences

count_vect = CountVectorizer()  

counts = count_vect.fit_transform(data['Tokens'])  
# Transform into inverted indice

transformer = TfidfTransformer().fit(counts)

counts = transformer.transform(counts)
counts.shape
# Split data train:test 80:20

train_x, test_x, train_y, test_y = train_test_split(counts, data['Sentiment'],

                                                    train_size=0.8, test_size=0.2, random_state=6)

class_names = ['Negative', 'Positive']
# Plot confusion matrix

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap='coolwarm'):

    

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes)

    plt.yticks(tick_marks, classes)

    

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")

    

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    

    plt.tight_layout()
def make_model(model, name):



    # Train the model using the training sets 

    model.fit(train_x, train_y)

    

    # Training score output

    print('Training score:', model.score(train_x, train_y))



    # Predict Output 

    predicted = model.predict(test_x)



    # Testing score output

    print('Testing score:', model.score(test_x, test_y))

    

    # Mean square error

    print('Mean-squared error:', mean_squared_error(test_y, predicted))

    

    # Confusion matrix

    cnf_matrix = confusion_matrix(test_y, predicted)

    print('Confusion Matrix without normalisation:\n', cnf_matrix)  

    

    # Draw non-normalised confusion matrix

    #plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False, title='Confusion matrix without normalisation')

    

    # Plot normalized confusion matrix

    #plt.figure()

    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Confusion matrix with normalisation')

    

    

    # Save plot

    plt.savefig(name+'.png', bbox_inches='tight', dpi=100)

    

    # Show

    plt.show()
make_model(MultinomialNB(fit_prior=False), 'MNBNoPrior')
make_model(MultinomialNB(fit_prior=True), 'MNBWithPrior')
from sklearn.linear_model import SGDClassifier



make_model(SGDClassifier(loss='hinge', penalty='l2', random_state=42), 'LinearSGD')

make_model(LogisticRegression(), 'LogisticalReg')
from sklearn.decomposition import TruncatedSVD



svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)

svd.fit(train_x)  
print(svd.explained_variance_ratio_)
print(svd.explained_variance_ratio_.sum())
print(svd.singular_values_)
# This converts the list of words into space-separated strings

# data['Tokens'] = data['Tokens'].apply(lambda x: ' '.join(x))



# Transform the data into occurrences

count_vect = CountVectorizer(ngram_range=(1,2))  

counts = count_vect.fit_transform(data['Tokens'])  



# Transform into inverted indice

transformer = TfidfTransformer().fit(counts)

counts = transformer.transform(counts)



# Split data train:test 80:20

train_x, test_x, train_y, test_y = train_test_split(counts, data['Sentiment'],

                                                    train_size=0.8, test_size=0.2, random_state=6)
make_model(LinearSVC(), 'SVC_1_2')
# Do counts again



# Transform the data into occurrences

count_vect = CountVectorizer(ngram_range=(1,3))  

counts = count_vect.fit_transform(data['Tokens'])  



# Transform into inverted indice

transformer = TfidfTransformer().fit(counts)

counts = transformer.transform(counts)



# Split data train:test 80:20

train_x, test_x, train_y, test_y = train_test_split(counts, data['Sentiment'],

                                                    train_size=0.8, test_size=0.2, random_state=6)
make_model(LinearSVC(), 'SVC_1_3')
# Do counts again



# Transform the data into occurrences

count_vect = CountVectorizer(ngram_range=(1,4))  

counts = count_vect.fit_transform(data['Tokens'])  



# Transform into inverted indice

transformer = TfidfTransformer().fit(counts)

counts = transformer.transform(counts)



# Split data train:test 80:20

train_x, test_x, train_y, test_y = train_test_split(counts, data['Sentiment'],

                                                    train_size=0.8, test_size=0.2, random_state=6)
make_model(LinearSVC(), 'SVC_1_4')
# Do counts again



# Transform the data into occurrences

count_vect = CountVectorizer(ngram_range=(2,2))  

counts = count_vect.fit_transform(data['Tokens'])  



# Transform into inverted indice

transformer = TfidfTransformer().fit(counts)

counts = transformer.transform(counts)



# Split data train:test 80:20

train_x, test_x, train_y, test_y = train_test_split(counts, data['Sentiment'],

                                                    train_size=0.8, test_size=0.2, random_state=6)



make_model(LinearSVC(), 'SVC_2_2')
# Do counts again



# Transform the data into occurrences

count_vect = CountVectorizer(ngram_range=(2,3))  

counts = count_vect.fit_transform(data['Tokens'])  



# Transform into inverted indice

transformer = TfidfTransformer().fit(counts)

counts = transformer.transform(counts)



# Split data train:test 80:20

train_x, test_x, train_y, test_y = train_test_split(counts, data['Sentiment'],

                                                    train_size=0.8, test_size=0.2, random_state=6)



make_model(LinearSVC(), 'SVC_2_3')
# Do counts again



# Transform the data into occurrences

count_vect = CountVectorizer(ngram_range=(1,5))  

counts = count_vect.fit_transform(data['Tokens'])  



# Transform into inverted indice

transformer = TfidfTransformer().fit(counts)

counts = transformer.transform(counts)



# Split data train:test 80:20

train_x, test_x, train_y, test_y = train_test_split(counts, data['Sentiment'],

                                                    train_size=0.8, test_size=0.2, random_state=6)



make_model(LinearSVC(), 'SVC_1_5')
# Do counts again



# Transform the data into occurrences

count_vect = CountVectorizer(ngram_range=(1,6))  

counts = count_vect.fit_transform(data['Tokens'])  



# Transform into inverted indice

transformer = TfidfTransformer().fit(counts)

counts = transformer.transform(counts)



# Split data train:test 80:20

train_x, test_x, train_y, test_y = train_test_split(counts, data['Sentiment'],

                                                    train_size=0.8, test_size=0.2, random_state=6)



make_model(LinearSVC(), 'SVC_1_6')
# Data for plotting

t = np.arange(2, 7)

s = [0.79292, 0.79574, 0.79589, 0.79508, 0.79439]



fig, ax = plt.subplots()

ax.plot(t, s)



ax.set(xlabel='Max n_gram', ylabel='Score',

       title='Max n_gram vs score')

ax.grid()



plt.savefig('Plot.png', bbox_inches='tight', dpi=100)



plt.show()