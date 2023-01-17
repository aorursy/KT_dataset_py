# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input dat
#a files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

data = pd.read_csv("../input/spam.csv",encoding='latin-1')


# Any results you write to the current directory are saved as output.
data.head()
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v1":"label", "v2":"sms"})
data.head()
data['y'] = data.label.map({'ham': 0, 'spam': 1})
print(data.shape)
print(data.label.value_counts())
# Add a column for the lenght of the SMS
data['length'] = data.sms.str.len()
data.head()
spam = data[data['label'] == 'spam']
ham = data[data['label'] == 'ham']
print("Data for the spam:")
print(spam.length.describe())

print("\nData for the ham:")
print(ham.length.describe())



# Import neccesary libraries 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from wordcloud import WordCloud 
import matplotlib.pyplot as plt


#nltk.download('stopwords')
#stemmer = SnowballStemmer('english')
word_set = set(nltk.corpus.words.words()) # a set containing all english words to determine  
stop_words = set(stopwords.words("english"))
# strings to store long strings of word for creating 
ham_wordlist = ''
total_ham_words = 0
total_misspelled_ham = 0

spam_wordlist = ''
total_spam_words = 0
total_misspelled_spam = 0

# tokenize and remove non alphanumerical 
tknzr = RegexpTokenizer(r'\w+')
for text in ham.sms:
    tokens = tknzr.tokenize(text.lower())
    # Remove all word
    for word in tokens:
        total_ham_words += 1 # increment total words for every word
        if word not in stop_words: # only save words that are not in stop words
            ham_wordlist = ham_wordlist + ' ' + word  
        if word not in word_set:
            total_misspelled_ham += 1 # count the total of misspelled words 

    
for text in spam.sms:
    tokens = tknzr.tokenize(text.lower())
    # Remove all word
    for word in tokens:
        total_spam_words += 1 # increment total words for every word
        if word not in stop_words: # only save words that are not in stop words
            spam_wordlist = spam_wordlist + ' ' + word 
        if word not in word_set:
            total_misspelled_spam += 1 # count the total of misspelled words 
    
spam_wordcloud = WordCloud(background_color="lightgrey", width=600, height=400).generate(spam_wordlist)
ham_wordcloud = WordCloud(background_color="lightgrey", width=600, height=400).generate(ham_wordlist)
# Ham wordcloud
plt.figure( figsize=(10,8))
plt.imshow(ham_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
# Spam wordcloud

plt.figure( figsize=(10,8))
plt.imshow(spam_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()
print('HAM \n total words: {} \n total misspells: {} \n %: {}'.format(total_ham_words,
                                                                     total_misspelled_ham, 
                                                                     total_misspelled_ham*100 / total_ham_words))

print('SPAM \n total words: {} \n total misspells: {} \n %: {}'.format(total_spam_words,
                                                                     total_misspelled_spam, 
                                                                     total_misspelled_spam*100 / total_spam_words))
# Function to calculate the number of misspells in each message
def calculate_misspells(x):
    #print(x)
    tokens = tknzr.tokenize(x.lower())
    #print(tokens)
    corr_spelled = [word for word in tokens if word in word_set]
    if len(tokens) == 0:
        return 0
    return len(corr_spelled)/len(tokens)
data['misspells'] = data.sms.apply(calculate_misspells)
spam = data[data['label'] == 'spam']
ham = data[data['label'] == 'ham']
print("Data for the spam:")
print(spam.misspells.describe())

print("\nData for the ham:")
print(ham.misspells.describe())
data_ready = data.drop([ "label"], axis=1)
data_ready.head()
# Create a function to remove all stopwords from the 
def remove_stopwords(x):
    #print(x)
    tokens = tknzr.tokenize(x.lower())
    #print(tokens)
    stop_removed = [word for word in tokens if word not in stop_words]
    
    return " ".join(stop_removed)
data_ready.sms = data.sms.apply(remove_stopwords)
data_x = data_ready.drop(['y'], axis=1)
data_x.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data_x,data["y"], test_size = 0.2, random_state = 1)
from sklearn.feature_extraction.text import TfidfVectorizer
transvector = TfidfVectorizer()

tfidf1 = transvector.fit_transform(X_train.sms)
X_train_df = tfidf1.todense()
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
# The misspells are already scaled between 0-1 but I had to include it due to some wierd error 
# when I tried to only minmax the length feature. 
scaler = MinMaxScaler() 
X_train[['length', 'misspells']] = scaler.fit_transform(X_train[['length', 'misspells']])
X_train.head()
# Convert the Pandas dataframe so that it is a Numpy matrix to concatinate with the tfidf features
X_train[['length', 'misspells']].as_matrix()
X_train_final = np.concatenate((X_train_df , X_train[['length', 'misspells']].as_matrix()), axis=1)
# Transform test set
tfidf_test = transvector.transform(X_test.sms)
X_test_df = tfidf_test.todense()
X_test[['length', 'misspells']] = scaler.transform(X_test[['length', 'misspells']])
X_test_final = np.concatenate((X_test_df , X_test[['length', 'misspells']].as_matrix()), axis=1)
# Try using both naive bayes models 
prediction = dict()
from sklearn.naive_bayes import GaussianNB, MultinomialNB
gnb = GaussianNB()
clf = MultinomialNB()
gnb.fit(X_train_final,y_train)
clf.fit(X_train_final,y_train)
prediction["gaussian"] = gnb.predict(X_test_final)
prediction["multinom"] = clf.predict(X_test_final)
# Compare models 
print("F-score Gaussian, F-score Multinom, Accuracy Gaussian, Accuracy Multinom")
from sklearn.metrics import fbeta_score, accuracy_score
print(fbeta_score( y_test, prediction["gaussian"], average='macro', beta=1.5))
print(fbeta_score( y_test, prediction["multinom"], average='macro', beta=1.5))
print(accuracy_score( y_test, prediction["gaussian"]))
print(accuracy_score( y_test, prediction["multinom"]))
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# Perform Grid Search on multinomial
param_grid = {'alpha': [0, 0.5, 1, 2,5,10], 'fit_prior': [True, False]}
multinom = MultinomialNB()
scorer = make_scorer(fbeta_score, beta=1.5)
clf = GridSearchCV(multinom, param_grid, scoring=scorer)
clf.fit(X_train_final,y_train)
best_clf = clf.best_estimator_
best_predictions = best_clf.predict(X_test_final)
print("Best model F-beta and Accuracy:")
print(fbeta_score( y_test, best_predictions, average='macro', beta=1.5))
print(accuracy_score( y_test, best_predictions))
#Benchmark model: 
print("Benchmark model metrics on complete set and test set:")
#whole dataset
print(fbeta_score( data.y, np.zeros_like(data.y), average='macro', beta=1.5))
print(accuracy_score( data.y, np.zeros_like(data.y)))
#test set
print(fbeta_score( y_test, np.zeros_like(y_test), average='macro', beta=1.5))
print(accuracy_score( y_test, np.zeros_like(y_test)))
# Missclassified as spam
X_test[y_test < best_predictions ]
# Missclassified as ham
X_test[y_test > best_predictions]
from sklearn.model_selection import *
# Testing the robustness of the model


kf = KFold(n_splits=5)
kf.get_n_splits(data_x)
print("Scores of the different folds")
for train_index, test_index in kf.split(data_x):
    X_train, X_test = data_x.iloc[train_index], data_x.iloc[test_index]
    y_train, y_test = data.y.iloc[train_index], data.y.iloc[test_index]
    
    
    tfidf_train = transvector.fit_transform(X_train.sms)
    X_train_df = tfidf_train.todense()
    X_train[['length', 'misspells']] = scaler.transform(X_train[['length', 'misspells']])
    X_train_final = np.concatenate((X_train_df , X_train[['length', 'misspells']].as_matrix()), axis=1)

    tfidf_test = transvector.transform(X_test.sms)
    X_test_df = tfidf_test.todense()
    X_test[['length', 'misspells']] = scaler.transform(X_test[['length', 'misspells']])
    X_test_final = np.concatenate((X_test_df , X_test[['length', 'misspells']].as_matrix()), axis=1)
    
    clf = MultinomialNB(alpha=0.5, fit_prior=False)
    clf.fit(X_train_final,y_train)
    
    predictions = clf.predict(X_test_final)
    
    print("fbeta:")
    print(fbeta_score( y_test, predictions, average='macro', beta=1.5))
    print("accuracy:")
    print(accuracy_score( y_test, predictions))

