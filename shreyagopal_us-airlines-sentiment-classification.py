#importing the required libraries & packages
import pandas as pd
import numpy as np
import time
import argparse
import string
from sklearn.model_selection import train_test_split
from nltk.tokenize import regexp_tokenize
from datetime import datetime
import pytz
#loading the US Airline Sentiment data
data_frame = pd.read_csv('../input/Tweets.csv')
data_frame.head()
data_frame.shape
#repalcing the categorical values of 'airline_sentiment' to numeric values
data_frame['airline_sentiment'].replace(('neutral', 'positive', 'negative'), (0, 1, 2), inplace=True)
data_frame['airline_sentiment'].value_counts()
#forming the feature & label variables
data = data_frame['text'].values.tolist()
labels = data_frame['airline_sentiment'].values.tolist()
#First five samples text
data[:5]
#first 5 samples label
labels[:5]
#splitting the data into 80 and 20 split
train_X, test_X, y_train, y_test = train_test_split(data, labels, test_size=0.2, 
                                                    random_state=42, shuffle=True)

print(f'Number of training examples: {len(train_X)}')
print(f'Number of testing examples: {len(test_X)}')
# Here is a default pattern for tokenization
default_pattern =  r"""(?x)                  
                        (?:[A-Z]\.)+          
                        |\$?\d+(?:\.\d+)?%?    
                        |\w+(?:[-']\w+)*      
                        |\.\.\.               
                        |(?:[.,;"'?():-_`])    
                    """
#funtion for tokenizing the data
""" Tokenize sentence with specific pattern
Arguments: text {str} -- sentence to be tokenized, such as "I love NLP"
Keyword Arguments: pattern {str} -- reg-expression pattern for tokenizer (default: {default_pattern})
Returns: list -- list of tokenized words, such as ['I', 'love', 'nlp'] """

def tokenize(text, pattern = default_pattern):
    text = text.lower()
    return regexp_tokenize(text, pattern)
# Tokenize training text into tokens
tokenized_text = []
for i in range(0, len(train_X)):
    tokenized_text.append(tokenize(train_X[i]))

X_train = tokenized_text

# Tokenize testing text into tokens
tokenized_text = []
for i in range(0, len(test_X)):
    tokenized_text.append(tokenize(test_X[i]))

X_test = tokenized_text
#tokenized train & test data
print(X_train[0], X_train[1])
print(X_test[0])
#building dictionary
""" Function: To create a dictionary of tokens from the data
Arguments: data in the type - list
Returns: Sorted dictionary of the tokens and their count in the data """

def createDictionary(data):
    dictionary = dict()
    for sample in  data:
        for token in sample:
            dictionary[token] = dictionary.get(token, 0) + 1
    
    #sorting the dictionary based on the values
    sorted_dict = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_dict)
bog = createDictionary(X_train)
#top 10 items in the dictionary
print("Top 10 tokens in the training dictionary:\n")
list(bog.items())[:10]
#Navie Bayes Classifier 
class NBClassifier:

    def __init__(self, X_train, y_train, size):
        tz_NY = pytz.timezone('America/New_York') 
        print("Model Start Time:", datetime.now(tz_NY).strftime("%H:%M:%S"))
        self.X_train = X_train
        self.y_train = y_train
        self.size = size

    def createDictionary(self):
        dictionary = dict()
        for sample in  X_train:
            for token in sample:
                dictionary[token] = dictionary.get(token, 0) + 1
        #sorting the dictionary based on the values
        sorted_dict = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_dict)
    
    def fit(self):
        """ Function: To compute the count of words in training data dictionary
        Arguments: Trianing data & Size of dictionary
        Returns: dictionary of tokens with their class wise probabilities """
      
        X_train_dict = self.createDictionary()
        if self.size == 'full':
            self.words_list = list(X_train_dict.keys())
            self.words_count = dict.fromkeys(self.words_list, None)
        else:
            self.words_list = list(X_train_dict.keys())[:int(self.size)]
            self.words_count = dict.fromkeys(self.words_list, None)
            
        #DataFrame of training data
        train = pd.DataFrame(columns = ['X_train', 'y_train'])
        train['X_train'] = X_train
        train['y_train'] = y_train

        train_0 = train.copy()[train['y_train'] == 0]
        train_1 = train.copy()[train['y_train'] == 1]
        train_2 = train.copy()[train['y_train'] == 2]

        #computing the prior of each class
        Pr0 = train_0.shape[0]/train.shape[0]
        Pr1 = train_1.shape[0]/train.shape[0]
        Pr2 = train_2.shape[0]/train.shape[0]

        self.Prior = np.array([Pr0, Pr1, Pr2])

        #converting list of lists into a list
        def flatList(listOfList):
            flatten = []
            for elem in listOfList:
                flatten.extend(elem)
            return flatten
  
        #Creating the data list for each class - tokens of each class
        X_train_0 = flatList(train[train['y_train'] == 0]['X_train'].tolist())
        X_train_1 = flatList(train[train['y_train'] == 1]['X_train'].tolist())
        X_train_2 = flatList(train[train['y_train'] == 2]['X_train'].tolist())
    
        self.X_train_len = np.array([len(X_train_0), len(X_train_1), len(X_train_2)])

        for token in self.words_list:
            #list to store three word counts of a token
            res = []

            #inserting count of token in class 0: Neutral
            res.insert(0, X_train_0.count(token))

            #inserting count of token in class 1: Positive
            res.insert(1, X_train_1.count(token))

              #inserting count of token in class 2: Negative
            res.insert(2, X_train_2.count(token))

            #assigning the count list to its token in the dictionary 
            self.words_count[token] = res
        return self

    def predict(self, X_test):
        """ Function: Predicts the label of the data
        Arguments: self and the test data
        Returns: List of predicted labels for the test data """     
        pred = []
        for sample in X_test:
            mul = np.array([1,1,1])
            for tokens in sample:
                vocab_count = len(self.words_list)
                if tokens in self.words_list:
                    prob = ((np.array(self.words_count[tokens])+1) / (self.X_train_len + vocab_count))
                mul = mul * prob
            val = mul * self.Prior
            pred.append(np.argmax(val))
        tz_NY = pytz.timezone('America/New_York') 
        print("Model End Time:", datetime.now(tz_NY).strftime("%H:%M:%S"))
        return pred
    
    def score(self, pred, labels):
        """ Function: To compute the perfoemance of the model
        Arguments: self, predicted labels and actual labels of the test data
        Returns: Number of lables correctly predicted and the accuracy of the model """
        correct = (np.array(pred) == np.array(labels)).sum()
        accuracy = correct/len(pred)
        return correct, accuracy
# Creating holders to store the model performance results
attributes = []
corr = []
acc = []

#function to call for storing the results
def storeResults(attr, cor,ac):
    attributes.append(attr)
    corr.append(round(cor, 3))
    acc.append(round(ac, 3))
#training the classifier     
nb = NBClassifier(X_train, y_train, 'full')  
nb.fit()

#predicting the labels for test samples
y_pred = nb.predict(X_test)

#Checking
print("NBClassifier Model miss any prediction???", len(X_test) != len(y_pred))
#Performance of the classifier
cor1, acc1 = nb.score(y_pred, y_test)
print("Count of Correct Predictions:", cor1)
print("Accuracy of the model: %i / %i = %.4f " %(cor1, len(y_pred), acc1))
#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('Unprocessed Data', cor1, acc1)
#string of punctiations
string.punctuation
#Removing the punctuation
'''Function: Removes the punctuation from the tokens
   Arguments: list of text data samples
   Returns: list of tokens of each sample without punctuation '''
def removePunctuation(data):
    update = []
    for sample in data:
        #removing punctuation from the tokens
        re_punct = [''.join(char for char in word if char not in string.punctuation) for word in sample]
        #removes the empty strings
        re_punct = [word for word in re_punct if word]
       
        update.append(re_punct)
    return update
#Removing punctuation from training data text tokens  
X_train_P = removePunctuation(X_train)

#Removing punctuation from testing data text tokens
X_test_P = removePunctuation(X_test)

#train & test data after removing punctuation
print(X_train_P[0])
print(X_test_P[0])
#training the classifier     
nb_punct = NBClassifier(X_train_P, y_train, 'full')
nb_punct.fit()

#predicting the labels for test samples
y_pred_P = nb_punct.predict(X_test_P)

#Checking
print("NBClassifier Model miss any prediction???", len(X_test) != len(y_pred_P))
#Performance of the classifier
cor2, acc2 = nb_punct.score(y_pred_P, y_test)
print("Count of Correct Predictions:", cor2)
print("Accuracy of the model: %i / %i = %.4f " %(cor2, len(y_pred_P), acc2))
#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('No Punctuation Data', cor2, acc2)
'''Function: Removes the stopwords from the tokens
   Arguments: list of text data samples
   Returns: list of tokens of each sample without punctuation '''
def removeStopWords(data):
    update = []
    stopwords = ['the', 'at','i', 'of', 'us', 'have', 'a', 'you','ours', 'themselves', 
                 'that', 'this', 'be', 'is', 'for']
    for sample in data:
        #removing stopwords from tokenized data
        re_stop = [word for word in sample if word not in stopwords]
        
        update.append(re_stop)
    return update
#Removing stopwords from training data text tokens  
X_train_S = removeStopWords(X_train)

#Removing stopwords from testing data text tokens
X_test_S = removeStopWords(X_test)

#train & test data after removing stopwords
print(X_train_S[0])
print(X_test_S[0])
#training the classifier     
nb_stop = NBClassifier(X_train_S, y_train, 'full')
nb_stop.fit()

#predicting the labels for test samples
y_pred_S = nb_stop.predict(X_test_S)

#Checking
print("NBClassifier Model miss any prediction???", len(X_test) != len(y_pred_S))
#Performance of the classifier
cor3, acc3 = nb_stop.score(y_pred_S, y_test)
print("Count of Correct Predictions:", cor3)
print("Accuracy of the model: %i / %i = %.4f " %(cor3, len(y_pred_S), acc3))
#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('Removed few Stopwords', cor3, acc3)
#Removing stopwords from training data text tokens  
X_train_PS = removeStopWords(X_train_P)

#Removing stopwords from testing data text tokens
X_test_PS = removeStopWords(X_test_P)

#train & test data after removing stopwords
print(X_train_PS[0])
print(X_test_PS[0])
#training the classifier     
nb_PS = NBClassifier(X_train_PS, y_train, 'full')
nb_PS.fit()

#predicting the labels for test samples
y_pred_PS = nb_PS.predict(X_test_PS)

#Checking
print("NBClassifier Model miss any prediction???", len(X_test) != len(y_pred_PS))
#Performance of the classifier
cor4, acc4 = nb_PS.score(y_pred_PS, y_test)
print("Count of Correct Predictions:", cor4)
print("Accuracy of the model: %i / %i = %.4f " %(cor4, len(y_pred_PS), acc4))
#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('Removed both Punctuation & Few Stopwords', cor4, acc4)
#total tokens in training dictionary
print('Total tokens in the dictionary:', len(bog))
#training the classifier - 5000 tokens 
nb_5k = NBClassifier(X_train, y_train, '5000')
nb_5k.fit()

#predicting the labels for test samples
y_pred_5k = nb_5k.predict(X_test)

#Checking
print("NBClassifier Model miss any prediction???", len(X_test) != len(y_pred_5k))
#Performance of the classifier
cor5, acc5 = nb_5k.score(y_pred_5k, y_test)
print("Count of Correct Predictions:", cor5)
print("Accuracy of the model: %i / %i = %.4f " %(cor5, len(y_pred), acc5))
#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('5k Tokens of Voab - Unprocessed Data', cor5, acc5)
#training the classifier - 5000 tokens 
nb_5k_P = NBClassifier(X_train_P, y_train, '5000')
nb_5k_P.fit()

#predicting the labels for test samples
y_pred_5k_P = nb_5k_P.predict(X_test_P)

#Checking
print("NBClassifier Model miss any prediction???", len(X_test) != len(y_pred_5k_P))
#Performance of the classifier
cor6, acc6 = nb_5k.score(y_pred_5k_P, y_test)
print("Count of Correct Predictions:", cor6)
print("Accuracy of the model: %i / %i = %.4f " %(cor6, len(y_pred), acc6))
#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('5k Tokens of Voab - No Punctuation Data', cor6, acc6)
#training the classifier - 5000 tokens 
nb_5k_S = NBClassifier(X_train_S, y_train, '5000')
nb_5k_S.fit()

#predicting the labels for test samples
y_pred_5k_S = nb_5k_S.predict(X_test_S)

#Checking
print("NBClassifier Model miss any prediction???", len(X_test) != len(y_pred_5k_S))
#Performance of the classifier
cor7, acc7 = nb_5k_S.score(y_pred_5k_S, y_test)
print("Count of Correct Predictions:", cor7)
print("Accuracy of the model: %i / %i = %.4f " %(cor7, len(y_pred), acc7))
#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('5k Tokens of Voab - Removed few Stopwords', cor7, acc7)
#training the classifier - 5000 tokens 
nb_5k_PS = NBClassifier(X_train_PS, y_train, '5000')
nb_5k_PS.fit()

#predicting the labels for test samples
y_pred_5k_PS = nb_5k_PS.predict(X_test_PS)

#Checking
print("NBClassifier Model miss any prediction???", len(X_test) != len(y_pred_5k_PS))
#Performance of the classifier
cor8, acc8 = nb_5k_PS.score(y_pred_5k_PS, y_test)
print("Count of Correct Predictions:", cor8)
print("Accuracy of the model: %i / %i = %.4f " %(cor8, len(y_pred), acc8))
#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('5k Tokens of Voab - Removed both Punctuation & Few Stopwords', cor8, acc8)
#training the classifier - 5000 tokens 
nb_10k = NBClassifier(X_train, y_train, '5000')
nb_10k.fit()

#predicting the labels for test samples
y_pred_10k = nb_10k.predict(X_test)

#Checking
print("NBClassifier Model miss any prediction???", len(X_test) != len(y_pred_10k))
#Performance of the classifier
cor9, acc9 = nb_10k.score(y_pred_10k, y_test)
print("Count of Correct Predictions:", cor9)
print("Accuracy of the model: %i / %i = %.4f " %(cor9, len(y_pred), acc9))
#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('10k Tokens of Voab - Unprocessed Data', cor9, acc9)
#training the classifier - 10000 tokens 
nb_10k_P = NBClassifier(X_train_P, y_train, '10000')
nb_10k_P.fit()

#predicting the labels for test samples
y_pred_10k_P = nb_10k_P.predict(X_test_P)
  
#Checking
print("NBClassifier Model miss any prediction???", len(X_test) != len(y_pred_10k_P))
#Performance of the classifier
cor10, acc10 = nb_10k_P.score(y_pred_10k_P, y_test)
print("Count of Correct Predictions:", cor10)
print("Accuracy of the model: %i / %i = %.4f " %(cor10, len(y_pred), acc10))
#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('10k Tokens of Voab - No Punctuation Data', cor10, acc10)
#training the classifier - 10000 tokens 
nb_10k_S = NBClassifier(X_train_S, y_train, '10000')
nb_10k_S.fit()

#Sredicting the labels for test samSles
y_pred_10k_S = nb_10k_S.predict(X_test_S)
  
#Checking
print("NBClassifier Model miss any Srediction???", len(X_test) != len(y_pred_10k_S))
#Performance of the classifier
cor11, acc11 = nb_10k_S.score(y_pred_10k_S, y_test)
print("Count of Correct Predictions:", cor11)
print("Accuracy of the model: %i / %i = %.4f " %(cor11, len(y_pred), acc11))
#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('10k Tokens of Voab - Removed few Stopwords', cor11, acc11)
#training the claPSPSifier - 10000 tokenPS 
nb_10k_PS = NBClassifier(X_train_PS, y_train, '10000')
nb_10k_PS.fit()

#PSredicting the labelPS for tePSt PSamPSlePS
y_pred_10k_PS = nb_10k_PS.predict(X_test_PS)
  
#Checking
print("NBClaPSPSifier Model miSS any PSrediction???", len(X_test) != len(y_pred_10k_PS))
#Performance of the classifier
cor12, acc12 = nb_10k_PS.score(y_pred_10k_PS, y_test)
print("Count of Correct Predictions:", cor12)
print("Accuracy of the model: %i / %i = %.4f " %(cor12, len(y_pred), acc12))
#storing the results. The below mentioned order of parameter passing is important.
#Caution: Execute only once to avoid duplications.
storeResults('10k Tokens of Voab - Removed both Punctuation & Few Stopwords', cor12, acc12)
#creating dataframe
results = pd.DataFrame({ 'Data Modification': attributes,    
    'Correct Predictions': corr,
    'Model Accuracy': acc})
results.sort_values(by=['Model Accuracy', 'Correct Predictions'], ascending=False)