# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import math

import re

from sklearn.model_selection import train_test_split
def tokenize(mail):                                                      # separating words

        mail=mail.lower()                                                    #convert to lowercase

        all_words= re.findall("[a-z0-9]+",mail)  #Taking all the words

        return set (all_words)

    

def count_words(training_set):                                           # total count of each word in spam and non spam mails

        """ pairs (mail, is_spam) """                                        # It will return a dictionary with word as key and list of counts in spam and non spam mails as value.

        counts={}

        for index, row in training_set.iterrows():

            for word in tokenize(row[0]):

                if word in counts:

                    counts[word][0 if row[1] else 1]+=1

                else:

                    counts[word]=[0,0]

                    counts[word][0 if row[1] else 1]+=1

        return counts

    

def word_probabilities(counts, total_spams, total_non_spams, k=0.5):      # It will return a triplet with word and probabilities

        word_triplet=[]

        for word in counts:

            List1=[]

            List1.append(word)

            List1.append((counts.get(word)[0]+k)/(total_spams+2*k))

            List1.append((counts.get(word)[1]+k)/(total_non_spams+2*k))

            word_triplet.append(List1)

        return word_triplet

    

def spam_probability(word_probs, mail):                                  # Here we will assign the probability to each mail

        mail_words=[]

        for x in mail:

            mail_words += tokenize(x)

        log_prob_if_spam = log_prob_if_not_spam = 0.0

        

        for word, prob_if_spam, prob_if_not_spam in word_probs:

            if word in mail_words:

#                 print(word, prob_if_spam, prob_if_not_spam)

                log_prob_if_spam += math.log(prob_if_spam)

                log_prob_if_not_spam += math.log(prob_if_not_spam)

                

        prob_if_spam = math.exp(log_prob_if_spam)

        prob_if_not_spam = math.exp(log_prob_if_not_spam)

        if (prob_if_spam + prob_if_not_spam == 0):

            return 0

        return prob_if_spam/ (prob_if_spam + prob_if_not_spam)
class NaiveBayesClassifier:

    def _init_(self, k=0.5):

        self.k= k

        self.word_probs = []                                                #Constructor

                      

    def train(self, training_set):

        num_spams = len([spam for spam in training_set.iloc[:,1] if spam])

        num_non_spams = len(training_set) - num_spams

                    

        word_counts= count_words(training_set)

        self.word_probs = word_probabilities(word_counts, num_spams, num_non_spams, 0.5)

                    

    def classify(self, mail):

        return spam_probability(self.word_probs, mail)

                     

    
spam_df=pd.read_csv('../input/spam-filter/emails.csv')

spam_df.head()                    
x= spam_df[['text']]

y=spam_df.spam
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.1, random_state = 0)
xTrain['spam']=yTrain

xTrain.head()
Naive=NaiveBayesClassifier()

Naive.train(xTrain)
submission=[]

for i, row in xTest.iterrows():

    prob=Naive.classify(row)

    

    if prob>0.6:

        submission.append(1)

    else:

        submission.append(0)
xTest['spam']=yTest

xTest['submission']=submission

print(xTest)
xTest['correct']=np.where((xTest['spam']==xTest['submission']), 1, 0)

xTest.head()
score=xTest['correct'].value_counts()
score
Accuracy = (score[1]/(score[1] + score[0]))
print("{} % of the data has been correctly predicted by our algorthm".format(Accuracy * 100))