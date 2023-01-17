#loading the necessary modules and loading the data

import pandas as pd

import numpy as np



# reading in the data with correct encoding, utf-8 throws and error.

messages = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',encoding='ISO-8859-1')



# removing two unnecessary cols

messages = messages[['v1', 'v2']]

messages = messages.rename({'v1': 'Label',

                           'v2': 'SMS'}, axis=1)

messages.head()
round(messages['Label'].value_counts(normalize=True)*100)
#randomizing the dataset before we split. 

messages = messages.sample(frac=1, random_state=1)



#importing display module for displaying several tables/graphs

import IPython.display



#seperating train 80%, and test 20%

train = messages.iloc[:4458, :]

test = messages.iloc[4458:, :]



display(round(test['Label'].value_counts(normalize=True)*100))

display(round(train['Label'].value_counts(normalize=True)*100))
'''removing punctuation from messages. 

re.sub('\W', ' ', 'Message') This is our regex for replacing any upper/lowercase 

character that is not a letter, digits.'''

pd.options.mode.chained_assignment = None





train['SMS'] = train['SMS'].str.replace(r'\W', ' ')

train['SMS'] = train['SMS'].str.lower()
#creating a vocabulary containing all unique words from all messages.

vocabulary = []



for sms in train['SMS'].str.split():

    for word in sms:

        vocabulary.append(word)



vocabulary = list(set(vocabulary))

len(vocabulary)
#Function to create a dictionary of words with word count for each SMS.

word_counts_per_sms = {unique_word: [0] * (len(train['SMS'])) for unique_word in vocabulary}



for index, sms in enumerate(train['SMS'].str.split()):

    for word in sms:

        word_counts_per_sms[word][index] += 1

        

# transforming dictionary to a dataframe.

final_training = pd.DataFrame(word_counts_per_sms)
#concatenating the train and final_training datasets.

dfs = [train, final_training]

final_training_set = pd.concat(dfs, axis=1)
# finding probabilities of spam and ham messages

p_ham_spam = round(final_training_set['Label'].value_counts(normalize=True),2)

p_ham = p_ham_spam['ham']

p_spam = p_ham_spam['spam']





#seperating ham and spam messages

ham_messages = final_training_set.loc[final_training_set['Label']=='ham', 'SMS']

spam_messages = final_training_set.loc[final_training_set['Label']=='spam', 'SMS']





#counting number of words for ham and spam messages seperately

n_ham = 0

n_spam = 0





for message in ham_messages:

    message = str(message)

    message = message.split()

    n_ham += len(message)



for message in spam_messages:

    message = str(message)

    message = message.split()

    n_spam += len(message)

    

# Lapllace smoothing with a = 1

a = 1





# number of words in vocabulary

n_vocabulary = len(vocabulary)
# we will create two dictionaries to store probabilities for each type of message

p_word_given_spam = {unique_word:0 for unique_word in vocabulary} 

p_word_given_ham = {unique_word:0 for unique_word in vocabulary} 





# we will create a counter of each type of word using collections.Counter object

import collections





spam_words = []

ham_words = []

for message in ham_messages.str.split():

    for word in message:

        ham_words.append(word)

for message in spam_messages.str.split():

    for word in message:

        spam_words.append(word)

     

    

# Now that we have a list of all spam/ham words, we will create a dictionary of their count. 

ham_word_count = collections.Counter(ham_words)

spam_word_count = collections.Counter(spam_words)



for word in vocabulary:

    n_ham_word = ham_word_count[word]

    n_spam_word = spam_word_count[word]

    

    p_word_given_spam[word] = (n_spam_word + a) / (n_spam + a * n_vocabulary)

    p_word_given_ham[word] = (n_ham_word + a) / (n_ham + a * n_vocabulary)
import re



def classify(message):



    message = re.sub('\W', ' ', message)

    message = message.lower()

    message = message.split()



    

    # calculation of probabilities for spam and ham

    p_spam_given_message = p_spam

    p_ham_given_message = p_ham

    

    

    #iterate

    for word in message:

        if word in p_word_given_spam:

            p_spam_given_message *= p_word_given_spam[word]

        if word in p_word_given_ham:

            p_ham_given_message *= p_word_given_ham[word]

    



    #classification based on comparison results

    if p_ham_given_message > p_spam_given_message:

        return 'ham'

    elif p_ham_given_message < p_spam_given_message:

        return 'spam'

    else:

        return 'needs human classification!'
message_1 = 'WINNER!! This is the secret code to unlock the money: C3421.'

message_2 = "Sounds good, Tom, then see u there"

classify(message_1), classify(message_2)
# applying our function to our test dataset

test['predicted'] = test['SMS'].apply(classify)

test.head()
# we will check the accuracy of our model. 

correct = 0

total = len(test.SMS)



for row in test.iterrows():

    label = row[1][0]

    predicted = row[1][2]

    if label == predicted:

        correct += 1

accuracy = round((correct / total) * 100, 2)

accuracy