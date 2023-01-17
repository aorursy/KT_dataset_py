import pandas as pd
import numpy as np
import re
colec = pd.read_csv('../input/sms-data-labelled-spam-and-non-spam/SMSSpamCollection', sep='\t', header=None)
colec
colec.columns = ['Label', 'SMS']
colec.head()
colec.shape
ham_perc = ((len(colec[colec['Label'] == 'ham']))/len(colec))*100
spam_perc = ((len(colec[colec['Label'] == 'spam']))/len(colec))*100
ham_perc
spam_perc
# Randomize the dataset
data_randomized = colec.sample(frac=1, random_state=1)

# Calculate index for split
training_test_index = round(len(data_randomized) * 0.8)

# Training/Test split
training_set = data_randomized[:training_test_index].reset_index(drop=True)
test_set = data_randomized[training_test_index:].reset_index(drop=True)

print(training_set.shape)
print(test_set.shape)
len(training_set)
len(test_set)
(len(training_set[training_set['Label'] == 'spam'])/len(training_set))*100
training_set['SMS'] = training_set['SMS'].str.replace('\W', ' ')
training_set['SMS'] = training_set['SMS'].str.lower()
training_set.head()
training_set['SMS'] = training_set['SMS'].str.split()

vocabulary = []

for sms in training_set['SMS']:
    for word in sms:
        vocabulary.append(word)
training_set['SMS'].head()
vocabulary = list(set(vocabulary))
len(vocabulary)
word_counts_per_sms = {unique_word: [0] * len(training_set['SMS']) for unique_word in vocabulary}

for index, sms in enumerate(training_set['SMS']):
    for word in sms:
        word_counts_per_sms[word][index] += 1
word_counts = pd.DataFrame(word_counts_per_sms)
word_counts.head()
training_set_clean = pd.concat([training_set, word_counts], axis=1)
training_set_clean.shape
training_set_clean = training_set_clean[training_set_clean['Label'].notnull()]
training_set_clean.shape
spam_messages = training_set_clean[training_set_clean['Label'] == 'spam']
ham_messages = training_set_clean[training_set_clean['Label'] == 'ham']
p_spam = len(spam_messages)/len(training_set_clean)
p_spam
len(training_set_clean)
p_ham = len(ham_messages)/len(training_set_clean)
p_ham
n_spam = spam_messages['SMS'].apply(len).sum()
n_spam
n_ham = ham_messages['SMS'].apply(len).sum()
n_ham
n_vocabulary = len(vocabulary)

n_vocabulary
alpha = 1
word_counts_per_sms = {unique_word: [0] * len(training_set['SMS']) for unique_word in vocabulary}

for index, sms in enumerate(training_set['SMS']):
    for word in sms:
        word_counts_per_sms[word][index] += 1
parameters_spam = {unique_word:0 for unique_word in vocabulary}
for word in vocabulary:
    n_word_given_spam = spam_messages[word].sum()
    p_word_given_spam = (n_word_given_spam + alpha)/(n_spam + alpha * n_vocabulary)
    parameters_spam[word] = p_word_given_spam
parameters_ham = {unique_word:0 for unique_word in vocabulary}
for word in vocabulary:
    n_word_given_ham = ham_messages[word].sum()
    p_word_given_ham = (n_word_given_ham + alpha)/(n_ham + alpha * n_vocabulary)
    parameters_ham[word] = p_word_given_ham
def classify(message):
    '''
    message: a string
    '''
    
    message = re.sub('\W', ' ', message)
    message = message.lower().split()
    
    p_spam_given_message = p_spam
    p_ham_given_message = p_ham

    for word in message:
        if word in parameters_spam:
            p_spam_given_message *= parameters_spam[word]
            
        if word in parameters_ham:
            p_ham_given_message *= parameters_ham[word]
   
    if p_ham_given_message > p_spam_given_message:
        return 'ham'
    if p_spam_given_message > p_ham_given_message:
        return 'spam'
    if p_spam_given_message == p_ham_given_message:
        return 'needs human classification'
classify('WINNER!! This is the secret code to unlock the money: C3421.')
classify('Sounds good, Tom, then see u there')
test_set['predicted'] = test_set['SMS'].apply(classify)
test_set.head()
correct = 0
total = len(test_set)
corr = len(test_set[test_set['Label'] == test_set['predicted']])
corr
wrong = len(test_set[test_set['Label'] != test_set['predicted']])
wrong
(corr/total)*100
test_set[test_set['Label'] != test_set['predicted']]