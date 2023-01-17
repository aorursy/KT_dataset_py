import numpy as np

import pandas as pd

import re

from sklearn.metrics import recall_score, precision_score
sms_df = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv', header=1, encoding='latin-1', names=['Label', 'SMS', 'Unknown1', 'Unknown2', 'Unknown3'])

sms_df.head()
sms_df.drop(['Unknown1','Unknown2','Unknown3'], axis=1, inplace=True)
sms_df.info()
sms_df.groupby('Label').describe()
list(sms_df[sms_df['Label'] == 'spam']['SMS'])
data_randomized = sms_df.sample(frac=1, random_state=1)

split_index = round(len(data_randomized) * 0.8)

sms_train = data_randomized[:split_index].reset_index(drop=True)

sms_test = data_randomized[split_index:].reset_index(drop=True)
def clean_and_split_message(message):

    message = message.lower()

    message = re.sub(r'\W', ' ', message)

    return message.split()



sms_train['SMS'] = sms_train['SMS'].apply(clean_and_split_message)
vocabulary = {word for sms_words in list(sms_train['SMS']) for word in sms_words}

word_counts_per_sms = {unique_word: [0] * len(sms_train['SMS']) for unique_word in vocabulary}



for index, sms in enumerate(sms_train['SMS']):

    for word in sms:

        word_counts_per_sms[word][index] += 1

        

sms_train = pd.concat([sms_train, pd.DataFrame(word_counts_per_sms)], axis=1)
sms_train.head()
vocab_cols = sms_train.columns[2:]

N_ham = sms_train[sms_train['Label'] == 'ham'][vocabulary].sum(axis=1).sum()

N_spam = sms_train[sms_train['Label'] == 'spam'][vocabulary].sum(axis=1).sum()

alpha = 1

P_ham = sms_train[sms_train['Label'] == 'ham'].shape[0]/sms_train.shape[0]

P_spam = sms_train[sms_train['Label'] == 'spam'].shape[0]/sms_train.shape[0]

N_vocab = len(vocabulary)
P_wi_given_ham = { wi:0 for wi in vocabulary}

P_wi_given_spam = { wi:0 for wi in vocabulary}

sms_train_ham = sms_train[sms_train['Label'] == 'ham']

sms_train_spam = sms_train[sms_train['Label'] == 'spam']



for wi in vocabulary:

    N_wi_given_ham = sms_train_ham[wi].sum()

    N_wi_given_spam = sms_train_spam[wi].sum()

    P_wi_given_ham[wi] = (N_wi_given_ham + alpha)/(N_ham + alpha*N_vocab)

    P_wi_given_spam[wi] = (N_wi_given_spam + alpha)/(N_spam + alpha*N_vocab)
def classify(message):

    message = clean_and_split_message(message)

    

    P_ham_given_message = P_ham

    P_spam_given_message = P_spam

    

    for word in message:

        if word in P_wi_given_ham:

            P_ham_given_message *= P_wi_given_ham[word]

        if word in P_wi_given_spam:

            P_spam_given_message *= P_wi_given_spam[word]



    if P_ham_given_message > P_spam_given_message:

        return 'ham'

    elif P_spam_given_message > P_ham_given_message:

        return 'spam'

    else:

        return 'needs human classification'
def accuracy(y_true, predicted):

    return len(y_true[y_true == predicted])/len(y_true)
predictions = sms_test['SMS'].apply(classify)

print(accuracy(sms_test['Label'], predictions))

print(recall_score(sms_test['Label'], predictions, pos_label='spam'))

print(precision_score(sms_test['Label'], predictions, pos_label='spam'))
predictions.value_counts()
sms_test[sms_test['Label'] != predictions].values