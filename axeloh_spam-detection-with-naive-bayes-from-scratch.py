import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import string

import re

from nltk.stem.porter import PorterStemmer

import random

from sklearn.model_selection import train_test_split



data = pd.read_csv("../input/sms-spam-collection-dataset/spam.csv", encoding='latin-1')

data.head()
# Drop unwanted columns and rename remaining columns

if len(data.columns) > 3: 

    data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)

if data.columns[0] != 'label':

    data = data.rename(columns={'v1': 'label', 'v2': 'text'})

data.head()
# Count observations in each label

data.label.value_counts()

#data.info()
# Convert label to numerical variable

data['label'] = data.label.map({'ham': 0, 'spam': 1})

data.head()
# Extract every text 

texts = []

for index, row in data.iterrows():

    texts.append((row['text'], row['label']))

texts[:5]
# * * * PREPROCESSING * * * 

# Remove whitespace and punctutation 

tokenized = []

for t in texts:

    m = t[0]

    text = re.sub('[' + string.punctuation + ']', ' ', m)

    text = re.sub('[\n\t\r]', '', text)

    words = text.split()

    tokenized.append((words, t[1]))

tokenized[0] # First element
# Remove stopwords

stopwords = []

try:

    f = open('../input/stopword-lists-for-19-languages/englishST.txt', 'r')

    stopwords = f.read().split('\n')

except IOError:

    print('Problem opening file')

finally:

    f.close()

print('Sentence before stopwrods removed: \n', tokenized[51])

filtered = []

for t in tokenized:

    text = t[0]

    f_text = []

    for word in text:

        if word not in stopwords and len(word) > 2:

            f_text.append(word)

    filtered.append((f_text, t[1]))



print('\nSentence after stopwords removed: \n', filtered[51])
# Stem the words

stemmer = PorterStemmer()

stemmed = []

for t in filtered:

    text = t[0]

    stemmed_text = []

    for word in text:

        stemmed_word = stemmer.stem(word.lower())

        stemmed_text.append(stemmed_word)

    stemmed.append((stemmed_text, t[1]))



stemmed[51]
# Counting number of texts each word occurs

word_count = {}

for t in stemmed:

    text = t[0]

    already_counted = []

    for word in text:

        if word not in word_count:

            word_count[word] = 1

        elif word not in already_counted:

            word_count[word] += 1

            already_counted.append(word)



#  Removing the words that only occurs once

for i in range(len(stemmed)):

    stemmed[i] = (list(filter(lambda x: word_count[x] > 4, stemmed[i][0])), stemmed[i][1])
# Splitting data in trainingdata and testdata (80-20 ratio)

totaltexts = data.label.value_counts()

total = totaltexts[0] + totaltexts[1] # Total number of texts

test_number = int(0.20 * total) # Number of testing mails

# Picking randomly

test_set = []

taken = {}

while len(test_set) < test_number:

    #print(len(train_texts))

    num = random.randint(0, test_number - 1)

    if num not in taken.keys():

        test_set.append(stemmed.pop(num))

        taken[num] = 1



train_set = stemmed # Trainset is the remaining texts

        

# Total number of hams and spams

number_of_hams = data.label.value_counts()[0]

number_of_spams = data.label.value_counts()[1]



len(train_set)/total, len(test_set)/total
# * * * TRAINING THE MODEL * * * 



# meaning: Computing probabilities needed for P(Spam|Word)



# Need to train these 4 possibilities:

# 1) Probability that a word appears in spam messages

# 2) Probability that a word appears in ham messages

# 3) Overall probability that any given message is spam

# 4) Overall probability that any given message is not spam (is ham)



def p_appears_in_spam(word):

    count = 0

    total_spams = 0

    for t in train_set:

        text = t[0]

        if t[1] == 1:

            total_spams += 1

            if word in text:

                count += 1

    return count/total_spams

             



def p_appears_in_ham(word):

    count = 0

    total_hams = 0

    for t in train_set:

        text = t[0]

        if t[1] == 0:

            total_hams += 1

            if word in text:

                count += 1

    return count/total_hams



def total_spams_and_hams(tset):

    spams = 0

    hams = 0

    for t in tset:

        spams += 1 if t[1] == 1 else 0

        hams += 1 if t[1] == 0 else 0

    return spams, hams





p_spam = total_spams_and_hams(train_set)[0]/len(train_set) # Probability that a message is spam

p_ham = total_spams_and_hams(train_set)[1]/len(train_set) # Probability that a message is ham



# Finally we can compute P(Spam | Word)

def p_is_spam_given_word(word):

    return (p_appears_in_spam(word)*p_spam)/((p_appears_in_spam(word)*p_spam + p_appears_in_ham(word)*p_ham))



word = 'free'

print('Probability that a message is spam given the word "{}" is: {}'.format(word, p_is_spam_given_word(word)))
# Collecting the probabilities in a dictionary

probabilities = {}

for t in train_set:

    text = t[0]

    for word in text:

        if word not in probabilities:

            p = p_is_spam_given_word(word)

            if p == 0:

                probabilities[word] = 0.2 # To deal with the zero probability problem. Tweaking this value

            elif p == 1:

                probabilities[word] = 0.98 # Tweaking this value

            else:

                probabilities[word] = p
# * * * TESTING THE MODEL * * * 

# Training is done

# This function will be used to classify new messages, using the trained probabilities 



from functools import reduce

def p_is_spam(words):

    probs = []

    for word in words:

        if word in probabilities:

            probs.append(probabilities[word])

        # 'else' is for unseen word, a value to tweak

        # Assumes it is somewhat higher probability that an unseen word belongs to a ham message than a spam message

        # as 

        else:

            probs.append(0.4) 

    probs_not = list(map(lambda prob: 1-prob, probs))

    product = reduce(lambda x, y: x * y, probs, 1) 

    product_not = reduce(lambda x, y: x * y, probs_not, 1)

    return product/(product + product_not)
total_correct = 0

true_spam_as_spam = 0

true_spam_as_ham = 0

true_ham_as_ham = 0

true_ham_as_spam = 0



# Care most about minimizing false positives, that is: labeling non-spam messages as spam

false_positives = []



for t in test_set:

    guess = -1

    words = t[0]

    answer = t[1]

    p_spam = p_is_spam(words)

    # If p > 0.95, predict 'yes' (is spam)

    guess = 1 if p_spam > 0.95 else 0

    if guess == answer:

        total_correct += 1

        if answer == 0: # true negative

            true_ham_as_ham += 1

        else: # true positive

            true_spam_as_spam += 1 

    else:

        if answer == 0: # false positive

            true_ham_as_spam += 1

            false_positives.append((words, p_spam))

        else: # true negative

            true_spam_as_ham += 1



            

true_spams = total_spams_and_hams(test_set)[0]

true_hams = total_spams_and_hams(test_set)[1]



print('Total test texts: ', len(test_set))

print('Number of correct: ', total_correct)

print('Accuracy: ', total_correct*100/(true_spams+true_hams))

print('-------------------------------')

print('Ham precision: ', true_ham_as_ham/(true_ham_as_ham + true_spam_as_ham))

print('Ham recall: ', true_ham_as_ham/(true_ham_as_ham + true_ham_as_spam))

print('Spam precision: ', true_spam_as_spam/(true_spam_as_spam + true_ham_as_spam)) # Most important 

print('Spam recall: ', true_spam_as_spam/(true_spam_as_spam + true_spam_as_ham))

print('-------------------------------')

print('False Positives (hams that got labeled as spam):')

for i, (text, p) in enumerate(false_positives):

    print('{}: Words in text: {} | Degree of certainty: {}'.format(i+1, text, p))
# * * * VISUALISATIONS * * * 

from wordcloud import WordCloud



spam_words = ""

ham_words = ""



all = train_set + test_set



for t in all:

    text = t[0]

    s = ""

    for word in text:

        s += word + ' '

    if t[1] == 0:

        ham_words += s

    else:

        spam_words += s + ' '



# # Generate a word cloud image

spam_wordcloud = WordCloud(width=600, height=400).generate(spam_words)

ham_wordcloud = WordCloud(width=600, height=400).generate(ham_words)

#Spam Word cloud

plt.figure( figsize=(10,8), facecolor='k')

plt.imshow(spam_wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)
# Ham Word cloud

plt.figure( figsize=(10,8), facecolor='k')

plt.imshow(ham_wordcloud)

plt.axis("off")

plt.tight_layout(pad=0)