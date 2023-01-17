import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



pal = sns.color_palette()



print("Data files:")

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Read in data

df = pd.read_csv('../input/spam.csv', encoding='latin-1')



# Preview data

df.head()
# Drop redundant columns and rename columns so the titles are meaningful

df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

# Note: if you rename 'v2' as something like 'text', be sure not to

# overwrite the column when you create words as features (and so) 

# have columns named 'text' later on!

df = df.rename(columns={"v1":"label", "v2":"sms_text"})

df.head()
# How many spam messages are there?

df.label.value_counts()
print(round(747 / (747 + 4825) * 100, 2), "% of messages in our dataset are spam.")
# Character count



# Each line of code creates a list of messages, saving each message as a string.



# We want three lists: one with all messages in them, one with only ham messages 

# and one with only spam messages.

messages = pd.Series(df['sms_text'].tolist()).astype(str)

ham_messages = pd.Series(df[df['label'] == 'ham']['sms_text'].tolist()).astype(str)

spam_messages = pd.Series(df[df['label'] == 'spam']['sms_text'].tolist()).astype(str)



# Create the corresponding distribution of character counts for each list.

# We count the number of characters in each message using the `len` function.

dist_all = messages.apply(len)

dist_ham = ham_messages.apply(len)

dist_spam = spam_messages.apply(len)
# Plot distribution of character count of all messages



plt.figure(figsize=(12, 8))

plt.hist(dist_all, bins=100, range=[0,400], color=pal[3], normed=True, label='All')

plt.title('Normalised histogram of character count in all messages', fontsize=15)

plt.legend()

plt.xlabel('Number of characters', fontsize=15)

plt.ylabel('Probability', fontsize=15)



print('# Summary statistics for character count of all messages')

print('mean-all {:.2f} \nstd-all {:.2f} \nmin-all {:.2f} \nmax-all {:.2f}'.format(dist_all.mean(), 

                          dist_all.std(), dist_all.min(), dist_all.max()))
# Plot distributions of character counts for spam vs ham messages



plt.figure(figsize=(12,8))

plt.hist(dist_ham, bins=100, range=[0,250], color=pal[1], normed=True, label='ham')

plt.hist(dist_spam, bins=100, range=[0, 250], color=pal[2], normed=True, alpha=0.5, label='spam')

plt.title('Normalised histogram of character count in messages', fontsize=15)

plt.legend()

plt.xlabel('Number of characters', fontsize=15)

plt.ylabel('Probability', fontsize=15)



print('# Summary statistics for character count of ham vs spam messages')

print('mean-ham  {:.2f}   mean-spam {:.2f} \nstd-ham   {:.2f}   std-spam   {:.2f} \nmin-ham    {:.2f}   min-ham    {:.2f} \nmax-ham  {:.2f}   max-spam  {:.2f}'.format(dist_ham.mean(), 

                         dist_spam.mean(), dist_ham.std(), dist_spam.std(), dist_ham.min(), dist_spam.min(), dist_ham.max(), dist_spam.max()))
# Word Count



# We split each message into words using `.split(' ')`

# and count the number of words in each message using `len`.

dist_all = messages.apply(lambda x: len(x.split(' ')))

dist_ham = ham_messages.apply(lambda x: len(x.split(' ')))

dist_spam = spam_messages.apply(lambda x: len(x.split(' ')))
# Plot distribution of word count of all messages



plt.figure(figsize=(12, 8))

plt.hist(dist_all, bins=100, color=pal[3], normed=True, label='All')

plt.title('Normalised histogram of word count in all messages', fontsize=15)

plt.legend()

plt.xlabel('Number of words', fontsize=15)

plt.ylabel('Probability', fontsize=15)



print('# Summary statistics for word count of all messages')

print('mean-all {:.2f} \nstd-all {:.2f} \nmin-all {:.2f} \nmax-all {:.2f}'.format(dist_all.mean(), 

                          dist_all.std(), dist_all.min(), dist_all.max()))
# Plot distributions of word counts for spam vs ham messages



plt.figure(figsize=(12,8))

plt.hist(dist_ham, bins=65, range=[0,75], color=pal[1], normed=True, label='ham')

plt.hist(dist_spam, bins=65, range=[0, 75], color=pal[2], normed=True, alpha=0.5, label='spam')

plt.title('Normalised histogram of word count in messages', fontsize=15)

plt.legend()

plt.xlabel('Number of words', fontsize=15)

plt.ylabel('Probability', fontsize=15)



print('# Summary statistics for word count of ham vs spam messages')

print('mean-ham  {:.2f}   mean-spam {:.2f} \nstd-ham   {:.2f}   std-spam   {:.2f} \nmin-ham    {:.2f}   min-ham    {:.2f} \nmax-ham  {:.2f}   max-spam  {:.2f}'.format(dist_ham.mean(), 

                         dist_spam.mean(), dist_ham.std(), dist_spam.std(), dist_ham.min(), dist_spam.min(), dist_ham.max(), dist_spam.max()))
# Add our features to our dataframe

df['word_count'] = pd.Series(df['sms_text'].tolist()).astype(str).apply(lambda x: len(x.split(' ')))

df['char_count'] = pd.Series(df['sms_text'].tolist()).astype(str).apply(len)



# For some models the target label has to be int, float or bool

df['is_spam'] = (df['label'] == 'spam')



# Check things worked as expected

df.head()
from sklearn.model_selection import train_test_split



X = df[['word_count', 'char_count']]

y = df[['is_spam']]



# Split data into training and test sets

# TODO: Might want to split such that train and test sets have equal proportion of spam messages

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Split some training data for validation

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
import xgboost as xgb



# Set our parameters for xgboost

params = {}

params['objective'] = 'binary:logistic'

params['eval_metric'] = 'error'

params['eta'] = 0.02

params['max_depth'] = 4



d_train = xgb.DMatrix(X_train, label=y_train)

d_valid = xgb.DMatrix(X_valid, label=y_valid)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
from sklearn.metrics import accuracy_score



# Predict values for test set

d_test = xgb.DMatrix(X_test)

p_test = bst.predict(d_test)



# Apply function round() to each element in np array

# so predictions are all either 0 or 1.

npround = np.vectorize(round)

p_test_ints = npround(p_test)



# Error rate for test set

accuracy = accuracy_score(y_test, p_test_ints)

print("Test Accuracy: ", accuracy)
# To avoid cheating, we will first split the data into train and test sets and then only

# count top words for our training data.



X = df

y = df[['is_spam']]



# Split data into training and test sets

# TODO: Might want to split such that train and test sets have equal proportion of spam messages

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



ham_messages_train = pd.Series(X_train[X_train['label'] == 'ham']['sms_text'].tolist()).astype(str)

spam_messages_train = pd.Series(X_train[X_train['label'] == 'spam']['sms_text'].tolist()).astype(str)
from wordcloud import WordCloud

# WordCloud automatically excludes stop words



# Draw word cloud for spam messages

spam_messages_one_string = " ".join(spam_messages_train.astype(str))

spam_cloud = WordCloud().generate(spam_messages_one_string)

plt.figure(figsize=(12,8))

plt.imshow(spam_cloud)
# Draw word cloud for ham messages

ham_messages_one_string = " ".join(ham_messages_train.astype(str))

ham_cloud = WordCloud().generate(ham_messages_one_string)

plt.figure(figsize=(12,8))

plt.imshow(ham_cloud)
from collections import Counter

ham_words_list = ham_messages_one_string.split()

total_ham_words = len(ham_words_list)

print("Total number of words in ham messages: ", total_ham_words)

ham_words_dict = Counter(ham_words_list).most_common()

ham_words_dict[:25]
spam_words_list = spam_messages_one_string.split()

total_spam_words = len(spam_words_list)

print("Total number of words in spam messages: ", total_spam_words)

spam_words_dict = Counter(spam_words_list).most_common()

spam_words_dict[:25]
from nltk.corpus import stopwords

from collections import defaultdict

import operator



stopwords = set(stopwords.words("english"))



ham_words_lowercase = ham_messages_one_string.lower().split()



ham_words_nostop = []

for word in ham_words_lowercase:

    if word not in stopwords:

        ham_words_nostop.append(word)



ham_words_freq = Counter(ham_words_nostop).most_common()

ham_words_freq[:25]
spam_words_lowercase = spam_messages_one_string.lower().split()



spam_words_nostop = []

for word in spam_words_lowercase:

    if word not in stopwords:

        spam_words_nostop.append(word)



spam_words_freq = Counter(spam_words_nostop).most_common()

spam_words_top25 = spam_words_freq[:25]

spam_words_top25
spam_words_top25_list = [tuple[0] for tuple in spam_words_top25]

ham_words_top25 = [tuple[0] for tuple in ham_words_freq[:25]]
spam_words_top25_pruned = []

for word in spam_words_top25_list:

    if word not in ham_words_top25:

        spam_words_top25_pruned.append(word)

        

ham_words_top25_pruned = []

for word in ham_words_top25:

    if word not in spam_words_top25_list:

        ham_words_top25_pruned.append(word)



print("Number of non-duplicates in each list: ", len(spam_words_top25_pruned))
for word in (spam_words_top25_pruned + ham_words_top25_pruned):

    df[word] = (word in df['sms_text'])

    X_train[word] = (word in X_train['sms_text'])

    X_test[word] = (word in X_test['sms_text'])

df.head()
del X_train['sms_text']

del X_train['label']

del X_test['sms_text']

del X_test['label']



# Split some training data for validation

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_train.head()
d_train = xgb.DMatrix(X_train, label=y_train)

d_valid = xgb.DMatrix(X_valid, label=y_valid)



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



bst = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
# Predict values for test set

d_test = xgb.DMatrix(X_test)

p_test = bst.predict(d_test)



# Apply function round() to each element in np array

# so predictions are all either 0 or 1.

npround = np.vectorize(round)

p_test_ints = npround(p_test)



# Error rate for test set

accuracy = accuracy_score(y_test, p_test_ints)

print("Test Accuracy: ", accuracy)