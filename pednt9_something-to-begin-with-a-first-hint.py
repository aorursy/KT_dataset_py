# load the basic librairies

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from collections import Counter
# load the data

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

sub = pd.read_csv('../input/sample_submission.csv')
train_df.head(3)
test_df.head(5)
# first feature: create a 'length' column

train_df['length'] = train_df.text.apply(len)

test_df['length'] = test_df.ciphertext.apply(len)





# filter the test dataframes by cypher level

df_level_1 = test_df[test_df.difficulty==1].copy()

df_level_2 = test_df[test_df.difficulty==2].copy()

df_level_3 = test_df[test_df.difficulty==3].copy()

df_level_4 = test_df[test_df.difficulty==4].copy()



df_level_1.head(3)
print('train_df.shape:', train_df.shape, '\ntest_df.shape:', test_df.shape)
for i in range(5):

    print(train_df.text[i], '\n')
plain_char_cntr = Counter(''.join(train_df['text'].values))

plain_stats_train = pd.DataFrame([[x[0], x[1]] for x in plain_char_cntr.items()], columns=['Letter', 'Frequency'])

plain_stats_train = plain_stats_train.sort_values(by='Frequency', ascending=False)



f, ax = plt.subplots(figsize=(15, 5))

plt.bar(np.array(range(len(plain_stats_train))) + 0.5, plain_stats_train['Frequency'].values)

plt.xticks(np.array(range(len(plain_stats_train))) + 0.5, plain_stats_train['Letter'].values)

plt.show()
for i in range(20):

    try:

        print(df_level_1.ciphertext[i], '\n')

    except KeyError:

        pass
plain_char_cntr = Counter(''.join(df_level_1['ciphertext'].values))

plain_stats_train = pd.DataFrame([[x[0], x[1]] for x in plain_char_cntr.items()], columns=['Letter', 'Frequency'])

plain_stats_train = plain_stats_train.sort_values(by='Frequency', ascending=False)



f, ax = plt.subplots(figsize=(15, 5))

plt.bar(np.array(range(len(plain_stats_train))) + 0.5, plain_stats_train['Frequency'].values)

plt.xticks(np.array(range(len(plain_stats_train))) + 0.5, plain_stats_train['Letter'].values)

plt.show()
for i in range(30):

    try:

        print(df_level_2.ciphertext[i], '\n')

    except KeyError:

        pass
plain_char_cntr = Counter(''.join(df_level_2['ciphertext'].values))

plain_stats_train = pd.DataFrame([[x[0], x[1]] for x in plain_char_cntr.items()], columns=['Letter', 'Frequency'])

plain_stats_train = plain_stats_train.sort_values(by='Frequency', ascending=False)



f, ax = plt.subplots(figsize=(15, 5))

plt.bar(np.array(range(len(plain_stats_train))) + 0.5, plain_stats_train['Frequency'].values)

plt.xticks(np.array(range(len(plain_stats_train))) + 0.5, plain_stats_train['Letter'].values)

plt.show()
for i in range(8):

    try:

        print(df_level_3.ciphertext[i], '\n')

    except KeyError:

        pass
plain_char_cntr = Counter(''.join(df_level_3['ciphertext'].values))

plain_stats_train = pd.DataFrame([[x[0], x[1]] for x in plain_char_cntr.items()], columns=['Letter', 'Frequency'])

plain_stats_train = plain_stats_train.sort_values(by='Frequency', ascending=False)



f, ax = plt.subplots(figsize=(15, 5))

plt.bar(np.array(range(len(plain_stats_train))) + 0.5, plain_stats_train['Frequency'].values)

plt.xticks(np.array(range(len(plain_stats_train))) + 0.5, plain_stats_train['Letter'].values)

plt.show()
for i in range(3):

    try:

        print(df_level_4.ciphertext[i], '\n')

    except KeyError:

        pass
plain_char_cntr = Counter(''.join(df_level_4['ciphertext'].values))

plain_stats_train = pd.DataFrame([[x[0], x[1]] for x in plain_char_cntr.items()], columns=['Letter', 'Frequency'])

plain_stats_train = plain_stats_train.sort_values(by='Frequency', ascending=False)



f, ax = plt.subplots(figsize=(15, 5))

plt.bar(np.array(range(len(plain_stats_train))) + 0.5, plain_stats_train['Frequency'].values)

plt.xticks(np.array(range(len(plain_stats_train))) + 0.5, plain_stats_train['Letter'].values)

plt.show()
plain_char_cntr = Counter(''.join(train_df['text'].values))

plain_stats_train = pd.DataFrame([[x[0], x[1]] for x in plain_char_cntr.items()], columns=['Letter', 'Frequency'])

plain_stats_train = plain_stats_train.sort_values(by='Frequency', ascending=False)
plain_char_test = Counter(''.join(df_level_1['ciphertext'].values))

plain_stats_test = pd.DataFrame([[x[0], x[1]] for x in plain_char_test.items()], columns=['Letter', 'Frequency'])

plain_stats_test = plain_stats_test.sort_values(by='Frequency', ascending=False)

plain_stats_test['Frequency'] -= 21130 # to remove the influence of random padding caracters
f, ax = plt.subplots(figsize=(15, 5))

plt.bar(np.array(range(len(plain_stats_test))) + 0.5, plain_stats_test['Frequency'].values)

plt.bar(np.array(range(len(plain_stats_train))) + 0.5, plain_stats_train['Frequency'].values//4, alpha=.5,color='green')

plt.xticks(np.array(range(len(plain_stats_test))) + 0.5, plain_stats_test['Letter'].values)

plt.show()
# first trick: use the length of some text pieces to understand the first cypher level

df_level_1.length.sort_values(ascending=False).head()



# we can see that 1 piece of text is of len 500, meaning that its original length is between 

# 401 and 500 (recall that every original piece of text is padded with random caracters to

# the next hundred)
df_level_1.length.describe([.999])
# and here is the corresponding text

df_level_1.loc[45272].ciphertext
# then we look in the training data to find the passage with the corresponding length

matching_pieces = train_df[(train_df.length>=401) & (train_df.length<=500)]

matching_pieces

# only three unciphered texts length are in the interval: let's print them
matching_pieces.text.values
print('Unciphered text:\n', train_df.loc[13862].text, '\n\nCiphered text (level 1):\n', 

      df_level_1.loc[45272].ciphertext)
# Let's do the same thing for a second piece of text now.

# With the same procedure, we get a second match:

print(train_df.loc[6938].text, '\n\n', df_level_1.loc[95019].ciphertext)