!pip install pytorch-pretrained-bert
import torch

from pytorch_pretrained_bert import BertTokenizer



# Load pre-trained model tokenizer (vocabulary)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
with open("vocabulary.txt", 'w') as f:

    

    # For each token...

    for token in tokenizer.vocab.keys():

        

        # Write it out and escape any unicode characters.            

        f.write(token + '\n')
one_chars = []

one_chars_hashes = []



# For each token in the vocabulary...

for token in tokenizer.vocab.keys():

    

    # Record any single-character tokens.

    if len(token) == 1:

        one_chars.append(token)

    

    # Record single-character tokens preceded by the two hashes.    

    elif len(token) == 3 and token[0:2] == '##':

        one_chars_hashes.append(token)
one_chars = []

one_chars_hashes = []



# For each token in the vocabulary...

for token in tokenizer.vocab.keys():

    

    # Record any single-character tokens.

    if len(token) == 1:

        one_chars.append(token)

    

    # Record single-character tokens preceded by the two hashes.    

    elif len(token) == 3 and token[0:2] == '##':

        one_chars_hashes.append(token)

print('Number of single character tokens:', len(one_chars), '\n')



# Print all of the single characters, 40 per row.



# For every batch of 40 tokens...

for i in range(0, len(one_chars), 40):

    

    # Limit the end index so we don't go past the end of the list.

    end = min(i + 40, len(one_chars) + 1)

    

    # Print out the tokens, separated by a space.

    print(' '.join(one_chars[i:end]))
print('Number of single character tokens with hashes:', len(one_chars_hashes), '\n')



# Print all of the single characters, 40 per row.



# Strip the hash marks, since they just clutter the display.

tokens = [token.replace('##', '') for token in one_chars_hashes]



# For every batch of 40 tokens...

for i in range(0, len(tokens), 40):

    

    # Limit the end index so we don't go past the end of the list.

    end = min(i + 40, len(tokens) + 1)

    

    # Print out the tokens, separated by a space.

    print(' '.join(tokens[i:end]))
print('Are the two sets identical?', set(one_chars) == set(tokens))
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



sns.set(style='darkgrid')



# Increase the plot size and font size.

sns.set(font_scale=1.5)

plt.rcParams["figure.figsize"] = (10,5)



# Measure the length of every token in the vocab.

token_lengths = [len(token) for token in tokenizer.vocab.keys()]



# Plot the number of tokens of each length.

sns.countplot(token_lengths)

plt.title('Vocab Token Lengths')

plt.xlabel('Token Length')

plt.ylabel('# of Tokens')



print('Maximum token length:', max(token_lengths))
num_subwords = 0



subword_lengths = []



# For each token in the vocabulary...

for token in tokenizer.vocab.keys():

    

    # If it's a subword...

    if len(token) >= 2 and token[0:2] == '##':

        

        # Tally all subwords

        num_subwords += 1



        # Measure the sub word length (without the hashes)

        length = len(token) - 2



        # Record the lengths.        

        subword_lengths.append(length)

vocab_size = len(tokenizer.vocab.keys())



print('Number of subwords: {:,} of {:,}'.format(num_subwords, vocab_size))



# Calculate the percentage of words that are '##' subwords.

prcnt = float(num_subwords) / vocab_size * 100.0



print('%.1f%%' % prcnt)
sns.countplot(subword_lengths)

plt.title('Subword Token Lengths (w/o "##")')

plt.xlabel('Subword Length')

plt.ylabel('# of ## Subwords')
print('beginning' in tokenizer.vocab) # Right

print('begining' in tokenizer.vocab) # Wrong

print('separate' in tokenizer.vocab) # Right

print('seperate' in tokenizer.vocab) # Wrong

print("can't" in tokenizer.vocab)

print("cant" in tokenizer.vocab)

print('goverment' in tokenizer.vocab) # Wrong

print('government' in tokenizer.vocab) # Right

print('mispelled' in tokenizer.vocab) # Wrong

print('misspelled' in tokenizer.vocab) # Right
# For each token in the vocabulary...

for token in tokenizer.vocab.keys():

    

    # If it's a subword...

    if len(token) >= 2 and token[0:2] == '##':

        if not token[2:] in tokenizer.vocab:

            print('Did not find a token for', token[2:])

            break
print('##ly' in tokenizer.vocab)

print('ly' in tokenizer.vocab)
!pip install wget
import wget

import random 



print('Beginning file download with wget module')



url = 'http://www.gutenberg.org/files/3201/files/NAMES.TXT'

wget.download(url, 'first-names.txt')

# Read them in.

with open('first-names.txt', 'rb') as f:

    names_encoded = f.readlines()



names = []



# Decode the names, convert to lowercase, and strip newlines.

for name in names_encoded:

    try:

        names.append(name.rstrip().lower().decode('utf-8'))

    except:

        continue



print('Number of names: {:,}'.format(len(names)))

print('Example:', random.choice(names))

num_names = 0



# For each name in our list...

for name in names:



    # If it's in the vocab...

    if name in tokenizer.vocab:

        # Tally it.

        num_names += 1



print('{:,} names in the vocabulary'.format(num_names))
# Count how many numbers are in the vocabulary.

count = 0



# For each token in the vocabulary...

for token in tokenizer.vocab.keys():



    # Tally if it's a number.

    if token.isdigit():

        count += 1

        

        # Any numbers >= 10,000?

        if len(token) > 4:

            print(token)



print('Vocab includes {:,} numbers.'.format(count))
# Count how many dates between 1600 and 2021 are included.

count = 0 

for i in range(1600, 2021):

    if str(i) in tokenizer.vocab:

        count += 1



print('Vocab includes {:,} of 421 dates from 1600 - 2021'.format(count))