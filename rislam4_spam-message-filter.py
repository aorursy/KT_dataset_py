# Importing librqary
import numpy as np
import pandas as pd
# loading dataset
filename= "/kaggle/input/sms-data-labelled-spam-and-non-spam/SMSSpamCollection"
df = pd.read_csv(filename, header= None, names= ['Label', 'SMS'], sep= '\t')
df.head()
pd.set_option('max_colwidth', 5000)

df.head()
df.shape
df.Label.value_counts(normalize= True)
# Randomize the dataset
shuffled_set = df.sample(frac= 1, random_state= 1)

# Training and test dataset
training_set = shuffled_set[ : round(len(shuffled_set)*.8)]      # 1st 80%
test_set = shuffled_set[round(len(shuffled_set)*.8) : ]          # last 20%

# training and test dataset shape
print(training_set.shape)
print(test_set.shape)
# to set the datset index as 0,1,2,3........
training_set.reset_index(drop= True, inplace= True)  

test_set.reset_index(drop= True, inplace = True)    
training_set.head()
test_set.head()
training_set.Label.value_counts(normalize= True)
test_set.Label.value_counts(normalize= True)
def cleaned_sms(x):
    import re
    x = x.lower()                              # to lower case
    x = re.sub('[^\w\s]', '', x)               # remove character except a-z, A-z,1-9 and white space.
    return x

# adding this cleaned column
training_set['Cleaned_sms'] = training_set.SMS.apply(cleaned_sms)
training_set.head()
# Determine a set of all unique words.
all_words = []    

for row in training_set.Cleaned_sms:
    for word in row.split():
        all_words.append(word)
    
unique_words = set(all_words)
len(all_words)       # Number of all vocabularies
len(unique_words)    # Number of all unique vocabularies
# Printing first 10 words to as example
print(set(list(unique_words)[:10]))
# Creating a dictionary with the same length as training set.
unique_words_dic = {item : [0]*len(training_set) for item in unique_words}

# Counting the number of wrods for each row.
for index, sentence in enumerate(training_set.Cleaned_sms):
    sentence = sentence.split()
    for word in sentence:
        unique_words_dic[word][index] += 1  
training_set_words = pd.DataFrame(unique_words_dic)
training_set_words.head()
# Now, creating the final set.
training_set_final = pd.concat([training_set, training_set_words], axis= 1)
training_set_final.head(2)
# Training set with ham message only.
training_set_ham = training_set_final[training_set_final.Label == 'ham']

# Training set with spam message only.
training_set_spam = training_set_final[training_set_final.Label == 'spam']
# probability of spam 'p(spam)' and probabilty of ham 'p(ham)'

p_spam = len(training_set_spam) / len(training_set_final)
p_ham = len(training_set_ham) / len(training_set_final)

print('p_ham: ', p_ham)
print('p_spam: ', p_spam)
# number of total words in spam (Nspam) and in ham (Nham)
# In spam
length = 0
for row in training_set_spam.Cleaned_sms:
    split = row.split()         # splitting the text by space and which contains all words
    length = length + len(split)
total_words = length

n_spam = total_words

# In ham
length = 0
for row in training_set_ham.Cleaned_sms:
    split = row.split()         # splitting the text by space and which contains all words
    length = length + len(split)
total_words = length

n_ham = total_words

print('n_spam (number of total words in spam): ', n_spam)
print('n_ham (number of total words in ham): ', n_ham)
# total number of vocabulary in the whole training set

n_vocabulary = len(unique_words)
print('n_vocabulary (number of total words in dataset): ', n_vocabulary)
# Laplace smoothing
alpha = 1
print(alpha)
print('alpha: ',alpha, end= '\t\t\t')
print('n_vocabulary: ', n_vocabulary)
print('p_ham: ', p_ham, end= '\t')
print('p_spam: ', p_spam)
print('n_ham: ', n_ham, end= '\t\t\t')
print('n_spam: ', n_spam)
# Creating an empty dictionary for each word with value 0.
prob_word_given_ham = {}
prob_word_given_spam = {}

for word in unique_words:
    prob_word_given_ham[word] = 0
    prob_word_given_spam[word] = 0
# Function to determine P(Word|ham)
def prob_word_ham(word):
    n_w_ham = training_set_ham[word].sum()
    numerator = n_w_ham + alpha
    denominator = n_ham + (alpha * n_vocabulary)
    
    answer = numerator / denominator
    return answer

# Function to determine P(Word|spam)
def prob_word_spam(word):
    n_w_spam = training_set_spam[word].sum()
    numerator = n_w_spam + alpha
    denominator = n_spam + (alpha * n_vocabulary)
    
    answer = numerator / denominator
    return answer

# Applying the function and assigning the value into the dictionary
# For ham
for item in prob_word_given_ham:
    prob_word_given_ham[item] = prob_word_ham(item)

# For spam.
for item in prob_word_given_spam:
    prob_word_given_spam[item] = prob_word_spam(item)
# Example
prob_word_given_ham['slap']
# Example
prob_word_given_spam['slap']
import re
def classify(text):
    
    # Cleaning the text
    text = text.lower()
    text = re.sub('[^\w\s]', ' ', text)
    text = text.split()
    
    # Assigning the initial value
    p_spam_given_the_sms = p_spam
    p_ham_given_the_sms = p_ham
    
    # calculating the probability for each word.
    for word in text:
        if word in prob_word_given_ham:
            p_ham_given_the_sms *= prob_word_given_ham[word]
        if word in prob_word_given_spam:
            p_spam_given_the_sms *= prob_word_given_spam[word]
            
    print('p(ham|sms) = ', p_ham_given_the_sms, end='\t\t')
    print('p(spam|sms) = ', p_spam_given_the_sms)
    print('\n')
    
    # Condition
    if p_ham_given_the_sms > p_spam_given_the_sms:
        print('SMS/Message is NOT SPAM')
    elif p_ham_given_the_sms < p_spam_given_the_sms: 
        print('SMS/Message is SPAM')
    else:
        print('Need human help to classify.')
# Testing a spam message
classify('WINNER!! This is the secret code to unlock the money: C3421.')
classify("Hello, this is Rakib. Have you got any secret message about money from CEO today?")
import re
def classify_on_test(text):
    
    # Cleaning the text
    text = text.lower()
    text = re.sub('[^\w\s]', ' ', text)
    text = text.split()
    
    # Assigning the initial value
    p_spam_given_the_sms = p_spam
    p_ham_given_the_sms = p_ham
    
    # calculating the probability for each word.
    for word in text:
        if word in prob_word_given_ham:
            p_ham_given_the_sms *= prob_word_given_ham[word]
        if word in prob_word_given_spam:
            p_spam_given_the_sms *= prob_word_given_spam[word]
        
    # Condition
    if p_ham_given_the_sms > p_spam_given_the_sms:
        return('ham')
    elif p_ham_given_the_sms < p_spam_given_the_sms: 
        return('spam')
    else:
        return('Need humans.')
test_set.head()
# Testing on the test set and assigning a column with the return value
test_set['predicted'] = test_set['SMS'].apply(classify_on_test)
test_set.head()
test_set.predicted.value_counts()
test_set.Label.value_counts()
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.figure(figsize=(7,4), dpi= 95)


test_set.predicted.value_counts().plot.bar(color= 'blue', width= .22, label= 'Predicted', rot= 0)
test_set.Label.value_counts().plot.bar(color= 'green', width= .22, position= .1, label= 'Orignal', rot= 0)

plt.legend()
plt.show()
# Accuracy check
correct = 0
incorrect = 0
for label, predicted in zip(test_set.Label, test_set.predicted):
    if label == predicted:
        correct += 1
    else:
        incorrect += 1

accuracy = correct / (correct + incorrect)
print(accuracy)
