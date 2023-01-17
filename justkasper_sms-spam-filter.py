import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        file_path = os.path.join(dirname, filename)
import pandas as pd



spam_collection = pd.read_csv(file_path, sep='\t', header=None, names=['Label', 'SMS']) #data is separated with 'tabs' so we include 'sep' parameter
print(spam_collection.info(),'\n')

print(spam_collection.describe(),'\n')

print(spam_collection.head())
#spam/ham percentage



spam_share = spam_collection['Label'].value_counts(normalize=True)

print(spam_share * 100)
randomized = spam_collection.sample(frac=1, random_state=1) #using 'random_state' parameter to set a random seed and to be able to reproduce/control the randomization
#finding the split-point



split_point = int(len(randomized) * .2)



test = randomized[:split_point].reset_index(drop=True)

train = randomized[split_point:].reset_index(drop=True)
#checking the spit results



print('Test:',test.shape,'\n',test['Label'].value_counts(normalize=True) * 100,'\n\nTrain:',train.shape,'\n',train['Label'].value_counts(normalize=True) * 100)
train['SMS'] = train['SMS'].str.replace('\W',' ').str.lower()



#check

print(train.head(0))
#splitting messages



train['SMS'] = train['SMS'].str.split()
#getting all unique words in one list



vocabulary = []



for row in train['SMS']:

    for word in row:

        if not word in vocabulary:

            vocabulary.append(word)



#voc = list(set(vocabulary))
#check what we have here



print(len(vocabulary))
#making dict with unique words and length of SMS column and then filling the 'table'



word_counts_per_sms = {unique_word: [0] * len(train['SMS']) for unique_word in vocabulary}



for index, sms in enumerate(train['SMS']):

    for word in sms:

        word_counts_per_sms[word][index] += 1
#transforming dict into dataframe and adding columns from old training dataframe



transformed_train = pd.DataFrame(word_counts_per_sms)

transformed_train = pd.concat([train, transformed_train], axis=1)
#quick check



transformed_train.head()
#isolating spam and ham messages first



ham = transformed_train[transformed_train['Label'] == 'ham']

spam = transformed_train[transformed_train['Label'] == 'spam']
#calculating probabilites



p = transformed_train['Label'].value_counts()

p_spam = p['spam']

p_ham = p['ham']
#number of words in all 'spam' messages and in all 'ham'' messages



transformed_train['n_words'] = transformed_train['SMS'].apply(len) 



n_spam = transformed_train[transformed_train['Label'] == 'spam']['n_words'].sum()

n_ham = transformed_train[transformed_train['Label'] == 'ham']['n_words'].sum()



n_vocabulary = len(vocabulary)
#laplace smoothing



laplace = 1
#counting word probabilities for 'spam' and 'ham' messages



dict_spam = {x : 0 for x in vocabulary}

dict_ham = {x : 0 for x in vocabulary}



for word in dict_spam:

    dict_spam[word] = (laplace + spam[word].sum()) / (n_spam + laplace * n_vocabulary)

    

for word in dict_ham:

    dict_ham[word] = (laplace + ham[word].sum()) / (n_ham + laplace * n_vocabulary)    
import re



def classify(message:str):

    '''

    labels message with "Spam" or "Ham"

    '''

    message = re.sub('\W', ' ', message)

    message = message.lower()

    message = message.split()



    p_spam_given_message = p_spam

    p_ham_given_message = p_ham

    for word in message:

        if word in dict_spam:

            p_spam_given_message *= dict_spam[word]

        if word in dict_ham:

            p_ham_given_message *= dict_ham[word]

    

    print('P(Spam|message):', p_spam_given_message)

    print('P(Ham |message):', p_ham_given_message)



    if p_ham_given_message > p_spam_given_message:

        print('Label: Ham')

    elif p_ham_given_message < p_spam_given_message:

        print('Label: Spam')

    else:

        print('Equal proabilities, have a human classify this!')
classify('This is the secret code to wish you luck, Tom')
classify("Sounds good, Tom, i'll be in a winning tram in an hour")
def classify_test(message:str):

    '''

    labels message with "Spam" or "Ham"

    '''

    message = re.sub('\W', ' ', message)

    message = message.lower()

    message = message.split()



    p_spam_given_message = p_spam

    p_ham_given_message = p_ham

    for word in message:

        if word in dict_spam:

            p_spam_given_message *= dict_spam[word]

        if word in dict_ham:

            p_ham_given_message *= dict_ham[word]

    

    if p_ham_given_message > p_spam_given_message:

        return 'ham'

    elif p_spam_given_message > p_ham_given_message:

        return 'spam'

    else:

        return 'needs human classification'    
#implementing into test dataframe



test['predicted'] = test['SMS'].apply(classify_test)

test.head()
#calculating accuracy



correct = test['Label'] == test['predicted']



accuracy = sum(correct) / len(test)



print('Correct:', correct.sum())

print('Incorrect:', len(test) - sum(correct))

print('Accuracy: {:.2%}'.format(accuracy)) 