# data samples for spam

data_spam = [

    'Free cell phone',

    'Free gift DVD',

]



# data samples for non-spam

data_non_spam = [

    'my phone number is 86413322',

    'come to my office now',

    'feel free to contact me',

]



# test samples

test_sample = [

    'Free DVD',

]
# creates a vocabulary of unique words from the given arrays

# numbers are not treated as a word (thus, ignored)

def create_vocab(*samples):

    vocabulary = set()



    for sample in samples:

        for document in sample:

            for word in document.split():

                if not word.isdigit():

                    vocabulary.add(word.lower())



    return vocabulary



# counts all the target words in a given sample

# if target is None, all the words in the sample are counted

# otherwise, all the target words are counted

def get_word_count(sample, target=None):

    accumulator = 0



    for document in sample:

        for word in document.split():

            if target is None:

                accumulator += 1

            elif word.lower() == target:

                accumulator += 1



    return accumulator
# collate samples vocabulary (numbers are not treated as unique words)

vocabulary = create_vocab(data_spam, data_non_spam)

n_vocab = len(vocabulary)
# count all the data samples

n_data_spam = len(data_spam)

n_data_non_spam = len(data_non_spam)

n_data = n_data_spam + n_data_non_spam



# compute prior probability of the classes

prior_spam = n_data_spam / n_data

prior_non_spam = n_data_non_spam / n_data



# count the total number of words for each classes

n_data_spam_words = get_word_count(data_spam)

n_data_non_spam_words = get_word_count(data_non_spam)



# compute likelihood of each word in the vocabulary in each of the classes (spam or non-spam)

likelihood_spam = { word: (get_word_count(data_spam, target=word) + 1) / (n_data_spam_words + n_vocab) for word in vocabulary }

likelihood_non_spam = { word: (get_word_count(data_non_spam, target=word) + 1) / (n_data_non_spam_words + n_vocab) for word in vocabulary }
def classify(text):

    spam = 1

    non_spam = 1



    for word in text.split():

        # transform word to lowercase

        word = word.lower()



        # get the likelihood of the current word or 0 if word is not in the dictionary (it means the current word is unknown)

        spam *= likelihood_spam.get(word, 1/(n_data_spam_words + n_vocab))

        non_spam *= likelihood_non_spam.get(word, 1/(n_data_non_spam_words + n_vocab))



    # add in the prior probability of each of the classes

    spam *= prior_spam

    non_spam *= prior_non_spam



    # print out results

    if spam > non_spam:

        print('Input Text: "%s"' % text)

        print('Result: Spam')

    else:

        print('Input Text: "%s"' % text)

        print('Result: Non-spam')
classify('Free DVD')
classify('Go to class')