import pandas as pd

import numpy as np 

from sklearn.model_selection import train_test_split

import re
def load_data():

    df = pd.read_csv('../input/spam-filter/emails.csv', encoding='latin-1')

    df_for_tests = df.head()

    

    idx = np.arange(df.shape[0])

    np.random.shuffle(idx)



    train_set_size = int(df.shape[0] * 0.8)



    train_set = df.loc[idx[:train_set_size]]

    test_set = df.loc[idx[train_set_size:]]

    

    return train_set, test_set, df_for_tests
train_set, test_set, df_for_tests = load_data()
# Clean the data



def clean_data(message):

    

    """ 

    Returns string which consists of message words

    

    Argument:

    message -- message from dataset; 

        type(message) -> <class 'str'>

    

    Returns:

    result -- cleaned message, which contains only letters a-z, and numbers 0-9, with only one space between words;

        type(clean_data(message)) -> <class 'str'>

    

    """

    

    ### START CODE HERE ###

    return " ".join("".join(re.findall("[a-zA-Z0-9_ ]", message)).lower().split())

    

    ### END CODE HERE ###
# Preparation data for model



def prep_for_model(train_set, test_set):

    

    """ 

    Returns arrays of train/test features(words) and train/test targets(labels)

    

    Arguments:

    train_set -- train dataset, which consists of train messages and labels; 

        type(train_set) -> pandas.core.frame.DataFrame

    test_set -- test dataset, which consists of test messages and labels; 

        type(train_set) -> pandas.core.frame.DataFrame

    

    Returns:

    train_set_x -- array which contains lists of words of each cleaned train message; 

        (type(train_set_x) ->numpy.ndarray[list[str]], train_set_x.shape = (num_messages,))

    train_set_y -- array of train labels (names of classes), 

        (type(train_set_y) -> numpy.ndarray, train_set_y.shape = (num_messages,))

    test_set_x -- array which contains lists of words of each cleaned test message;

        (type(test_set_x) numpy.ndarray[list[str]], test_set_x.shape = (num_messages,)

    test_set_y -- array of test labels (names of classes), 

        (type(test_set_y) -> numpy.ndarray, test_set_y.shape = (num_messages,))

    

    """

    

    ### START CODE HERE ###

    train_set_x = []

    for mesg in train_set["text"]:

        train_set_x.append(clean_data(mesg).split())

    train_set_x = np.array(train_set_x)

    # train_set_x = np.array([clean_data(mes).split() for mes in train_set["v2"]])



    train_set_y = []

    for mesg in train_set["spam"]:

        train_set_y.append(mesg)

    train_set_y = np.array(train_set_y)

    # train_set_y = train_set["v1"].to_numpy()



    test_set_x = []

    for mesg in test_set["text"]:

        test_set_x.append(clean_data(mesg).split())

    test_set_x = np.array(test_set_x)

    # test_set_x = np.array([clean_data(mes).split() for mes in test_set["v2"]])



    test_set_y = []

    for mesg in test_set["spam"]:

        test_set_y.append(mesg)

    test_set_y = np.array(test_set_y)

    # test_set_y = test_set["v1"].to_numpy()

    

    ### END CODE HERE ###

    

    return train_set_x, train_set_y, test_set_x, test_set_y



train_set_x, train_set_y, test_set_x, test_set_y = prep_for_model(train_set, test_set)
# Check words in categories



def categories_words(x_train, y_train):

    

    """

    Returns arrays of features(words) in each category and in both categories

    

    Arguments:

    x_train -- array which contains lists of words of each cleaned train message; 

        (type(x_train) -> numpy.ndarray[list[str]], x_train.shape = (num_messages,))

    

    Returns:

    all_words_list -- array of all words in both categories;

        (type(all_words_list) -> numpy.ndarray[str], all_words_list.shape = (num_words,))

    ham_words_list -- array of words in 'ham' class;

        (type(ham_words_list) -> numpy.ndarray[str], ham_words_list.shape = (num_words,))

    spam_words_list -- array of words in 'spam' class;

        (type(spam_words_list) -> numpy.ndarray[str], spam_words_list.shape = (num_words,))        

    """

    all_words_list = []

    ham_words_list = []

    spam_words_list = []

    

    ### START CODE HERE ###

    for i in range(x_train.shape[0]):

        all_words_list += x_train[i]

        if y_train[i] == 0:

            ham_words_list += x_train[i]

        else:

            spam_words_list += x_train[i]

    

    ### END CODE HERE ###

    

    return all_words_list, ham_words_list, spam_words_list



# all_words_list_a1, ham_words_list_a1, spam_words_list_a1 = categories_words(a1, a2)
class Naive_Bayes(object):

    """

    Parameters:

    -----------

    alpha: int

        The smoothing coeficient.

    """

    def __init__(self, alpha):

        self.alpha = alpha

        

        self.train_set_x = None

        self.train_set_y = None

        

        self.all_words_list = []

        self.ham_words_list = []

        self.spam_words_list = []

    

    def fit(self, train_set_x, train_set_y):

        

        # Generate all_words_list, ham_words_list, spam_words_list using function 'categories_words'; 

        # Calculate probability of each word in both categories

        ### START CODE HERE ### 

        self.train_set_x = train_set_x

        self.train_set_y = train_set_y

        self.all_words_list, self.ham_words_list, self.spam_words_list = categories_words(self.train_set_x, self.train_set_y)



        self.prob = {w: ((self.alpha + self.spam_words_list.count(w)) / (len(self.train_set_y) * self.alpha + len(self.spam_words_list)),

				(self.alpha + self.ham_words_list.count(w)) / (len(self.train_set_y) * self.alpha + len(self.ham_words_list)))

				for w in self.all_words_list}

        

        ### END CODE HERE ### 

        

    def predict(self, test_set_x):

        

        # Return list of predicted labels for test set; type(prediction) -> list, len(prediction) = len(test_set_y)

        ### START CODE HERE ###

        y_pred = []

        for mesg in test_set_x:

            spam = 0

            ham = 0            

            for word in mesg:

                if word in self.all_words_list:

                    # print(np.log(prob[word][0]))

                    spam += np.log(self.prob[word][0])

                    ham += np.log(self.prob[word][1])

                    # print(np.log(prob[word][1]))

            y_pred.append(1 if spam > ham else 0)

    

            

        ### END CODE HERE ### 

        return y_pred



        # Check words in categories

a = 1

model = Naive_Bayes(alpha=a)

model.fit(train_set_x, train_set_y)

y_predictions = model.predict(test_set_x)



actual = list(test_set_y)

accuracy = (y_predictions == test_set_y).mean()

print(accuracy)