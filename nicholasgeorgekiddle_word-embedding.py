# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Code based on notebook from Nick Koprowicz: https://www.kaggle.com/nkoprowicz/a-simple-solution-using-only-word-counts/notebook



import pandas as pd

import numpy as np

import pprint as pp

from gensim.models import Word2Vec

# CountVectorizer will help calculate word counts

from sklearn.feature_extraction.text import CountVectorizer

# Optionally, use the sklearn logistic regression model instead of the homebrewed one

from sklearn.linear_model import LogisticRegression

# Import the string dictionary that we'll use to remove punctuation

import string

# Make training/test split

from sklearn.model_selection import train_test_split







class LogisticRegressionCustom():



    def __init__(self, H = 0.00001, batch_size = -1, iters = 200, verbose = False):

        self.H = H

        self.batch_size = batch_size

        self.iters = iters

        self.verbose = False



    def score(self, X, Y):

        # Calculate input values for sigmoid function

        # (if negative, then predict 0)

        #print(X, W)

        predictors = np.matmul(X, self.W)

        #print("X?", np.where(X > 0))

        #print("W?", np.where(W > 0))



        #print(predictors)



        # Change predictors to 0 or 1 to predict events. (Iif W'x >= 0,

        # then predict 1, that is the drawing is a 9 not a 4)

        # Here we use standard decision rule

        #print(np.where(pred

        # ictors > 0))

        predictors = np.where(predictors < 0, 0, 1)



        # Calculate the number of falses (both false positives and false negatives)

        falses = np.sum(np.abs(Y - predictors.T))



        # Return the accuracy (the fraction of correct trials divided by total trials)

        1 - (falses / (len(Y)))

        return 1 - (falses / (len(Y)))



    def predict(self, X):

        return np.reciprocal((np.exp(-1 * np.matmul(X, self.W)) + 1))



    def fit(self, X, Y):

        X = np.nan_to_num(X)

        Y = np.nan_to_num(Y)

        #print(batch_size)



        # Initialize W (weights) to be zero vector corresponding to columns in attribute mtx

        W = np.zeros(X.shape[1])

        batch_num = 0

        # initialize gradient

        grad = float('inf')



        if self.batch_size == -1:

            self.batch_size = X.shape[0]



        # Also need condition to prevent exceeding number of training attributes

        #while (grad > eps):

        for j in range(int(self.iters * (X.shape[0] / self.batch_size))):



            # Get the slice of attribute matrix corresponding to batch

            batch = X[batch_num*self.batch_size:(batch_num+1)*self.batch_size,:]



            # Get vector of etimated outputs using old weight

            # Since batch has row vectors of attributes, multiply by column vector W

            # (Versus standard practice of multiplying W transpose by a column vector or attributes)

            estimated = np.reciprocal((np.exp(-1 * np.matmul(batch, W)) + 1))



            # Get column vector of differences between estimated and actual

            actual = Y[batch_num*self.batch_size:(batch_num+1)*self.batch_size]

            #actual = Y

            diffs = (estimated.T - actual)

            #print(np.linalg.norm(diffs))



            # Use diffs to construct diagonal matrix, multiply that by batch, then sum along the rows

            grad = np.sum(np.multiply(diffs, batch.T), axis = 1)

            #pp.pprint(diffs)

            #pp.pprint(np.diag(diffs))



            # Adjust weights

            #print(np.linalg.norm(W), np.linalg.norm(H*grad))

            W = W - (self.H * grad)

            # replace grad with the magnitude (L2 norm) of the vector

            grad = np.linalg.norm(grad)

            #print(grad, np.linalg.norm(W))



            # Increment batch number

            batch_num += 1

            # Check if the next batch will exceed the number of training trials

            if ((batch_num+1)*self.batch_size - 1 > X.shape[0]):

                batch_num = 0

            

            self.W = W

            if self.verbose is True:

                print(self.score(X, Y))



        if self.verbose is True:

            print("Found weights are:")

            pp.pprint(W)





def apply_model(model, x):

    #print(x)

    sum = np.zeros(50)

    count = 0

    for i in range(len(x)):

        if x[i] in model.wv.vocab:

            sum += model.wv[x[i]]

            count += 1

    

    #print(sum.shape)

    #print(count)

    #if count == 0:

    #    print(x)



    return np.divide(sum, count)



def is_positive(x):

    if x == "positive":

        return 1

    else:

        return 0



def is_negative(x):

    if x == "negative":

        return 1

    else:

        return 0



def get_logistic(full_train):

    # build word2vec model

    #print(((pos_train['text'])[:20]).values.split(' '))

    text = full_train.apply(lambda x: x['text'].split(), axis = 1)





    #text = pos_train['text'].values

    #split_text = np.where(text, text.split())

    print("Training word embedding...")

    model = Word2Vec(text.values,

                     min_count=2,

                     window=2,

                     size=50,

                     sample=6e-5, 

                     alpha=0.03,

                     min_alpha=0.0007,

                     negative=20)



    applied_model = text.apply(lambda x: apply_model(model, x))



    X = np.array(list(applied_model.values))

    print("Done with that!\n")



    Ypos = full_train['sentiment'].apply(lambda x: is_positive(x))

    Ypos = Ypos.values.ravel()



    Yneg = full_train['sentiment'].apply(lambda x: is_negative(x))

    Yneg = Yneg.values.ravel()



    X = np.nan_to_num(X)

    Ypos = np.nan_to_num(Ypos)

    Yneg = np.nan_to_num(Yneg)



    print("Fitting logistic regression models...")

    # all parameters not specified are set to their defaults

    posLogisticRegr = LogisticRegression()

    posLogisticRegr.fit(X, Ypos)



    negLogisticRegr = LogisticRegression()

    negLogisticRegr.fit(X, Yneg)

    # Use score method to get accuracy of model

    pos_score = posLogisticRegr.score(X, Ypos)

    neg_score = negLogisticRegr.score(X, Yneg)

    print("Done!\n")

    print("Positive Score: " + str(pos_score))

    print("Negative Score: " + str(neg_score))



    

    return (posLogisticRegr, negLogisticRegr, model)



def calculate_selected_text(df_row, positiveModel, negativeModel, word2vecModel, tol = 0):

        tweet = df_row['text']

        sentiment = df_row['sentiment']



        if(sentiment == 'neutral'):

            return tweet

        elif(sentiment == 'positive'):

            logisticToUse = positiveModel # Calculate word weights using the pos_words dictionary

        elif(sentiment == 'negative'):

            logisticToUse = negativeModel # Calculate word weights using the neg_words dictionary



        words = tweet.split()

        words_len = len(words)

        subsets = [words[i:j+1] for i in range(words_len) for j in range(i,words_len)]



        score = 0

        selection_str = '' # This will be our choice

        lst = sorted(subsets, key = len) # Sort candidates by length



        failed_subsets = 0

        for i in range(len(subsets)):



            sum = np.zeros(50)

            count = 0

            for j in range(len(lst[i])):

                if (lst[i][j]) in word2vecModel.wv.vocab:

                    sum += word2vecModel.wv[lst[i][j]]

                    count += 1

            if count > 0:

                new_score = logisticToUse.predict([np.divide(sum, count)])

            else:

                new_score = 0

                failed_subsets += 1

            

            # If the sum is greater than the score, update our current selection

            if(new_score > score + tol):

                score = new_score

                selection_str = lst[i]

                #tol = tol*5 # Increase the tolerance a bit each time we choose a selection

        # print(str(failed_subsets) + " / " + str(len(subsets)) + str(" subsets failed"))



        # If we didn't find good substrings, return the whole text

        if(len(selection_str) == 0):

            selection_str = words



        return ' '.join(selection_str)



def jaccard(str1, str2):

    a = set(str1.lower().split())

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))



def which_longer(truth, prediction):

    truth_set = set(truth.lower().split())

    pred_set = set(prediction.lower().split())

    intersect = truth_set.intersection(pred_set)



    return float(len(truth_set - intersect) + len(pred_set - intersect))



def load_data():

    # Import datasets

    train = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

    test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

    sample = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')

    

    # The row with index 13133 has NaN text, so remove it from the dataset

    train[train['text'].isna()]



    train.drop(314, inplace = True)



    # Make all the text lowercase - casing doesn't matter when

    # we choose our selected text.

    train['text'] = train['text'].apply(lambda x: x.lower())

    test['text'] = test['text'].apply(lambda x: x.lower())

    return train, test, sample





def main():

    tol = 0.001



    train, test, sample = load_data()

    # Use K-fold validation

    K = 5

    indexes = np.arange(train.shape[0])

    np.random.shuffle(indexes)

    bestScore = 0

    bestModel = None



    for group in range(K):

        group_start = int(group * (indexes.shape[0] / K ))

        group_end = int((group + 1) * (indexes.shape[0] / K ))

        #print("Group size: " + str(group_end - group_start))

        X_train_part_1 = train.iloc[indexes[:group_start]] # from 0 to start of group

        X_train_part_2 = train.iloc[indexes[group_end:]] # from end of group to end of data

        X_train = pd.concat([X_train_part_1, X_train_part_2])

        #print(X_train_part_1.shape, X_train_part_2.shape, X_train.shape)

        X_val = train.iloc[group_start:group_end]

        

        posLogisticModel, negLogisticModel, word2vecModel = get_logistic(X_train)



        pd.options.mode.chained_assignment = None

 

        X_val['predicted_selection'] = ''



        for index, row in X_val.iterrows():

            selected_text = calculate_selected_text(row, posLogisticModel, negLogisticModel, word2vecModel)

            X_val.loc[X_val['textID'] == row['textID'], ['predicted_selection']] = selected_text



        X_val['which_longer'] = X_val.apply(lambda x: which_longer(x['selected_text'], x['predicted_selection']), axis = 1)

        X_val['jaccard'] = X_val.apply(lambda x: jaccard(x['selected_text'], x['predicted_selection']), axis = 1)

        print('The jaccard score for the validation set is:', np.mean(X_val['jaccard']))

        print('The selected text for negative is on average {} words different'.format(str(np.mean((X_val[X_val['sentiment'] == 'negative'])['which_longer']))))

        print('The selected text for positive is on average {} words different'.format(str(np.mean((X_val[X_val['sentiment'] == 'positive'])['which_longer']))))

        print('The selected text for neutral is on average {} words different'.format(str(np.mean((X_val[X_val['sentiment'] == 'neutral'])['which_longer']))))



        if np.mean(X_val['jaccard']) > bestScore:

            bestModel = (posLogisticModel, negLogisticModel, word2vecModel)



    (posLogisticModel, negLogisticModel, word2vecModel) = bestModel

    

    for index, row in test.iterrows():

        selected_text = calculate_selected_text(row, posLogisticModel, negLogisticModel, word2vecModel)

        sample.loc[sample['textID'] == row['textID'], ['selected_text']] = selected_text





    sample.to_csv('submission.csv', index = False)





if __name__ == "__main__":

  main()
