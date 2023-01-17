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
import pandas as pd

import numpy as np

import math



# CountVectorizer will help calculate word counts

from sklearn.feature_extraction.text import CountVectorizer

# Import the string dictionary that we'll use to remove punctuation

import string

# Make training/test split

from sklearn.model_selection import train_test_split





def build_vocab(cv, count_df, train, threshold):

    neg_count_df, pos_count_df, neutral_count_df = count_df

    neg_train, pos_train, neutral_train = train



    # Create dictionaries of the words within each sentiment group, where the values are the proportions of tweets that

    # contain those words



    pos_words = {}

    neutral_words = {}

    neg_words = {}



    for k in cv.get_feature_names():

        pos = pos_count_df[k].sum()

        neutral = neutral_count_df[k].sum()

        neg = neg_count_df[k].sum()



        pos_words[k] = pos/pos_train.shape[0]

        neutral_words[k] = neutral/neutral_train.shape[0]

        neg_words[k] = neg/neg_train.shape[0]



    # We need to account for the fact that there will be a lot of words used in tweets of every sentiment.

    # Therefore, we reassign the values in the dictionary by subtracting the proportion of tweets in the other

    # sentiments that use that word.



    neg_words_adj = {}

    pos_words_adj = {}

    neutral_words_adj = {}



    for key, value in neg_words.items():

        if(neutral_words[key] <= threshold and pos_words[key] <= threshold):

            neg_words_adj[key] = math.sqrt(np.abs(neg_words[key]))

        else:

            neg_words_adj[key] = neg_words[key] - (neutral_words[key] + pos_words[key])



    for key, value in pos_words.items():

        if(neutral_words[key] <= threshold and neg_words[key] <= threshold):

            pos_words_adj[key] = math.sqrt(np.abs(pos_words[key]))

        else:

            pos_words_adj[key] = pos_words[key] - (neutral_words[key] + neg_words[key])





    for key, value in neutral_words.items():

        if(pos_words[key] == threshold and neg_words[key] == threshold):

            neutral_words_adj[key] = math.sqrt(np.abs(neutral_words[key]))

        else:

            neutral_words_adj[key] = neutral_words[key] - (neg_words[key] + pos_words[key])



    return (neg_words_adj, pos_words_adj, neutral_words_adj)



def calculate_selected_text(pos_words_adj, neg_words_adj, df_row, tol = 0):

        tweet = df_row['text']

        sentiment = df_row['sentiment']



        if(sentiment == 'neutral'):

            return tweet

        elif(sentiment == 'positive'):

            dict_to_use = pos_words_adj # Calculate word weights using the pos_words dictionary

        elif(sentiment == 'negative'):

            dict_to_use = neg_words_adj # Calculate word weights using the neg_words dictionary



        words = tweet.split()

        words_len = len(words)

        subsets = [words[i:j+1] for i in range(words_len) for j in range(i,words_len)]



        score = 0

        selection_str = '' # This will be our choice

        lst = sorted(subsets, key = len) # Sort candidates by length



        for i in range(len(subsets)):



            new_sum = 0 # Sum for the current substring



            # Calculate the sum of weights for each word in the substring

            for p in range(len(lst[i])):

                if(lst[i][p].translate(str.maketrans('','',string.punctuation)) in dict_to_use.keys()):

                    new_sum += dict_to_use[lst[i][p].translate(str.maketrans('','',string.punctuation))]



            # If the sum is greater than the score, update our current selection

            if(new_sum > score + tol):

                score = new_sum

                selection_str = lst[i]

                #tol = tol*5 # Increase the tolerance a bit each time we choose a selection



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



    return float(len(pred_set) - len(truth_set))



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

    # tol = 0.001

    tol = 0.0009

    train, test, sample = load_data()

    # Modify code here to implent k-fold x validation

    K = 10

    jaccard_scores = []

    vocab_weights = []

    indexes = np.arange(train.shape[0])

    np.random.shuffle(indexes)

    print("Indexes shape: " + str(indexes.shape))

    print("Train shape: " + str(train.shape))



    for group in range(K):

        group_start = int(group * (indexes.shape[0] / K ))

        group_end = int((group + 1) * (indexes.shape[0] / K ))

        #print("Group size: " + str(group_end - group_start))

        X_train_part_1 = train.iloc[indexes[:group_start]] # from 0 to start of group

        X_train_part_2 = train.iloc[indexes[group_end:]] # from end of group to end of data

        X_train = pd.concat([X_train_part_1, X_train_part_2])

        #print(X_train_part_1.shape, X_train_part_2.shape, X_train.shape)

        X_val = train.iloc[group_start:group_end]

        

        #X_train, X_val = train_test_split(train, train_size = 0.80, random_state = 0)



        pos_train = X_train[X_train['sentiment'] == 'positive']

        neutral_train = X_train[X_train['sentiment'] == 'neutral']

        neg_train = X_train[X_train['sentiment'] == 'negative']

        

        # Use CountVectorizer to get the word counts within each dataset

        cv = CountVectorizer(max_df=0.95, min_df=4, max_features=3000, stop_words='english')



        # X_train_cv = cv.fit_transform(X_train['text'])

        cv.fit_transform(X_train['text'])



        X_pos = cv.transform(pos_train['text'])

        X_neutral = cv.transform(neutral_train['text'])

        X_neg = cv.transform(neg_train['text'])



        pos_count_df = pd.DataFrame(X_pos.toarray(), columns=cv.get_feature_names())

        neutral_count_df = pd.DataFrame(X_neutral.toarray(), columns=cv.get_feature_names())

        neg_count_df = pd.DataFrame(X_neg.toarray(), columns=cv.get_feature_names())



        count_df = (neg_count_df, pos_count_df, neutral_count_df)

        train_df = (neg_train, pos_train, neutral_train)



        threshold = 0

        neg_words_adj, pos_words_adj, neutral_words_adj = build_vocab(cv, count_df, train_df, threshold)

        vocab_weights.append({

            'neg': neg_words_adj,

            'pos': pos_words_adj,

            'neu': neutral_words_adj

        })

        pd.options.mode.chained_assignment = None



        X_val['predicted_selection'] = ''



        for index, row in X_val.iterrows():

            selected_text = calculate_selected_text(pos_words_adj, neg_words_adj, row, tol)

            X_val.loc[X_val['textID'] == row['textID'], ['predicted_selection']] = selected_text



        X_val['which_longer'] = X_val.apply(lambda x: which_longer(x['selected_text'], x['predicted_selection']), axis = 1)

        X_val['jaccard'] = X_val.apply(lambda x: jaccard(x['selected_text'], x['predicted_selection']), axis = 1)

        jaccard_scores.append(np.mean(X_val['jaccard']))

        print('-------------------------------- K = ' + str(group + 1) + ' ---------------------------------------------')

        print('The jaccard score for the validation set is:', np.mean(X_val['jaccard']))

        print('The selected text for negative is on average {} words smaller'.format(str(np.mean((X_val[X_val['sentiment'] == 'negative'])['which_longer']))))

        print('The selected text for positive is on average {} words smaller'.format(str(np.mean((X_val[X_val['sentiment'] == 'positive'])['which_longer']))))

        print('The selected text for neutral is on average {} words smaller'.format(str(np.mean((X_val[X_val['sentiment'] == 'neutral'])['which_longer']))))

    print("Average Jaccard score is: " + str(sum(jaccard_scores)/len(jaccard_scores)))



    max_vocab_weights = vocab_weights[jaccard_scores.index(max(jaccard_scores))]

    neg_words_adj = max_vocab_weights['neg']

    pos_words_adj = max_vocab_weights['pos']

    neutral_words_adj = max_vocab_weights['neu']



    # pos_tr = train[train['sentiment'] == 'positive']

    # neutral_tr = train[train['sentiment'] == 'neutral']

    # neg_tr = train[train['sentiment'] == 'negative']



    # cv = CountVectorizer(max_df=0.95, min_df=2, max_features=10000, stop_words='english')



    # # final_cv = cv.fit_transform(train['text'])

    # cv.fit_transform(train['text'])



    # X_pos = cv.transform(pos_tr['text'])

    # X_neutral = cv.transform(neutral_tr['text'])

    # X_neg = cv.transform(neg_tr['text'])



    # pos_final_count_df = pd.DataFrame(X_pos.toarray(), columns=cv.get_feature_names())

    # neutral_final_count_df = pd.DataFrame(X_neutral.toarray(), columns=cv.get_feature_names())

    # neg_final_count_df = pd.DataFrame(X_neg.toarray(), columns=cv.get_feature_names())

    

    

    # count_df_final = (neg_final_count_df, pos_final_count_df, neutral_final_count_df)

    # tr = (neg_tr, pos_tr, neutral_tr)

    # neg_words_adj, pos_words_adj, neutral_words_adj = build_vocab(cv, count_df_final, tr)





    for index, row in test.iterrows():

        selected_text = calculate_selected_text(pos_words_adj, neg_words_adj, row, tol)

        sample.loc[sample['textID'] == row['textID'], ['selected_text']] = selected_text





    sample.to_csv('submission.csv', index = False)



if __name__ == "__main__":

  main()
