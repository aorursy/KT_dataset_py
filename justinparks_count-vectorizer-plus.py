# Split the training set into training and validation

def splitTraining(train):



    # Grab a quarter of the testing set for the validation set

    split = (int(len(train)/4))

    return train[:-split], train[-split:]
import pandas as pd 

import numpy as np



# CountVectorizer will help calculate word counts

from sklearn.feature_extraction.text import CountVectorizer



# Import the string dictionary that we'll use to remove punctuation

import string
# Import datasets

train, valid = splitTraining(pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv'))

test = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

sample = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
# Drop any rows where the text column is NaN for our datasets

train = train.drop(train[train['text'].isna()].index.tolist())

valid = valid.drop(valid[valid['text'].isna()].index.tolist())
# Make all the text lowercase - casing doesn't matter when 

# we choose our selected text.

train['text'] = train['text'].apply(lambda x: x.lower())

valid['text'] = valid['text'].apply(lambda x: x.lower())

test['text']  = test['text'].apply(lambda x: x.lower())
pos_train = train[train['sentiment'] == 'positive']

neutral_train = train[train['sentiment'] == 'neutral']

neg_train = train[train['sentiment'] == 'negative']
# Use CountVectorizer to get the word counts within each dataset



cv = CountVectorizer(max_df=.95, min_df=2,

                                     max_features=10000,

                                     stop_words='english')



X_train_cv = cv.fit_transform(train['text'])



X_pos = cv.transform(pos_train['text'])

X_neutral = cv.transform(neutral_train['text'])

X_neg = cv.transform(neg_train['text'])



pos_count_df = pd.DataFrame(X_pos.toarray(), columns=cv.get_feature_names())

neutral_count_df = pd.DataFrame(X_neutral.toarray(), columns=cv.get_feature_names())

neg_count_df = pd.DataFrame(X_neg.toarray(), columns=cv.get_feature_names())



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

    neg_words_adj[key] = neg_words[key] - (neutral_words[key] + pos_words[key])

    

for key, value in pos_words.items():

    pos_words_adj[key] = pos_words[key] - (neutral_words[key] + neg_words[key])

    

for key, value in neutral_words.items():

    neutral_words_adj[key] = neutral_words[key] - (neg_words[key] + pos_words[key])
def calculate_selected_text(df_row, tol = 0):

    

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

        #if(new_sum > score):

            score = new_sum

            selection_str = lst[i]

            #tol = tol*5 # Increase the tolerance a bit each time we choose a selection



    # If we didn't find good substrings, return the whole text

    if(len(selection_str) == 0):

        selection_str = words

        

    return ' '.join(selection_str)
pd.options.mode.chained_assignment = None
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
tol = 0.001



train['predicted_selection'] = ''



for index, row in train.iterrows():

    

    selected_text = calculate_selected_text(row, tol)

    

    train.loc[train['textID'] == row['textID'], ['predicted_selection']] = selected_text
train['jaccard'] = train.apply(lambda x: jaccard(x['selected_text'], x['predicted_selection']), axis = 1)

print('The jaccard score for the training set is:', np.mean(train['jaccard']))
tol = 0.001



valid['predicted_selection'] = ''



for index, row in valid.iterrows():

    

    selected_text = calculate_selected_text(row, tol)

    

    valid.loc[valid['textID'] == row['textID'], ['predicted_selection']] = selected_text
valid['jaccard'] = valid.apply(lambda x: jaccard(x['selected_text'], x['predicted_selection']), axis = 1)

print('The jaccard score for the validation set is:', np.mean(valid['jaccard']))
train = pd.concat([train, valid], ignore_index=True)

pos_tr = train[train['sentiment'] == 'positive']

neutral_tr = train[train['sentiment'] == 'neutral']

neg_tr = train[train['sentiment'] == 'negative']
cv = CountVectorizer(max_df=0.95, min_df=2,

                                     max_features=10000,

                                     stop_words='english')



final_cv = cv.fit_transform(train['text'])



X_pos = cv.transform(pos_tr['text'])

X_neutral = cv.transform(neutral_tr['text'])

X_neg = cv.transform(neg_tr['text'])



pos_final_count_df = pd.DataFrame(X_pos.toarray(), columns=cv.get_feature_names())

neutral_final_count_df = pd.DataFrame(X_neutral.toarray(), columns=cv.get_feature_names())

neg_final_count_df = pd.DataFrame(X_neg.toarray(), columns=cv.get_feature_names())
pos_words = {}

neutral_words = {}

neg_words = {}



for k in cv.get_feature_names():

    pos = pos_final_count_df[k].sum()

    neutral = neutral_final_count_df[k].sum()

    neg = neg_final_count_df[k].sum()

    

    pos_words[k] = pos/(pos_tr.shape[0])

    neutral_words[k] = neutral/(neutral_tr.shape[0])

    neg_words[k] = neg/(neg_tr.shape[0])
neg_words_adj = {}

pos_words_adj = {}

neutral_words_adj = {}



for key, value in neg_words.items():

    neg_words_adj[key] = neg_words[key] - (neutral_words[key] + pos_words[key])

    

for key, value in pos_words.items():

    pos_words_adj[key] = pos_words[key] - (neutral_words[key] + neg_words[key])

    

for key, value in neutral_words.items():

    neutral_words_adj[key] = neutral_words[key] - (neg_words[key] + pos_words[key])
tol = 0.001



for index, row in test.iterrows():

    

    selected_text = calculate_selected_text(row, tol)

    

    sample.loc[sample['textID'] == row['textID'], ['selected_text']] = selected_text

    
print(sample)

sample.to_csv('submission.csv', index = False)