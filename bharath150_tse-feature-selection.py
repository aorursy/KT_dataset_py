# libraries



from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from nltk.tokenize import WordPunctTokenizer 

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.metrics import accuracy_score



import pandas as pd

import numpy as np
# reading train data

data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')



# creating a new target variable based on sentiment



data['target'] = 0



data.loc[data['sentiment']=='positive', 'target'] = 1

data.loc[data['sentiment']=='negative', 'target'] = 2



data


# removing empty rows

data['text'].replace('', np.nan, inplace=True)

data.dropna(subset=['text'], inplace=True)

data.reset_index(drop=True, inplace=True)



# spliting train data into train and cv

x_train, x_cv, y_train, y_cv = train_test_split(data.drop(['sentiment'],axis = 1),data['target'], test_size = 0.2, random_state = 30)

# Creating wordpuncttokenizer and using it as tokenizer for countvectorizer

tokenizer = WordPunctTokenizer()

use_tokenizer = False



# creating BOW

if use_tokenizer:

    vectorizer = CountVectorizer(tokenizer = tokenizer.tokenize,max_features = 10000, min_df=2, max_df=0.95)

else:

    vectorizer = CountVectorizer(max_features = 10000 , min_df=2, max_df=0.95)



x_train_text = vectorizer.fit_transform(x_train['text'],)

x_cv_text = vectorizer.transform(x_cv['text'],)



# Training a classifier to predict sentiment

alpha = [0.00001,0.0001,0.001,0.01,0.1,1,10,100]

cv_score = []

for i in alpha:

    clf = MultinomialNB(alpha = i)

    clf.fit(x_train_text, y_train)

    print('accuracy for alpha=',i, 'is:',accuracy_score(y_cv, clf.predict(x_cv_text)))

    cv_score.append(accuracy_score(y_cv, clf.predict(x_cv_text)))
# retraining with best alpha

best_alpha = alpha[cv_score.index(max(cv_score))]

clf = MultinomialNB(alpha = best_alpha)

clf.fit(x_train_text, y_train)

print('accuracy for best alpha=',best_alpha, 'is:',accuracy_score(y_cv, clf.predict(x_cv_text)))
# metric code is from kaggle page. https://www.kaggle.com/c/tweet-sentiment-extraction/overview/evaluation

def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    return float(len(c)) / (len(a) + len(b) - len(c))
# each dictionary consists of probability of word in that target(0,1,2)

dict0 = dict(zip(vectorizer.get_feature_names(),clf.feature_log_prob_[0] ))

dict1 = dict(zip(vectorizer.get_feature_names(),clf.feature_log_prob_[1] ))

dict2 = dict(zip(vectorizer.get_feature_names(),clf.feature_log_prob_[2] ))

def threshold_preds(k, use_tokenizer,df):

    """

    Performs predictions of selected_text for tweets in given data frame.

    inputs:

    k: multiplier for average probabilties for words in each tweet

    use_tokenizer: whether to use simple split or wordpuncttokenizer

    df: dataframe on which to perform predictions

    

    output:

    list of predictions of selected_text for a given dataframe"""

    

    preds = []



    for index,item in df.iterrows():



        if item.target!=0 :

            if(use_tokenizer):

                temp = WordPunctTokenizer().tokenize(item['text'])

            else:

                temp = item['text'].split()

            sentiment = item['target']

            if sentiment==1:

                probs = dict1

            else:

                probs = dict2

            temp_score = 0

            temp_text = ''

            for a in temp:

                a = a.lower()

                if a in probs:

                    temp_score+=probs[a]

                



            for a in temp:

                a = a.lower()

                if a in probs:

                    if probs[a]>k*temp_score/len(temp):

                     

                        temp_text=temp_text +' ' + a

                

            preds.append(temp_text)

        else:

            preds.append(item.text)

    return preds
# finding best k based on cv_score

k = np.linspace(0,11,20)

cv_score = []



for i in k:

    preds = threshold_preds(i,use_tokenizer,x_cv)

    score = 0

    count = 0

    for index,item in x_cv.iterrows():

        score+=jaccard(item.selected_text,preds[count])

        count+=1

    score = score/count

    cv_score.append(score)

    print('jaccard score for threshold',i,'is:',score)       

best_k = k[cv_score.index(max(cv_score))]
# test predictions



test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')

test['target'] = 0

test.loc[test['sentiment']=='positive', 'target'] = 1

test.loc[test['sentiment']=='negative', 'target'] = 2





preds = threshold_preds(best_k,use_tokenizer,test)



submission = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')

submission['selected_text'] = preds



submission.to_csv('submission.csv', index = False)