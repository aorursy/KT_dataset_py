# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import string

import copy

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



train_CSV_Name,test_CSV_Name='/kaggle/input/tweet-sentiment-extraction/train.csv','/kaggle/input/tweet-sentiment-extraction/test.csv'





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        

print(train_CSV_Name,test_CSV_Name)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df=pd.read_csv(train_CSV_Name)

test_df=pd.read_csv(test_CSV_Name)



train_df=train_df.dropna()

test_df=test_df.dropna()







def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = str(text).lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text



ori_train_df=train_df.copy()

ori_test_df=test_df.copy()

train_df['text'] = train_df['text'].apply(lambda x: clean_text(x))

train_df['selected_text'] = train_df['selected_text'].apply(lambda x: clean_text(x))

test_df['text'] = test_df['text'].apply(lambda x: clean_text(x))



ori_train_df,ori_valid_df=train_test_split(

    ori_train_df, train_size = 0.9, random_state = 0)



train_df,valid_df = train_test_split(

    train_df, train_size = 0.9, random_state = 0)









def makeVocabulary(Xarr):

    vectorizer = CountVectorizer(min_df=20,preprocessor=clean_text,

                                     stop_words='english')

    wordsDict = {}

    

    words=vectorizer.fit(Xarr)

    voca = {k:v for k, v in vectorizer.vocabulary_.items()}

#     totalvals=sum(voca.values())

    totalvals=len(Xarr)

    return voca



def assignWeight(Xarr,all_voca):

    vectorizer = CountVectorizer(preprocessor=clean_text,

                                     stop_words='english')

    words=vectorizer.fit(Xarr)

    voca = {k:v for k, v in vectorizer.vocabulary_.items()}

    tempvoca=all_voca.copy()

    for key in all_voca:

        if(key in voca):

            tempvoca[key]=tempvoca[key]/len(Xarr)

    return tempvoca



def emphasizeVoca(voca1,voca2,voca3):

    temp=copy.deepcopy(voca1)

    for word in voca1:

        if(word in voca2):

            temp[word]=temp[word]-voca2[word]

        if(word in voca3):

            temp[word]=temp[word]-voca3[word]

    return temp



ALL_df=train_df.copy()

ALL_X=train_df['text'].tolist()

ALL_Voca=makeVocabulary(ALL_X)



positive_df=train_df.loc[train_df['sentiment'] == 'positive']

positive_X=positive_df['text'].tolist()

positive_voca=assignWeight(positive_X,ALL_Voca)



negative_df=train_df.loc[train_df['sentiment'] == 'negative']

negative_X=negative_df['text'].tolist()

negative_voca=assignWeight(negative_X,ALL_Voca)



neutral_df=train_df.loc[train_df['sentiment'] == 'neutral']

neutral_X=neutral_df['text'].tolist()

neutral_voca=assignWeight(neutral_X,ALL_Voca)







new_positive_voca=emphasizeVoca(positive_voca,negative_voca,neutral_voca)

new_negative_voca=emphasizeVoca(negative_voca,positive_voca,neutral_voca)



print(new_positive_voca)
def predict_Selected_text(clean_text,ori_text,voca):

    tweet = clean_text

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

            if(lst[i][p].translate(str.maketrans('','',string.punctuation)) in voca.keys()):

                new_sum += voca[lst[i][p].translate(str.maketrans('','',string.punctuation))]

            

        # If the sum is greater than the score, update our current selection

        if(new_sum > score):

            score = new_sum

            selection_str = lst[i]

            #tol = tol*5 # Increase the tolerance a bit each time we choose a selection



    # If we didn't find good substrings, return the whole text

    if(len(selection_str) == 0):

        selection_str = words

        

    return ' '.join(selection_str)

    

def generateResult(test_df,ori_test_df,positive_voca,negative_voca):

    selected_texts=[]

    ori_texts=ori_test_df['text']

    for index,row in test_df.iterrows():

        text = row.text

        output_str = ""

#         print('selected\t',row.selected_text)

        if row.sentiment == 'neutral'or len(text.split()) <= 2:

            selected_texts.append(text)

#             print('result\t',text)

#             print('\n')

        elif row.sentiment == 'positive':

            selected_texts.append(predict_Selected_text(text,ori_texts[index],positive_voca))

        else:

            selected_texts.append(predict_Selected_text(text,ori_texts[index],negative_voca))

            

    return selected_texts



predict_selected_texts1=generateResult(test_df,ori_test_df,new_positive_voca,new_negative_voca)

print(test_df.shape)

print(len(predict_selected_texts1))



predict_selected_texts2=generateResult(train_df,ori_train_df,new_positive_voca,new_negative_voca)

print(train_df.shape)

print(len(predict_selected_texts2))



predict_selected_texts3=generateResult(valid_df,ori_valid_df,new_positive_voca,new_negative_voca)

print(valid_df.shape)

print(len(predict_selected_texts3))
def jaccard(str1, str2): 

    a = set(str1.lower().split()) 

    b = set(str2.lower().split())

    c = a.intersection(b)

    if((len(a) + len(b) - len(c))==0):

        return 1

    return float(len(c)) / (len(a) + len(b) - len(c))



selected_texts2=train_df['selected_text']

selected_texts3=valid_df['selected_text']





score=0



for x,y in zip(selected_texts2,predict_selected_texts2):

    score+=jaccard(x,y)

print('Train: ',round(score/len(selected_texts2),3))



score=0



for x,y in zip(selected_texts3,predict_selected_texts3):

    score+=jaccard(x,y)

print('Valid: ',round(score/len(selected_texts3),3))





ids=ori_test_df['textID']

data={'textID':ids,'selected_text':predict_selected_texts1}

submission=pd.DataFrame(data) 

submission.to_csv('submission.csv', index = False)

print(submission)








