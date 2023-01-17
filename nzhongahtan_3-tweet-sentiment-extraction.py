import numpy as np 

import pandas as pd 

import re

import nltk.corpus

nltk.download('stopwords')

from nltk.corpus import stopwords

stop = stopwords.words('english')

from nltk import ngrams

from nltk import word_tokenize 

import string

import operator

from collections import Counter

from nltk.sentiment.vader import SentimentIntensityAnalyzer



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')

test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')



train.head(5)
string.punctuation

def remove_punc (string1):

    string1 = string1.lower()

    translation_table = dict.fromkeys(map(ord,string.punctuation),' ')

    string2 = string1.translate(translation_table)

    return string2

def remove_stop (string1):

    pattern = re.compile(r'\b(' +r'|'.join(stopwords.words('english'))+r')\b\s*')

    string2 = pattern.sub('', string1)

    return string2



train['text_backup'] = train['text']

train['text_select_backup'] = train['selected_text']

test['text_backup'] = test['text']



train['text'] = train['text'].astype(str)

train['selected_text'] = train['selected_text'].astype(str)

test['text'] = test['text'].astype(str)



train['text'] = train['text'].apply(lambda x:remove_stop(x))

train['text'] = train['text'].apply(lambda x:remove_punc(x))

train['selected_text'] = train['selected_text'].apply(lambda x:remove_stop(x))

train['selected_text'] = train['selected_text'].apply(lambda x:remove_punc(x))

test['text'] = test['text'].apply(lambda x:remove_stop(x))

test['text'] = test['text'].apply(lambda x:remove_punc(x))



train.head(5)
#Features

#length of input or count of words

train['Feature_1'] = train['text_backup'].apply(lambda x: len(str(x).split()))

train['Feature_1a'] = train['text_select_backup'].apply(lambda x: len(str(x).split()))

test['Feature_1'] = test['text_backup'].apply(lambda x: len(str(x).split()))



#avg length of word

train['Feature_a'] = train["text_select_backup"].apply(lambda x: len(str(x)))

train['Feature_a'] = train['Feature_a'] / train['Feature_1a']



#number of stopwords

stop_words = set(stopwords.words('english'))

train['Feature_2'] = train['text_backup'].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]))

train['Feature_2a'] = train['text_select_backup'].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]))

test['Feature_2'] = test['text_backup'].apply(lambda x: len([w for w in str(x).lower().split() if w in stop_words]))



#most words

all_text_without_sw = ''

for i in train.itertuples():

    all_text_without_sw = all_text_without_sw + str(i.text)

counts = Counter(re.findall(r"[\w']+",all_text_without_sw))

del counts ["'"]

sorted_x = dict(sorted(counts.items(),key=operator.itemgetter(1),reverse=True)[:50])

train['Feature_3'] = train['text'].apply(lambda x: len([w for w in str(x).lower().split()if w in sorted_x]))

train['Feature_3a'] = train['selected_text'].apply(lambda x: len([w for w in str(x).lower().split()if w in sorted_x]))

test['Feature_3'] = test['text'].apply(lambda x: len([w for w in str(x).lower().split()if w in sorted_x]))



#least words

reverted_x = dict(sorted(counts.items(),key=operator.itemgetter(1))[:10000])

train['Feature_4'] = train['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in reverted_x]))

train['Feature_4a'] = train['selected_text'].apply(lambda x: len([w for w in str(x).lower().split() if w in reverted_x]))

test['Feature_4'] = test['text'].apply(lambda x: len([w for w in str(x).lower().split() if w in reverted_x]))



#punctuation

train['Feature_5'] = train['text_backup'].apply(lambda x: len([w for w in str(x) if w in string.punctuation]))

train['Feature_5a'] = train['text_select_backup'].apply(lambda x: len([w for w in str(x) if w in string.punctuation]))

test['Feature_5'] = test['text_backup'].apply(lambda x: len([w for w in str(x) if w in string.punctuation]))



#Most popular word

p = train["text_select_backup"]

first = []

for sent in p:

    sent = str(sent)

    wordsss = sent.split()

    first.append(wordsss[0])

Counter = Counter(first)

most_occur= Counter.most_common(4)



train.head(5)
sid = SentimentIntensityAnalyzer()

avg_length = round(np.average(train['Feature_1a']), 0) + round(np.std(train['Feature_1a'])/2)

avg_stop = round(np.average(train['Feature_2a']),0) + round(np.std(train['Feature_2a'])/2 + 1)

avg_punc = round(np.average(train['Feature_5a']),0)

avg_word = round(np.average(train['Feature_a']),0)



subset = test['text_backup']

sentiment = test['sentiment']
def get_1st_pos(lst):

    index = []

    j=1

    for i in lst:

        if i <= 0:

            index.append(100000000)

        elif i > 0:

            index.append(j)

        j=j+1

    return np.argmin(index)



def get_last_pos (lst):

    index = []

    j = 1

    for i in lst:

        if i <= 0:

            index.append(-10000000)

        elif i > 0:

            index.append(j)

        j = j+1

    return np.argmax(index)



def get_1st_neg (lst):

    index = []

    j=1

    for i in lst:

        if i >= 0:

            index.append(100000000)

        elif i < 0:

            index.append(j)

        j=j+1

    return np.argmin(index)



def get_last_neg (lst):

    index = []

    j = 1

    for i in lst:

        if i >= 0:

            index.append(-10000000)

        elif i < 0:

            index.append(j)

        j = j+1

    return np.argmax(index)



def remove(string):

    return string.replace (" ","")
bad_words = ['not','no','oh']
def positive (sent):

    sent=re.sub('http[s]?://\S+','',str(sent))

    words = sent.split()

    score_list=[]

    s = ' '

    a=0

    for w in words:

        w = w.translate(str.maketrans('','',string.punctuation))

        w = remove(w)

        score = sid.polarity_scores(w)['compound']

        score_list.append(score)

        if score < 0:

            a = a+1

    if words[np.argmin(score_list)-1] in bad_words:

        b = np.argmin(score_list)

        j = np.argmin(score_list)-1

        score_list.remove(min(score_list))

        score_list.insert(b,0)

        del score_list[j]

        score_list.insert(j,0)

        a = a-1 

    if a < 0:

        a = 0

    #First Max

    max_index = np.argmax(score_list)

    max_val = score_list[max_index]

    

    #Second Max

    if len(words)>1 and np.count_nonzero(score_list) != 0:

        score_list.remove(score_list[max_index])

        if max_index > np.argmax(score_list):

            max2_index = np.argmax(score_list) 

            max_val2 = score_list[max2_index]

        elif max_index <= np.argmax(score_list):

            max2_index = np.argmax(score_list)+1

            max_val2 = score_list[max2_index-1]

        score_list.insert(max_index,max_val)

    else:

        max2_index = 0

    

    #Cycling Thorugh

    if np.count_nonzero(score_list)-a == 0:

        if a > 0 and np.argmin(score_list)>len(score_list)/2:

            ans = words[0:np.argmin(score_list)]

            return s.join(ans)

        elif a>0 and np.argmin(score_list)<len(score_list)/2:

            ans = words[np.argmin(score_list)+1: len(score_list)]

            return s.join(ans)

        else:

            return sent

    elif np.count_nonzero(score_list) - a == 1 :

        if a == 1 and np.argmin(score_list) < max_index:

            ans = words[np.argmin(score_list)+1:max_index+1]

            return s.join(ans)

        elif a==1 and np.argmin(score_list) > max_index:

            ans = words[max_index:np.argmin(score_list)]

            return s.join(ans)

        else:

            return words[max_index]

    elif np.count_nonzero(score_list) - a== 2:

        if abs(max_val - max_val2) < 0.36:

            if max_index > max2_index:

                ans =  words [max2_index:max_index+1]

                return s.join(ans)

            else:

                ans= words[max_index:max2_index+1]

                return s.join(ans)

        elif min(score_list) < 0: 

            if max_index < max2_index and np.argmin(score_list) < max2_index and np.argmin(score_list) > max_index:

                ans =  words [max_index:np.argmin(score_list)]

                return s.join(ans)

            elif max_index > max2_index and np.argmin(score_list) > max2_index and np.argmin(score_list) < max_index:

                ans =  words [max2_index:np.argmin(score_list)]

                return s.join(ans)

    else:

        first_max = get_1st_pos(score_list)

        last_max = get_last_pos(score_list)

        if min(score_list) < 0 and np.argmin(score_list) > first_max and np.argmin(score_list) < last_max:

            if max_index > np.argmin(score_list):

                ans =  words[np.argmin(score_list)+1:max_index+1]

                return s.join(ans)

            else:

                ans = words[max_index:np.argmin(score_list)]

                return s.join(ans)

        else:

            ans =  words[first_max:last_max+1]

            return s.join(ans)

            

def negative (sent):

    sent=re.sub('http[s]?://\S+','',str(sent))

    words = sent.split()

    score_list=[]

    s=' '

    a = 0

    for w in words:

        w = w.translate(str.maketrans('','',string.punctuation))

        w = remove(w)

        score = sid.polarity_scores(w)['compound']

        score_list.append(score)

        if score > 0:

            a = a+1

    if words[np.argmax(score_list)-1] in bad_words:

        b = np.argmax(score_list)

        j = np.argmax(score_list)-1

        score_list.remove(max(score_list))

        score_list.insert(b,0)

        del score_list[j]

        score_list.insert(j,0)

        a = a-1 

    if a < 0:

        a = 0

    #First Min

    min_index = np.argmin(score_list)

    min_val = score_list[min_index]

    

    #Second Min

    if len(words)>1 and np.count_nonzero(score_list) != 0:

        score_list.remove(score_list[min_index])

        if min_index > np.argmin(score_list):

            min2_index = np.argmin(score_list) 

            min_val2 = score_list[min2_index]

        elif min_index <= np.argmin(score_list):

            min2_index = np.argmin(score_list)+1

            min_val2 = score_list[min2_index-1]

        score_list.insert(min_index,min_val)

    else:

        min2_index = 0

    

    

    #Cycling Thorugh

    if np.count_nonzero(score_list)-a == 0:

        if a>0 and np.argmax(score_list) > len(score_list)/2:

            ans = words[0:np.argmax(score_list)]

            return s.join(ans)

        elif a>0 and np.argmax(score_list) < len(score_list)/2:

            ans = words[np.argmax(score_list)+1:len(score_list)]

            return s.join(ans)

        else:

            return sent

    elif np.count_nonzero(score_list)-a == 1:  

        if a == 1 and np.argmax(score_list) < min_index:

            ans = words[np.argmax(score_list)+1:min_index+1]

            return s.join(ans)

        elif a==1 and np.argmax(score_list) > min_index:

            ans =  words[min_index:np.argmax(score_list)]

            return s.join(ans)

        else:

            return words[min_index]

    elif np.count_nonzero(score_list)-a== 2:

        if abs(min_val - min_val2) < 0.375:

            if min_index > min2_index:

                ans = words [min2_index:min_index+1]

                return s.join(ans)

            else:

                ans =  words[min_index:min2_index+1]

                return s.join(ans)

        elif max(score_list) > 0: 

            if min_index < min2_index and np.argmax(score_list) < min2_index and np.argmax(score_list) > min_index:

                ans =  words [min_index:np.argmax(score_list)]

                return s.join(ans)

            elif min_index > min2_index and np.argmax(score_list) > min2_index and np.argmax(score_list) < min_index:

                ans =  words [min2_index:np.argmax(score_list)]

                return s.join(ans)

    else:

        first_min = get_1st_neg(score_list)

        last_min = get_last_neg(score_list)

        if max(score_list) > 0 and np.argmax(score_list) > first_min and np.argmax(score_list) < last_min:

            if min_index > np.argmax(score_list):

                ans =  words[np.argmax(score_list)+1:min_index+1]

                return s.join(ans)

            else:

                ans = words[min_index:np.argmax(score_list)]

                return s.join(ans)

        else:

            ans =  words[first_min:last_min+1]    

            return s.join(ans)

def neutral (sent):

    sent=re.sub('http[s]?://\S+','',str(sent))

    words = sent.split()

    score_list=[]

    s = ' '

    for w in words:

        w = w.translate(str.maketrans('','',string.punctuation))

        w = remove(w)

        score = sid.polarity_scores(w)['compound']

        score_list.append(score)

    if np.count_nonzero(score_list)==0:

        return sent

    elif np.count_nonzero(score_list) > 0 and (max(score_list) < 0.36 or abs(min(score_list)<0.4)):

        return sent

    elif np.count_nonzero(score_list)==1 and np.argmax(score_list) > 0:

        if len(words)/2 > np.argmax(score_list):

            ans =  words[np.argmax(score_list)+1:len(score_list)]

            return s.join(ans)

        else:

            ans =  words[0:np.argmax(score_list)]

            return s.join(ans)

    elif np.count_nonzero(score_list)==1 and np.argmin(score_list) < 0:

        if len(words)/2 > np.argmin(score_list):

            ans =  words[0:np.argmin(score_list)]

            return s.join(ans)

        else:

            ans =  words[np.argmin(score_list)+1:len(score_list)]

            return s.join(ans)

    else:

        max_index = np.argmax(score_list)

        max_val = score_list[max_index]

        min_index = np.argmin(score_list)

        min_val = score_list[min_index]

        if max_val > abs(min_val) and max_index > len(score_list)/2:

            ans = words[0:max_index]

            return s.join(ans)

        elif max_val > abs(min_val) and max_index < len(score_list)/2:

            ans = words[max_index+1:len(score_list)]

            return s.join(ans)

        elif abs(min_val) > max_val and min_index > len(score_list)/2:

            ans = words[0:min_index]

            return s.join(ans)

        else:

            ans = words[min_index+1:len(score_list)]

            return s.join(ans)
word_list=[]

i = 0

for sent in subset:

    if sentiment[i] == 'positive':

        word_list.append(positive(sent))

    elif sentiment[i] == 'negative':

        word_list.append(negative(sent))

    else:

        word_list.append(neutral(sent))

    i=i+1
submission=pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")

submission["selected_text"]=word_list

submission.to_csv('submission.csv',index=False)