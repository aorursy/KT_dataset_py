# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import re

import numpy as np

import pandas as pd

import nltk

from nltk.tokenize import sent_tokenize, word_tokenize

import string

from string import punctuation



#data sets

positive_data = open (r'/kaggle/input/ist-course/positive-words.txt')

positive = list()

for line in positive_data:

    line = line.rstrip('\n')

    positive.append(line)



negative_data = open (r'/kaggle/input/ist-course/negative-words.txt')

negative = list()

for line in negative_data:

    line = line.rstrip('\n')

    negative.append(line)



switcher_data = open (r'/kaggle/input/ist-course/switcher.txt')

switcher = list()

for line in switcher_data:

    line = line.rstrip('\n')

    switcher.append(line)





contrasting_data = open (r'/kaggle/input/ist-course/contrasting.txt')

contrasting = list()

for line in contrasting_data:

    line = line.rstrip('\n')

    contrasting.append(line)



#network programming

def sentiment(chars):

    sent = nltk.sent_tokenize(chars)

    sent = [each_sent.lower() for each_sent in sent]

    lst = ''

    count = 0

    for a in sent:

        a = a.rstrip('\n')

        if a[-1] == '?':

            lst = lst + 'neutral'

            count = count + 0

        elif len(re.findall('.* should .*',a)) > 0:

            lst = lst + 'neutral'

            count = count + 0

        elif len(re.findall('.* must .*',a)) > 0:

            lst = lst + 'neutral'

            count = count + 0

        elif len(re.findall('.* ought to .*',a)) > 0:

            lst = lst + 'neutral'

            count = count + 0

        elif len(re.findall('.* promise .*',a)) > 0:

            lst = lst + 'neutral'

            count = count + 0

        elif len(re.findall('.* wish .*',a)) > 0:

            lst = lst + 'neutral'

            count = count + 0

        elif len(re.findall('.* if only .*',a)) > 0:

            lst = lst + 'neutral'

            count = count + 0

        elif len(re.findall('.* if .*',a)) > 0:

            lst = lst + 'neutral'

            count = count + 0

        elif len(re.findall('.* hope .*',a)) > 0:

            lst = lst + 'neutral'

            count = count + 0

        else:

            a = a.translate(a.maketrans('','', string.punctuation))

            token = nltk.word_tokenize(a)

            data = np.zeros(len(token))

            arr = np.array(data)

            for i in range (len (token)):

                if token[i] in positive:

                    arr[i] = 1

                if token[i] in negative:

                    arr[i] = -1

                if token[i] in switcher:

                    arr[i] = 2

                if token[i] in contrasting:

                    arr[i] = 3

            array = arr[arr!=0]

            if 1 not in array and -1 not in array:

                lst = lst + 'neutral'

                count = count + 0

            elif 3 not in array:

                if array[0] == 1:

                    lst = lst + 'positive'

                    count = count + 1

                if array[0] == -1:

                    lst = lst + 'negative'

                    count = count -1

                if array[0] == 2:

                    if array[1] == 1:

                        lst = lst + 'negative'

                        count = count -1

                    if array[1] == -1:

                        lst = lst + 'positive'

                        count = count + 1

            else:

                for f in range(len(array)):

                    if array[f] ==3:

                        if 1 not in array and -1 not in array:

                            lst = lst + 'neutral'

                            count = count + 0

                        if 2 not in array[f:] and 1 in array[f:]:

                            lst = lst + 'positive'

                            count = count + 1

                        if 2 not in array[f:] and -1 in array[f:]:

                            lst = lst + 'negative'

                            count = count -1

                        if 2 in array[f:] and 1 in array[f:]:

                            lst = lst + 'negative'

                            count = count -1

                        if 2 in array[f:] and -1 in array[f:]:

                            lst = lst + 'positive'

                            count = count + 1

    # sentiment for multi sentence

    if count == 0:

        return 'neutral'

    elif count > 0:

        return 'positive'

    else:

        return 'negative'



def array(chars):

    sent = nltk.sent_tokenize(chars)

    sent = [each_sent.lower() for each_sent in sent]

    match = list()

    for a in sent:

        a = a.rstrip('\n')

        if a[-1] == '?':

            match.append(('[...?]', 'question'))

        elif len(re.findall('.* should .*',a)) > 0:

            match.append(('[..., should, ...]', 'suggestion'))

        elif len(re.findall('.* must .*',a)) > 0:

            match.append(('[..., must, ...]', 'suggestion'))

        elif len(re.findall('.* ought to .*',a)) > 0:

            match.append(('[..., ought to, ...]', 'suggestion'))

        elif len(re.findall('.* promise .*',a)) > 0:

            match.append(('[..., promise, ...]', 'promise'))

        elif len(re.findall('.* wish .*',a)) > 0:

            match.append(('[..., wish, ...]', 'wish'))

        elif len(re.findall('.* if only .*',a)) > 0:

            match.append(('[..., if only, ...]', 'wish'))

        elif len(re.findall('.* if .*',a)) > 0:

            match.append(('[..., if, ...]', 'wish'))

        elif len(re.findall('.* hope .*',a)) > 0:

            match.append(('[..., hope, ...]', 'wish'))

        else:

            a = a.translate(a.maketrans('','', string.punctuation))

            token = nltk.word_tokenize(a)

            data = np.zeros(len(token))

            arr = np.array(data)

            for i in range (len (token)):

                if token[i] in positive:

                    match.append((token[i], 'positive'))

                    arr[i] = 1

                if token[i] in negative:

                    match.append((token[i], 'negative'))

                    arr[i] = -1

                if token[i] in switcher:

                    match.append((token[i], 'switcher'))

                    arr[i] = 2

                if token[i] in contrasting:

                    match.append((token[i], 'contrasting'))

                    arr[i] = 3

            array = arr[arr!=0]

            match.append((arr, 'full array'))

            match.append((array, 'array'))

    return match



def positiveword(chars):

    wordlst = [a for a, b in chars if b == 'positive'or b=='negative' and b == 'switcher']

    return ' '.join(wordlst)



def negativeword(chars):

    wordlst = [a for a, b in chars if b == 'negative'or b=='positive' and b == 'switcher']

    return ' '.join(wordlst)



#sentiment analysis for train dataset

def file_comment():

    data = pd.read_csv(r'/kaggle/input/tweet-sentiment-extraction/train.csv')

    data = data.applymap(str)

    data["sentiment"] = data["text"].apply(lambda x: sentiment(x))

    data['compare'] = np.where(data['sentiment by Kaggle'] == data['sentiment'], 'same', 'different')

    data.to_csv ('justanothertest.csv')
