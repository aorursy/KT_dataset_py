# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from collections import defaultdict

from wordcloud import WordCloud, STOPWORDS

import nltk



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory







# Any results you write to the current directory are saved as output.

data = pd.read_csv('../input/debate.csv', encoding= "latin1")

data.head()
data['Speaker'].value_counts().plot(kind='bar', color='pink')
dates = data.Date.unique()



data1 = data[data.Date == dates[0]]

data2 = data[data.Date == dates[1]]

data3 = data[data.Date == dates[2]]
data1[data1.Line==13]
data1[data1.Line==15]
data1[(data1.Line==10) & (data.Speaker == 'Holt')].Text.values[0]
data1[(data1.Line==19) & (data.Speaker == "Holt")].Text.values[0]
def last_name(string, word1, word2):

    '''

    A function that returns the word (either word1 or word2) which was spoken 

    at the last in a particular string. If neither of word1 and word2 is present in the string, then the function

    returns 'null'.

    '''

    tokens = nltk.word_tokenize(string)

    return_word = 'null'

    for word in tokens:

        if word == word1:

           return_word = word1

        if word == word2:

           return_word = word2



    return return_word

    

    

Candi1 = 'Trump'

Candi2 = 'Clinton'



def whose_chance_to_speak(data, moderator):

    '''

    A function that returns a dictionary that returns the name of the person whose

    chance it is to speak for that particular line

    

    Data: data which has distinct lines (date=constant)

    moderator = a list of moderators/ can be a single name

    

    Returns a dictionary - line:person

    where line is the line number and person is the candidate whose chance it is to speak

    

    '''

    

    chance_dict = defaultdict(str)

    leader = data[data.Line == 1].Speaker.values[0]

    chance_dict[1] = leader    

    

    for line in data.Line:

        if chance_dict[line] == '':

            if (data.loc[(data.Line ==line), 'Speaker'] != Candi1).values[0] and (data.loc[(data.Line ==line), 'Speaker'] != Candi2).values[0] and (data.loc[(data.Line ==line), 'Speaker'] != moderator).values[0]:

                chance_dict[line] = leader

            

            elif (data[data.Line == line].Speaker.values[0] in moderator):

                name = last_name(data[data.Line==line].Text.values[0], Candi1, Candi2) 

                if name != 'null':

                    leader = name

                    chance_dict[line+1] = leader

                else:

                    continue

            else:

                chance_dict[line] = leader

        else:

            continue

    

    return chance_dict
chance_dict =  whose_chance_to_speak(data1, 'Holt')                 

dataframe = pd.DataFrame(list(chance_dict.items()), columns = ['Line','whose_chance'])        



data1 = pd.merge(data1,dataframe, how='left', on='Line')



#Check a sample of rows

data1[7:15]
data1[(data1.Speaker == 'Trump') & (data1.whose_chance == 'Clinton')]
data1[(data1.Speaker == 'Clinton') & (data1.whose_chance == 'Trump')]
clinton_interrupted = (len(data1[(data1.Speaker == 'Trump') & (data1.whose_chance == 'Clinton')])/len(data1[(data1.whose_chance == 'Clinton')]))*100



print("Clinton got interrupted by Donald %2f percent times out of all her chances to speak" % clinton_interrupted )
trump_interrupted = (len(data1[(data1.Speaker == 'Clinton') & (data1.whose_chance == 'Trump')])/len(data1[(data1.whose_chance == 'Trump')]))*100

print("Trump got interrupted by Clinton %2f percent times out of all his chances to speak" % trump_interrupted )