# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!pip install syllables

!pip install cmudict
import syllables

import cmudict
import string

syl_dict = cmudict.dict()



def num_syl_cmudict(word):

    '''Count the number of syllables in a word using cmudict

    

    Args:

        - word (str): the word to be counted

        

    Returns:

        - An integer representing the number of syllables, 

          or None if the syllables couldn't be counted

    '''

    # Remove all punctuation from the word

    word = word.translate(str.maketrans('', '', string.punctuation))

    

    # Check if word is in the cmudict

    syls = syl_dict[word]

    

    if len(syls) == 0:

        est = syllables.estimate(word)

        return est

    else:

        ct = 0

        # The cmudict actually lists all sounds in the word. Counting in

        # this way allows you to identify the number of syllables.

        # From StackOverflow:

        # https://datascience.stackexchange.com/questions/23376/how-to-get-the-number-of-syllables-in-a-word

        for sound in syls[0]:

            if sound[-1].isdigit():

                ct += 1

        

        return ct



def is_haiku(sent):

    '''Checks if a given piece of text can be split into the haiku 5-7-5 syllable structure

    

    Args:

        - sent (str): the sentence to be tested

    

    Returns:

        - A boolean indicating whether the text is composed of 

          words that can be separated into a haiku

    '''

    words = sent.split(" ")

    

    count = 0

    hit_5 = False

    hit_7 = False



    # Loop through all the words and check for the 

    # intermediate milestones of a haiku structure

    for word in words:

        num_syl = num_syl_cmudict(word)



        if num_syl is None:

            return False

        

        count += num_syl

        

        # We hit the first five syllables - reset the counter

        if count == 5 and not hit_5:

            count = 0

            hit_5 = True

        

        # If we hit five and then found 7 syllables, 

        # we have our second haiku line

        if count == 7 and hit_5:

            count = 0

            hit_7 = True

    

    # If we hit 5 and 7 and there are only 5

    # syllables left, we have a haiku-able text

    return ((count == 5) and hit_5 and hit_7)
# Process the dataframe and dump the output into CSV



news_df = pd.read_csv("../input/abcnews-date-text.csv")



news_df['is_haiku'] = news_df['headline_text'].map(is_haiku)

haiku = news_df[news_df['is_haiku']]

haiku.to_csv('haiku_list.csv')
np.sum(news_df['is_haiku'])