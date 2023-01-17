# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/All-seasons.csv')
df.head()
df.describe()
df['words'] = df['Line'].str.split()

stopwords = list(filter(lambda x: re.sub('\W','',x), ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', "ain't", 'all',

                    'also', 'although', 'am', 'an', 'and', 'any', 'anyone', 'are', "aren't", 'as', 'at', 'be',

                    'because', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',

                    'between', 'beyond', 'both', 'bout', 'but', 'by', 'can', 'cannot', 'cant', "can't", "can't",

                    'con', 'could', 'couldn', 'couldnt', "couldn't", 'did', 'didn', 'didnt', "didn't", "didn't",

                    'do', 'does', 'doesnt', "doesn't", "doesn't", 'doing', 'done', 'dont', "don't", "don't", 'down',

                    'during', 'each', 'either', 'else', 'elsewhere', 'every', 'except', 'few', 'for', 'from',

                    'further', "h'd", 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd",

                    "he'll", 'hence', 'her', 'here', "here's", 'hers', 'herself', "he's", "he's", 'him', 'himself',

                    'his', 'how', "how's", 'i', "i'd", "i'd", 'ie', 'if', "i'll", "i'll", 'im', "i'm", "i'm", 'in',

                    'into', 'is', 'isn', "isn't", 'it', "it'", "it'", "it'd", "it'll", "it'll", 'its', "it's", "it's",

                    'itself', "i've", "i've", 'just', "let's", 'made', 'make', 'many', 'may', 'me', 'might', 'mine',

                    'more', 'most', 'mostly', 'much', 'must', "mustn't", 'my', 'myself', 'neither', 'no', 'nor',

                    'not', 'of', 'off', 'on', 'once', 'only', 'onto', 'opposite', 'or', 'other', 'others', 'ought',

                    'our', 'ours', 'ourselves', 'out', 'over', 'own', 'per', 'rather', 'same', "shan't", 'she',

                    "she'd", "she'll", "she's", 'should', "shouldn't", 'since', 'so', 'some', 'such', 'than', 'that',

                    "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these',

                    'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'throughout',

                    'thru', 'thus', 'til', 'to', 'too', 'toward', 'under', 'until', 'up', 'upon', 've', 'very',

                    'was', "wasn't", 'we', "we'd", "we'll", 'were', "we're", "we're", "weren't", "we've", 'what',

                    "what's", 'when', 'whenever', "when's", 'where', "where's", 'wherever', 'whether', 'which',

                    'whichever', 'while', 'who', 'whom', "who's", 'whose', 'why', "why's", 'with', 'within',

                    'without', 'wonna', 'wont', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", 'your',

                    "you're", "you're", 'yours', 'yourself', 'yourselves', "you've", "you've"]))

chars = {}

import re

for line in df[['Character','words']].iterrows():

    character = line[1].tolist()[0]

    words = line[1]

    if character not in chars:

        chars[character] = {}

    all_words = chars[character]

    for word in words.tolist()[1]:

        w = re.sub('\W', "", word.lower())

        if w not in stopwords:

            if w not in all_words:

                all_words[w] = 0

            all_words[w] = all_words[w] + 1
chars
import operator

def top_ten(words):

    return sorted(words.items(), key=operator.itemgetter(1), reverse=True)[:10]
top_ten(chars['Cartman'])
top_ten(chars['Butters'])
top_ten(chars['Clyde'])