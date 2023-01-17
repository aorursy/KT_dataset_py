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

import random

f = open('../input/pararomance', "rt")

text = f.readlines()

df = pd.DataFrame({'novel_titles':text})

print(df.head(10))
#removing ... annd \n

data_1 = [i for i in df.novel_titles.str.strip(' ') if i != '']

data_1 = [word.replace(".", "") for word in data_1]

data = []

for i in range(0,len(data_1)):

    data.append(data_1[i].rstrip('\n'))



print(data[0:5])



#creating empty list for Markov chain       

markov=  {i:[] for i in data}

#linking the chain using key values

for before, after in zip(data, data[1:]):

    markov[before].append(after)
#randomly selecting a value

new = list(markov.keys())

seed = random.randrange(len(new))

currentWord = random.choice(new)

sentence = [currentWord]

for i in range(0, random.randrange(1, 2)):

    check = markov[currentWord]

    if (len(check) > 0):

                nextWord = random.choice(check)

                sentence.append(nextWord)

                currentWord = nextWord

    else:

                currentWord = random.choice(new)

print (" ".join(sentence))