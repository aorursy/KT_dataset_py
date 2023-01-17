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
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
def getTokens(inputString): #custom tokenizer. ours tokens are characters rather than full words
    tokens = []
    for i in inputString:
        tokens.append(i)
    return tokens
data = pd.read_csv(r'../input/data.csv', error_bad_lines=False)
data = pd.DataFrame(data)
passwords = np.array(data)
random.shuffle(passwords) #shuffling randomly for robustness
y = [d[1] for d in passwords] #labels
allpasswords= [d[0] for d in passwords]
vectorizer = TfidfVectorizer(tokenizer=getTokens) #vectorizing
X = vectorizer.fit_transform(allpasswords)
data_process = np.concatenate((np.array(y).reshape(669640, 1), X.toarray()), axis=1)
df = pd.DataFrame(data_process)
df.rename(columns={0: 'Label'})
df.to_csv('data_process.csv', index=False)
from collections import Counter
import re


def repeat(string):
    # print(list(filter(lambda x: x[-1] > 1, Counter(string).values())))
    n = sum(list(filter(lambda x: x > 1, Counter(string).values())))
    
    return - n * (n - 1)


def consecutive_upper(string):
    re_ = re.compile('[A-Z]{2,}')
    result = sum(map(len, re_.findall(string)))
    return - (result * 2)


def consecutive_lower(string):
    re_ = re.compile('[a-z]{2,}')
    result = sum(map(len, re_.findall(string)))
    return - (result * 2)




check_list = {
    "Number of characters in the password": (lambda x: len(x) * 4),
    "Number of LC characters": (lambda x: (len(x) - len([c for c in x if c.islower()])) * 2),
    "Number of UC characters ": (lambda x: (len(x) - len([c for c in x if c.isupper()])) * 2),
    "Number of digits": (lambda x: len([d for d in x if d.isdigit()]) * 4),
    "Number of symbols": (lambda x: len([c for c in x if not c.isalnum()]) * 6),
    "Characters only": (lambda x: -1 * len([c for c in x if c.isalpha()])),
    "Digits only": (lambda x: -1 * len([d for d in x if d.isdigit()])),
    "Number of repeat characters": (lambda x: repeat(x)),
    "Number of consecutive uppercase characters": (lambda x: consecutive_upper(x)),
    "Number of consecutive lower characters": (lambda x: consecutive_lower(x))
}
pw = "P@ssword123!"
for key, value in check_list.items():
    print(value(pw))
print("Score: ", sum([value(pw) for key, value in check_list.items()]))
Counter(pw).values()
