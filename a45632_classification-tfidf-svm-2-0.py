import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tqdm
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
import sklearn.svm

import warnings
warnings.filterwarnings("ignore")

print(os.listdir("../input"))

# source Cipher #1 & Cipher #2 Full Solutions (https://www.kaggle.com/leflal/cipher-1-cipher-2-full-solutions)
dic1 = {'1': ' ', '\x1b': 'e', 't': 't', 'v': 's', '#': 'r', '0': 'h', '8': 'l', 's': '\n', 'A': 'd', '^': 'o', ']': 'f', 'O': 'a', '\x02': 'n', 'o': 'y', 'c': 'u', '_': 'c', '{': 'T', '\x03': 'b', 'z': 'v', 'a': 'i', 'W': 'w', '-': 'm', '*': 'F', 'G': ':', ';': "'", 'f': 'k', 'F': 'L', "'": 'p', 'd': 'g', '>': 'S', 'X': 'j', '\x1a': 'q', 'w': 'K', '2': 'C', ':': 'I', '9': 'P', '@': 'U', 'x': 'D', '+': 'x', 'T': ',', '%': 'O', '\x08': '.', 'q': '-', '\x1e': 'R', 'h': 'z', '!': 'X', '\x7f': 'N', '/': 'A', 'b': '@', '}': 'J', 'J': 'B', 'e': 'M', '"': 'G', '|': '(', 'y': ')', 'g': 'H', 'u': '3', '\x06': '7', '\t': '5', ',': '4', 'L': '1', '\\': '0', 'n': '8', '[': '>', ' ': '<', 'r': '&', 'l': 'W', '\x18': 'V', 'U': '[', 'i': ']', '3': ';', '~': '+', '<': '9', 'H': '2', '5': '6', '?': '/', '4': '|', 'Z': '=', '.': '~', 'm': '\\', '\x10': 'Y', ')': '_', 'S': '\x08', '6': '"', '&': '?', '\x1c': '*', 'Q': '\t', 'I': '}', '`': '#', 'B': '$', 'P': '!', 'k': '{', 'Y': '`', 'V': 'Q', 'D': '\x0c', '=': '\x10', '$': '\x02', 'K': 'Z', 'p': '\x1e', '(': '^', '\x0c': '%', 'E': 'E'}
dic2 = {'8': ' ', 'z': '\n', '$': 'e', '{': 't', '7': 'h', 'e': 'o', 'd': 'f', 'V': 'a', '\x10': 'n', 'H': 'd', '*': 'r', '\x03': 'v', '^': 'w', 'h': 'i', '}': 's', 'v': 'y', '?': 'l', 'j': 'u', '1': 'F', '4': 'm', 'N': ':', '\x18': 'b', 'f': 'c', 'B': "'", 'm': 'k', 'M': 'L', '.': 'p', 'k': 'g', 'E': 'S', '_': 'j', '9': 'C', '#': 'q', 'n': 'H', '[': ',', '&': 'R', 'x': '-', '\x06': 'T', '~': 'K', 'A': 'I', 'G': 'U', '\x7f': 'D', 'L': 'E', ',': 'O', 'o': 'z', '6': 'A', "'": '<', 'P': '}', 'Q': 'B', '=': '"', ']': 'Q', ')': 'G', '\x19': '7', '\x1b': '5', '\\': '[', 'F': '/', 'r': '{', '/': '^', '5': '~', '\n': '+', 'I': '$', 'y': '&', '!': 'V', 'g': '#', 's': 'W', ';': '|', 'i': '@', '\x1a': '.', '\x08': '(', 'l': 'M', '\x02': ')', ' ': 'Y', '(': 'X', 'c': '0', '2': 'x', 'W': '!', '-': '?', '\t': 'J', '3': '4', '<': '6', '\x0c': 'N', '@': 'P', 'S': '1', 'O': '2', 'C': '9', 'u': '8', '|': '3', '\x1e': '%', 'b': '>', 'X': '\t', '0': '_', '>': '\x03', 'J': '\x1a', '%': '*', 'Z': '\x08', 'Y': '\x06', '"': '\x1b', 't': '\\', 'a': '=', 'R': 'Z', 'p': ']', ':': ';', 'T': '\x7f', 'K': '\x0c', 'q': '\x18', '\x1c': '\x19', '`': '`', '+': '\x02'}
def decrypt(str,dict):
    out=''
    for i in range(0,len(str)):
        transit = str[i]
        out += dict[transit]
    return out

train = pd.read_csv('../input/train.csv')
# add column for clear text
train['clear']=''

# Training set Decrypt level 1
for j in range(0,len(train)):
    if train.iloc[j,1] == 1:
        scrambled = train.iloc[j,2]
        clear = decrypt(scrambled,dic1)
        train.iloc[j,4] = clear

# Training set Decrypt level 2
for j in range(0,len(train)):
    if train.iloc[j,1] == 2:
        scrambled = train.iloc[j,2]
        clear = decrypt(scrambled,dic2)
        train.iloc[j,4] = clear
# update the files...
for i in range(0,len(train)):
    if train.iloc[i,4] != '':
        train.iloc[i,2] = train.iloc[i,4]

train.drop('clear', axis=1, inplace=True)
diff1 = train[train['difficulty'] == 1]
diff2 = train[train['difficulty'] == 2]
diff3 = train[train['difficulty'] == 3]
diff4 = train[train['difficulty'] == 4]
diff1 = diff1.drop(['Id','difficulty'], axis = 1)
diff2 = diff2.drop(['Id','difficulty'], axis = 1)
diff3 = diff3.drop(['Id','difficulty'], axis = 1)
diff4 = diff4.drop(['Id','difficulty'], axis = 1)
diff1.head()
# train = pd.concat([train, pd.get_dummies(train['difficulty'])], axis=1)
# train = train.drop(['Id', 'difficulty'], axis = 1)
# train = train.iloc[:,[0,2,3,4,5,1]]
# train.columns = ['ciphertext', '1d', '2d', '3d', '4d', 'target']
# train.head()
diff1['ciphertext'] = diff1['ciphertext'].apply(lambda x: x.replace('1', ' '))
diff2['ciphertext'] = diff2['ciphertext'].apply(lambda x: x.replace('8', ' '))
diff3['ciphertext'] = diff3['ciphertext'].apply(lambda x: x.replace('8', ' '))
diff4['ciphertext'] = diff4['ciphertext'].apply(lambda x: x.replace('8', ' '))
train_diff = pd.concat([diff1, diff2, diff3, diff4])
Xtrain, Xtest, ytrain, ytest = train_test_split(train_diff.iloc[:,:1], train_diff['target'], test_size = 0.1, random_state = 0)
start = time.time()
vect = TfidfVectorizer(analyzer = 'char_wb', lowercase = False, ngram_range=(1, 6))
train_vect = vect.fit_transform(Xtrain['ciphertext'])
test_vect = vect.transform(Xtest['ciphertext'])
print('Time: ' + str(time.time() - start) + 's')
start = time.time()
model = sklearn.svm.LinearSVC()
model.fit(train_vect, ytrain)
print('Time: ' + str(time.time() - start) + 's')
pred = model.predict(test_vect)
accuracy_score(pred, ytest)
f1_score(pred, ytest, average= 'macro')
test1 = pd.read_csv('../input/test.csv')
# add column for clear text
test1['clear']=''

# treat case of line 8151,38126,38130,42245,43762
test1.iloc[8151,1] = 2
test1.iloc[38130,1] = 2
test1.iloc[42245,1] = 3
test1.iloc[43762,1] = 2

# Test set Decrypt level 1
for j in range(0,len(test1)):
    if test1.iloc[j,1] == 1:
        scrambled = test1.iloc[j,2]
        clear = decrypt(scrambled,dic1)
        test1.iloc[j,3] = clear
# Test set Decrypt level 2
for j in range(0,len(test1)):
    if test1.iloc[j,1] == 2:
        scrambled = test1.iloc[j,2]
        clear = decrypt(scrambled,dic2)
        test1.iloc[j,3] = clear
for i in range(0,len(test1)):
    if test1.iloc[i,3] != '':
        test1.iloc[i,2] = test1.iloc[i,3]
test1.drop('clear', axis=1, inplace=True)

test = test1.copy()
test_diff1 = test[test['difficulty'] == 1]
test_diff2 = test[test['difficulty'] == 2]
test_diff3 = test[test['difficulty'] == 3]
test_diff4 = test[test['difficulty'] == 4]
test_diff1['ciphertext'] = test_diff1['ciphertext'].apply(lambda x: x.replace('1', ' '))
test_diff2['ciphertext'] = test_diff2['ciphertext'].apply(lambda x: x.replace('8', ' '))
test_diff3['ciphertext'] = test_diff3['ciphertext'].apply(lambda x: x.replace('8', ' '))
test_diff4['ciphertext'] = test_diff4['ciphertext'].apply(lambda x: x.replace('8', ' '))
test_diff = pd.concat([test_diff1, test_diff2, test_diff3, test_diff4])
test1.head()
test_diff = test_diff.set_index('Id').loc[test1['Id']]
test_vect = vect.transform(test_diff['ciphertext'])
test_pred = model.predict(test_vect)
submission = pd.DataFrame([test1['Id'],test_pred]).T
submission.columns = ['Id', 'Predicted']
submission.to_csv('submission.csv', index=False)