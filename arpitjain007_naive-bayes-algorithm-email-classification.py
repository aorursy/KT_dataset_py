# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from collections import Counter

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.naive_bayes import GaussianNB , MultinomialNB , BernoulliNB

from sklearn.metrics import accuracy_score , confusion_matrix , classification_report , precision_score

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
Train_path = "../input/chapter1/train-mails"

Test_path = "../input/chapter1/test-mails"
def makeMydict(path):

    raw = [os.path.join(path , e) for e in os.listdir(path)]

    all_words =[]

    for email in raw:

        with open(email) as m:

            for line in m:

                words = line.split()

                all_words+= words

    dictionary = Counter(all_words)

    list_to_remove= list(dictionary)

    for item in list_to_remove:

        if item.isalpha() == False:

            del dictionary[item]

        elif len(item) == 1:

            del dictionary[item]

    

    dictionary = dictionary.most_common(4000)

    

    return dictionary
def extractFeature(path):

    files = [ os.path.join(path, e) for e in os.listdir(path)]

    feature_matrix = np.zeros((len(files), 4000))

    labels = np.zeros((len(files)))

    count=0

    docID =0

    for file in files:

        with open(file) as f:

            for i , line in enumerate(f):

                if i==2:

                    words = line.split()

                    for word in words:

                        wordID=0

                        for i,d in enumerate(dictionary):

                            if d[0] == word:

                                wordID = i

                                feature_matrix[docID , wordID] = words.count(word)

            labels[docID]=0

            filepathtokens = file.split('/')

            lastToken = filepathtokens[-1]

            if lastToken.startswith('spmsg'):

                labels[docID] =1

                count +=1

            docID = docID+1

        

        return feature_matrix , labels
dictionary = makeMydict(Train_path)

dictionary_2 = makeMydict(Test_path)



train_features , train_labels = extractFeature(Train_path)

test_features , test_labels = extractFeature(Test_path)
model = GaussianNB()

model2 = MultinomialNB()

model3 = BernoulliNB()
model2.fit(train_features , train_labels)

model3.fit(train_features , train_labels)

model.fit(train_features , train_labels)



predict = model.predict(test_features)

predict2 = model2.predict(test_features)

predict3 = model3.predict(test_features)
accuracy = accuracy_score(test_labels , predict)

accuracy2 = accuracy_score(test_labels , predict2)

accuracy3 = accuracy_score(test_labels , predict3)
print(accuracy , accuracy2 , accuracy3)