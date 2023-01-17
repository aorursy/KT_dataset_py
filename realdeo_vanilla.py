import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nltk import word_tokenize, sent_tokenize

from random import shuffle



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/ukara-enhanced/dataset.csv")
df.head(10)
df.groupby(['kelompok', 'label']).count()
len(df)
groups = {}



kelompok = [1, 3, 4, 7, 8, 9, 10, 'A', 'B']



for I in kelompok:

    groups[(I,0)] = df.query("kelompok == '%s' and label == 0 " % (str(I)))['teks'].values 



for I in kelompok:

    groups[(I,1)] = df.query("kelompok == '%s' and label == 1 " % (str(I)))['teks'].values



for K in groups:

    length = len(groups[K])

    groups[K] = [list(groups[K][length*(J)//5 : length*(J+1)//5]) for J in range(5)]



def generate_fold_data(test_index, sentence_preprocess, dictionary):

    train = []

    test  = {}

    

    for K in dictionary:

        if K[0] not in test:

            test[K[0]] = []

        for index in range(5):

            if test_index == index:

                for M in dictionary[K][index]:

                    test[K[0]].append((sentence_preprocess(K, M), K[1]))

            else:

                for M in dictionary[K][index]:

                    train.append((sentence_preprocess(K, M), K[1]))

    

    shuffle(train)

    return train, test
file = open("/kaggle/input/ukara-enhanced/stopword_list.txt")

    

stopwords = [I.strip() for I in file.readlines()]



file.close()
def remove_stopwords(words):

    words = word_tokenize(words.lower())

    words = [I for I in words if I not in stopwords]

    return " ".join(words)
def vanila_preprocess(group_id, sentence):

    sentence = remove_stopwords(sentence)

    return sentence
from sklearn.feature_extraction.text import CountVectorizer as CV

from sklearn.feature_extraction.text import TfidfVectorizer as TV
countVector = CV(min_df = 5)

tfidfVector = TV(min_df = 5)
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import f1_score
def format_accuracy(dictionary_result):

    total = 0

    for I in dictionary_result:

        total += dictionary_result[I]

        print(I, dictionary_result[I])

    print("Macro All", total / len(dictionary_result))    
dictionary = {}

for J in kelompok:

    dictionary[J] = 0

    for test_index in range(5):

        train,test = generate_fold_data(test_index, vanila_preprocess, {(J,0) : groups[(J,0)], 

                                                                   (J,1) : groups[(J,1)]})

        train_X = countVector.fit_transform([I[0] for I in train])

        train_y = [I[1] for I in train]

        test_X  = countVector.transform([I[0] for I in test[J]])

        test_y  = [I[1] for I in test[J]]

        

        model = MultinomialNB()

        model.fit(train_X, train_y)

        

        prediction = model.predict(test_X)

        dictionary[J] += f1_score(test_y, prediction)

    dictionary[J] /= 5
format_accuracy(dictionary)
dictionary = {}

for J in kelompok:

    dictionary[J] = 0

    for test_index in range(5):

        train,test = generate_fold_data(test_index, vanila_preprocess, {(J,0) : groups[(J,0)], 

                                                                   (J,1) : groups[(J,1)]})

        train_X = tfidfVector.fit_transform([I[0] for I in train])

        train_y = [I[1] for I in train]

        test_X  = tfidfVector.transform([I[0] for I in test[J]])

        test_y  = [I[1] for I in test[J]]

        

        model = MultinomialNB()

        model.fit(train_X, train_y)

        

        prediction = model.predict(test_X)

        dictionary[J] += f1_score(test_y, prediction)

    dictionary[J] /= 5
format_accuracy(dictionary)
bigramVector = CV(min_df = 5, ngram_range=(1, 2))
dictionary = {}

for J in kelompok:

    dictionary[J] = 0

    for test_index in range(5):

        train,test = generate_fold_data(test_index, vanila_preprocess, {(J,0) : groups[(J,0)], 

                                                                   (J,1) : groups[(J,1)]})

        train_X = bigramVector.fit_transform([I[0] for I in train])

        train_y = [I[1] for I in train]

        test_X  = bigramVector.transform([I[0] for I in test[J]])

        test_y  = [I[1] for I in test[J]]

        

        model = MultinomialNB()

        model.fit(train_X, train_y)

        

        prediction = model.predict(test_X)

        dictionary[J] += f1_score(test_y, prediction)

    dictionary[J] /= 5
format_accuracy(dictionary)
from sklearn.svm import LinearSVC
dictionary = {}

for J in kelompok:

    dictionary[J] = 0

    for test_index in range(5):

        train,test = generate_fold_data(test_index, vanila_preprocess, {(J,0) : groups[(J,0)], 

                                                                   (J,1) : groups[(J,1)]})

        train_X = countVector.fit_transform([I[0] for I in train])

        train_y = [I[1] for I in train]

        test_X  = countVector.transform([I[0] for I in test[J]])

        test_y  = [I[1] for I in test[J]]

        

        model = LinearSVC()

        model.fit(train_X, train_y)

        

        prediction = model.predict(test_X)

        dictionary[J] += f1_score(test_y, prediction)

    dictionary[J] /= 5
format_accuracy(dictionary)
from sklearn.ensemble import AdaBoostClassifier as Ada
dictionary = {}

for J in kelompok:

    dictionary[J] = 0

    for test_index in range(5):

        train,test = generate_fold_data(test_index, vanila_preprocess, {(J,0) : groups[(J,0)], 

                                                                   (J,1) : groups[(J,1)]})

        train_X = countVector.fit_transform([I[0] for I in train])

        train_y = [I[1] for I in train]

        test_X  = countVector.transform([I[0] for I in test[J]])

        test_y  = [I[1] for I in test[J]]

        

        model = Ada()

        model.fit(train_X, train_y)

        

        prediction = model.predict(test_X)

        dictionary[J] += f1_score(test_y, prediction)

    dictionary[J] /= 5
format_accuracy(dictionary)
from sklearn.linear_model import LogisticRegression as LR
dictionary = {}

for J in kelompok:

    dictionary[J] = 0

    for test_index in range(5):

        train,test = generate_fold_data(test_index, vanila_preprocess, {(J,0) : groups[(J,0)], 

                                                                   (J,1) : groups[(J,1)]})

        train_X = countVector.fit_transform([I[0] for I in train])

        train_y = [I[1] for I in train]

        test_X  = countVector.transform([I[0] for I in test[J]])

        test_y  = [I[1] for I in test[J]]

        

        model = LR()

        model.fit(train_X, train_y)

        

        prediction = model.predict(test_X)

        dictionary[J] += f1_score(test_y, prediction)

    dictionary[J] /= 5
format_accuracy(dictionary)
dictionary = {}



for J in kelompok:

    dictionary[J] = 0



for test_index in range(5):

    train,test = generate_fold_data(test_index, vanila_preprocess, groups)

    train_X = bigramVector.fit_transform([I[0] for I in train])

    train_y = [I[1] for I in train]

    

    model = MultinomialNB()

    model.fit(train_X, train_y)

    for J in kelompok:

        test_X  = bigramVector.transform([I[0] for I in test[J]])

        test_y  = [I[1] for I in test[J]]

        

        prediction = model.predict(test_X)



        dictionary[J] += f1_score(test_y, prediction)



for J in kelompok:

    dictionary[J] /= 5
format_accuracy(dictionary)
dictionary = {}



for J in kelompok:

    dictionary[J] = 0



for test_index in range(5):

    train,test = generate_fold_data(test_index, vanila_preprocess, groups)

    train_X = bigramVector.fit_transform([I[0] for I in train])

    train_y = [I[1] for I in train]

    

    model = LinearSVC()

    model.fit(train_X, train_y)

    for J in kelompok:

        test_X  = bigramVector.transform([I[0] for I in test[J]])

        test_y  = [I[1] for I in test[J]]

        

        prediction = model.predict(test_X)



        dictionary[J] += f1_score(test_y, prediction)



for J in kelompok:

    dictionary[J] /= 5
format_accuracy(dictionary)
dictionary = {}



for J in kelompok:

    dictionary[J] = 0



for test_index in range(5):

    train,test = generate_fold_data(test_index, vanila_preprocess, groups)

    train_X = bigramVector.fit_transform([I[0] for I in train])

    train_y = [I[1] for I in train]

    

    model = LR()

    model.fit(train_X, train_y)

    for J in kelompok:

        test_X  = bigramVector.transform([I[0] for I in test[J]])

        test_y  = [I[1] for I in test[J]]

        

        prediction = model.predict(test_X)



        dictionary[J] += f1_score(test_y, prediction)



for J in kelompok:

    dictionary[J] /= 5
format_accuracy(dictionary)
dictionary = {}



for J in kelompok:

    dictionary[J] = 0



for test_index in range(5):

    train,test = generate_fold_data(test_index, vanila_preprocess, groups)

    train_X = bigramVector.fit_transform([I[0] for I in train])

    train_y = [I[1] for I in train]

    

    model = Ada()

    model.fit(train_X, train_y)

    for J in kelompok:

        test_X  = bigramVector.transform([I[0] for I in test[J]])

        test_y  = [I[1] for I in test[J]]

        

        prediction = model.predict(test_X)



        dictionary[J] += f1_score(test_y, prediction)



for J in kelompok:

    dictionary[J] /= 5
format_accuracy(dictionary)