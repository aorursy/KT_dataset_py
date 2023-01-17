# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import svm

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))

from collections import Counter

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

import time

import zipfile

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#  Train and Test file directories

TRAIN_DIR = r"../input/chapter1/train-mails"

TEST_DIR = r"../input/chapter1/test-mails"
#  Function to create a dictionary where non-alphabetical characters or single charcaters are removed

def make_Dictionary(root_dir):

    all_words = []

    emails = [os.path.join(root_dir,f) for f in os.listdir(root_dir)]

    for mail in emails:

#         Extracting each mail datas

        with open(mail, encoding='latin-1') as m:

            for line in m:

                words = line.split()

                all_words += words

#     creating a dictionary of words alog with number of occurences

    dictionary = Counter(all_words)

    list_to_be_removed = dictionary.keys()

    list_to_be_removed = list(list_to_be_removed)

    for item in list_to_be_removed:

        if item.isalpha() == False:

            del dictionary[item]

        elif len(item) == 1:

            del dictionary[item]

#     Extracting most common 3000 items from the dictionary

    dictionary = dictionary.most_common(3000)

    return dictionary
# Function to extract features from the set corpus



def extract_features(mail_dir):

    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]

    features_matrix = np.zeros((len(files),3000)) #Creating a Matrix of documents ID vs Word ID

    train_labels = np.zeros(len(files))

    count = 0;

    docID = 0;

    for fil in files:

        with open(fil, encoding='latin-1') as fi:

            for i,line in enumerate(fi): 

                if i == 2: # as the Main Text starts in the 3rd line where 1st and 2nd line corresponds to Subject of the mail and a newline respectively

                    words = line.split()

                    for word in words:

                        wordID = 0

                        for i,d in enumerate(dictionary):

                            if d[0] == word:

                                wordID = i

                                features_matrix[docID,wordID] = words.count(word)

                                

        train_labels[docID] = 0;

        filepathTokens = fil.split('/')

        lastToken = filepathTokens[len(filepathTokens) - 1]

        if lastToken.startswith("spmsg"): # Checks if the file name has "spmsg" in it

            train_labels[docID] = 1; # Marks the label as 1 if the mail name has "spmsg"

            count = count + 1

        docID = docID + 1

    return features_matrix, train_labels
start = time.time() # To check the start time

dictionary = make_Dictionary(TRAIN_DIR) # create a most common Word dictionary along with their counts for 3000 words

# print(dictionary)

print ("reading and Extracting emails from file.")

features_matrix, labels = extract_features(TRAIN_DIR)

test_feature_matrix, test_labels = extract_features(TEST_DIR)



end= time.time() # To check end time

print((end-start)/60) # to check total time taken by the procedure
# Using Gaussian Naive Bayes Model for Classification

model = GaussianNB()

# model = svm.SVC(C=0.1)

print(set(labels))

print ("Training model.")

#train model

model.fit(features_matrix, labels)



predicted_labels = model.predict(test_feature_matrix)



print ("FINISHED classifying. accuracy score : ")

print (accuracy_score(test_labels, predicted_labels))
# Using SVM Model for Classification with tuning Hyper-parameters

# model = GaussianNB()

model = svm.SVC(kernel='linear')

df_output = pd.DataFrame(columns=['C_val', 'Kernel', 'Gamma','Accuracy'])

print(set(labels))

count=0

print ("Training model.")

for ker_name in ['linear', 'poly', 'rbf', 'sigmoid']:

    for c_val in np.linspace(0,1,num=100)[1:][::9]:

        for gamma_val in ['auto', 'scale']:

            try:

                df_output.loc[count,'Kernel'] = ker_name

                df_output.loc[count,'C_val'] = c_val

                df_output.loc[count,'Gamma'] = gamma_val

                model = svm.SVC(C=c_val, kernel=ker_name, gamma=gamma_val)

                model.fit(features_matrix, labels)

                predicted_labels = model.predict(test_feature_matrix)

                df_output.loc[count,'Accuracy'] = accuracy_score(test_labels, predicted_labels)

                print(count)

                count+=1

            except:

                print(ker_name,c_val, gamma_val)

#             print ("FINISHED classifying. accuracy score with Parameteres C:{}, kernel:{},gamma :{} ".format(c_val,ker_name, gamma_val))

#             print (accuracy_score(test_labels, predicted_labels))

df_output.to_csv('accuracy with Tuned Parameters_SVM.csv', index=False)