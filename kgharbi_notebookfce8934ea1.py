#imports

#from IPython.core.interactiveshell import InteractiveShell

#InteractiveShell.ast_node_interactivity = "all"

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

import sklearn as skl

import numpy as np

import re
#read train file

train = pd.read_csv("../input/train.csv")

train.head()
#First let's see if there is NaN questions

train.count()
#We have to delete those 2 lines

train = train[train.question2.notnull()]

train.count()



#After a delete we must reset the index (we will use it later)

train = train.reset_index(drop=True)
########## This an example of what we will do it ####################

splitQuestion1 = train.loc[0,'question1'].split(' ')

splitQuestion2 = train.loc[0,'question2'].split(' ')

print('Q1: ', splitQuestion1)

print('\nQ2: ', splitQuestion2)
#So let's remove '?' and all special characters before the split

question1 = re.sub('[^a-zA-Z0-9 \n\.]', '', train.loc[0,'question1'])

question2 = re.sub('[^a-zA-Z0-9 \n\.]', '', train.loc[0,'question2'])

print('Q1: ', question1)

print('Q2: ', question2)
#Now let's split again...

splitQuestion1 = question1.split(' ')

#train.loc[0,'question1'].split(' ')

splitQuestion2 = question2.split(' ')

#train.loc[0,'question2'].split(' ')

print('Q1: ', splitQuestion1)

print('\nQ2: ', splitQuestion2)
#...and count the number of words in common

numberOfWordsInCommon = 0

for word in splitQuestion1:

    if word in splitQuestion2:

        splitQuestion2.remove(word)

        numberOfWordsInCommon+=1

             

print("the number of words in common is:", numberOfWordsInCommon)

######### the example ends here ##################
#Now that we understood what we want to do, let's do all this for the dataset

#Replacing special carachters...

train['question1'] = train['question1'].apply(lambda x: re.sub('[^a-zA-Z0-9 \n\.]', '', x))

train['question2'] = train['question2'].apply(lambda x: re.sub('[^a-zA-Z0-9 \n\.]', '', x))



#...and counting th number of words in common A REPRENDRE ICI
#Let's do all this for the dataset. WARNING: this code takes time. Around 3 minutes

listOfNumberOfWordsInCommon = []

for numberLine in range(train.shape[0]):

    splitQuestion1 = train.loc[numberLine,'question1'].split(' ')

    splitQuestion2 = train.loc[numberLine,'question2'].split(' ')

    numberOfWordsInCommon = 0

    for word in splitQuestion1:

        if word in splitQuestion2:

            splitQuestion2.remove(word)

            numberOfWordsInCommon+=1

    listOfNumberOfWordsInCommon.append(numberOfWordsInCommon)
#Now let's merge the listOfNumberOfWordsInCommon with the train dataframe

ColOfNumberOfWordsInCommon=[]

ColOfNumberOfWordsInCommon = pd.DataFrame({'NumberOfWordsInCommon': listOfNumberOfWordsInCommon})

train = pd.concat([train,ColOfNumberOfWordsInCommon], axis = 1)

train = train[['id', 'qid1', 'qid2', 'question1', 'question2','NumberOfWordsInCommon','is_duplicate']]

train.head()
#Let's see if the NumberOfWordsInCommon is related to is_duplicate

train2 = train[['NumberOfWordsInCommon','is_duplicate']]



#First we split the orginal Dataframe in 2. One for the duplicated questions and one for the non-duplicated questions

train_is_not_duplicate = train2[train2.is_duplicate == 0]

train_is_duplicate = train2[train2.is_duplicate == 1]
#Now let's count the number of rows for each NumberOfWordsInCommon

train_is_duplicate = train_is_duplicate.groupby("NumberOfWordsInCommon").count()

train_is_not_duplicate = train_is_not_duplicate.groupby("NumberOfWordsInCommon").count()
#And change the columns name

train_is_not_duplicate.columns = ['NumberOfQuestionPairsNotDuplicated']

train_is_duplicate.columns = ['NumberOfQuestionPairsDuplicated']
#Now we can merge

compare = pd.concat([train_is_duplicate, train_is_not_duplicate], axis=1)

compare
#The NA values means that there is 0 question pairs

compare.fillna(0, inplace=True)
#Comparing barplot

barWidth = 0.3

y1 = compare.NumberOfQuestionPairsNotDuplicated

y2 = compare.NumberOfQuestionPairsDuplicated

x1 = range(len(compare))

x2 = [i + barWidth for i in x1]

plt.figure(figsize=(20,7))



plt.bar(x1, y1, width = barWidth, color = "red")

plt.bar(x2, y2, width = barWidth, color = "green")

plt.show()