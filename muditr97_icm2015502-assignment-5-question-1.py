import numpy as np 

import pandas as pd 

import os

from sklearn.model_selection import train_test_split as tts

import random

import csv

print(os.listdir("../input"))

df = pd.read_csv("../input/assignment6.csv")

df = df[['v1','v2']]

df = df.replace({'ham' : 0, 'spam': 1 })

df.head()

print("done")
stop_words = ["", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", 

                  "you", "your", "yours", "yourself", "yourselves", "he", "him", 

                  "his", "himself", "she", "her", "hers", "herself", "it", "its",

                  "itself", "they", "them", "their", "theirs", "themselves", "what",

                  "which", "who", "whom", "this", "that", "these", "those", "am", "is",

                  "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",

                  "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because",

                  "as", "until", "while", "of", "at", "by", "for", "with", "about", "against",

                  "between", "into", "through", "during", "before", "after", "above", "below",

                  "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",

                  "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",

                  "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",

                  "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",

                  "can", "will", "just", "don", "should", "now"]



y = df['v1']

second_col = df['v2']

second_col = second_col[0:len(second_col)-2]

#print(second_col)

#print(type(second_col[0]))

l = []

for s in list(second_col):

    if type(s) == float:

        continue

    w = s.split(" ")

    for i in w:

        if i in stop_words:

            w.remove(i)

    

    str1 = ' '.join(str(e) for e in w)

    l.append(str1)

# l = list of strings in which the stop words are remnoved

#print(l)

tmp = ""

final_list = []

for s in list(l):

    if type(s) == float:

        continue

    for i in s:

        if  (i.isalnum()) or (i == " "):

            tmp += (str(i).lower())

    final_list.append(tmp)

    tmp = ""





#print(final_list)



word_freq = {}



for x in final_list:

    p = x.split(" ")

    for it in p:

        if it not in word_freq.keys():

            word_freq[it] = 1

        else:

            word_freq[it] = word_freq[it] + 1



#for z,x in word_freq.items():

#    print(str(z) + " " + str(x))

     



final_list = np.array(final_list)



x_df = final_list

y_df = y[0:5569]

y_df = np.array(y_df)

print(type(y_df))

#print(len(x_df))

#print(len(y_df))

X_train, X_test, Y_train, Y_test = tts(final_list, y_df, test_size = 0.3, random_state = 5)



print("done")
def x_data_transform(x, w_f) :

    obtained_data = []

    for i in range(x.shape[0]):

        temp = []

        for val in w_f:

            if val in x[i] : 

                temp.append(float(1));

            else :

                temp.append(float(0));

        temp = np.array(temp)

        obtained_data.append(temp)



    obtained_data = np.array(obtained_data)

    return obtained_data



def y_data_transform(y) :

    obtained_data = []

    for i in range(y.shape[0]) :

        if(y[i] == 1) :

            obtained_data.append(float(1))

        else :

            obtained_data.append(float(0))

    obtained_data = np.array(obtained_data)

    return obtained_data



print("done")
X_train = x_data_transform(X_train, word_freq)

X_test = x_data_transform(X_test, word_freq)

Y_train = y_data_transform(Y_train)

Y_test = y_data_transform(Y_test)

print("done")
# prediction for x_j(jth word of dictionary) when y = 1

def x_prediction_1(X_train, Y_train, word_index) :

    den = 0

    for i in range(Y_train.shape[0]) :

        den = den + Y_train[i]

    den = float(den)

    #total spams

    

    num = 0

    for i in range(X_train.shape[0]) :

        if(Y_train[i] == 1 and X_train[i][word_index] == 1) :

            num = num + 1

    num = float(num)

    #when classifying in spam, e.g how many times this have occurred in spam in this particular training set

    

    return float((num + 1.0)/ (den + 2.0))



# prediction for x when y = 0

def x_prediction_0(X_train, Y_train, word_index) :

    den = 0

    for i in range(Y_train.shape[0]) :

        den = den + (float(1) - Y_train[i])

    den = float(den)

    

    num = 0

    for i in range(X_train.shape[0]) :

        if(Y_train[i] == 0 and X_train[i][word_index] == 1) :

            num = num + 1

    num = float(num)

    

    return float((num + float(1))/ (den + float(2)))



# probability for some value of y,  for some jth word of dictionary

def x_probability(phi, x_val) :

    val1 = phi ** x_val

    val2 = (1 - phi) ** (1 - x_val)

    

    return val1  * val2





print("done")
den = float(Y_train.shape[0])

num = 0

for i in range(Y_train.shape[0]) :

    num = num + Y_train[i]

num = float(num)

phi = float(num / den)



y_probability_1 = phi

y_probability_0 = (1.0 - phi)





x_prob_vals_1 = np.zeros(X_train.shape[1])

x_prob_vals_0 = np.zeros(X_train.shape[1])



for i in range(X_train.shape[1]) :

    x_prob_vals_1[i] = x_prediction_1(X_train, Y_train, i)

    x_prob_vals_0[i] = x_prediction_0(X_train, Y_train, i)

print("done")
# check accuracy for testing data

correct = 0

for it in range(X_test.shape[0]) :

    den = 0.0

    den1 = y_probability_1

    for j in range(X_test.shape[1]) :

        den1 = den1 * x_probability(x_prob_vals_1[j], X_test[it][j])

    den = den + den1

    den1 = y_probability_0

    for j in range(X_test.shape[1]) :

        den1 = den1 * x_probability(x_prob_vals_0[j], X_test[it][j])

    den = den + den1



    # for spam prediction

    num = float(0)

    num1 = y_probability_1

    for j in range(X_test.shape[1]) :

        num1 = num1 * x_probability(x_prob_vals_1[j], X_test[it][j])

    num = num + num1

    

    spam_prediction = num / den

    

    # for spam prediction

    num = float(0)

    num1 = y_probability_0

    for j in range(X_test.shape[1]) :

        num1 = num1 * x_probability(x_prob_vals_0[j], X_test[it][j])

    num = num + num1

    

    ham_prediction = num / den

    

    if(spam_prediction >= ham_prediction and Y_test[it] == 1) :

        correct = correct + 1

    elif(spam_prediction < ham_prediction and Y_test[it] == 0) :

        correct = correct + 1

                

percentage_accuracy = (float(correct) / float(X_test.shape[0])) * 100

print('Percentage accuracy on testing data: ', percentage_accuracy)