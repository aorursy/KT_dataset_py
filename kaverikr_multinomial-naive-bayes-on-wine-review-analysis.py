import csv

import math

import numpy as np



file = open("../input/wine-reviews/winemag-data-130k-v2.csv",encoding="utf8")

reader = csv.reader(file)

data = [[row[2].lower().replace(",","").replace(".", "").replace("!", "").replace("?", "")

           .replace(";", "").replace(":", "").replace("*", "")

           .replace("(", "").replace(")", "")

           .replace("/", ""),row[4]]

           for row in reader]
data = data[1:]
#tokenize the string descriptions (converting the larger string into array of words).

data = [[obs[0].split(),int(obs[1])] for obs in data]
#shaffling the datset

np.random.shuffle(data)



training = data[:10000]



# classified classes 0 - for average comments and  1 - good comments

classes = [0,1]
def train_naive_bayes(training, classes):



    # initialisations

    D_c = [[]] * len(classes)



    n_c = [None] * len(classes)



    logprior = [None] *len(classes)



    loglikelihood = [None] * len(classes)



    #  splitting traning examples as into different classes

    #  if ratting >= 90

    #     class = 1

    #  otherwise if rating < 90

    #     class = 0



    for obs in training:

        if obs[1] >= 90:

            D_c[1] = D_c[1] + [obs]



        elif obs[1] < 90:

            D_c[0] = D_c[0] + [obs]





    # Creating vocabulary of words which we use in  our

    # event model to classing reviews.

    # In our event model consists of all the distict words that we have recieved in our reviews.



    V = []

    for obs in training:

         for word in obs[0]:

             if word in V:

                 continue

             else:

                 V.append(word)



    # size of vocabulary

    V_size = len(V)





    # size of traing examples

    n_docs = len(training)



    #

    for ci in range(len(classes)):



        # no. of training examples for a particular class

        n_c[ci] = len(D_c[ci])



        # P(ci)  = no. of taining example  classifies as class ci / total no. of training example in the entire dataset

        logprior[ci] = np.log((n_c[ci] +1)/n_docs)



        #Counts the total number of words in class ci

        count_w_in_V = 0

        for d in D_c[ci]:

            count_w_in_V = count_w_in_V + len(d[0])



        # appling laplace smothing by adding V_size

        denom = count_w_in_V + V_size



        # finding the P(wi/ci), i.e the frequency of words wi in training examples if class ci and saving the same in dic[wi]

        # and further in lolikelihood[ci], such that P(wi|ci) = loglikelihood[ci][word]

        dic={}

        for wi in V:



            count_w_in_D_c = 0

            for d in D_c[ci]:

                for word in d[0]:

                    if word == wi:

                        count_w_in_D_c = count_w_in_D_c + 1



            #applying laplace smothing by adding one, to resolve cases when count of word in dc is 0(to avoid impossible probability).

            numer = count_w_in_D_c + 1

            dic[wi] = np.log((numer)/(denom))



            loglikelihood[ci] = dic



    return (V,logprior,loglikelihood)
V, logprior, likelihood = train_naive_bayes(training[0:500],classes)

# print(logprior)

# print(likelihood)

def test_naive_bayes (testdoc, logprior, likelihood,V):

    # storing probabilities of each class

    logpost = [None] * len(classes)



    for ci in classes:

        sumofloglikelihood = 0



        for word in testdoc:



            if word in V:

                 # finding summation of all P(wi|ci) in testdoc

                 sumofloglikelihood += likelihood[ci][word]



        # log(P(ci) + lg(summation)

        logpost[ci] = logprior[ci] + sumofloglikelihood



        #returns the max class probability

    return logpost.index(max(logpost))
test = training[500:510]

for i in test:

 predicted_C =  test_naive_bayes(i[0],logprior,likelihood,V)

 print(i)

 print(predicted_C)