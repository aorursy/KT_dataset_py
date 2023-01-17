#Author: Michael Niemeyer 2017



# This work is an example of how an understanding of handwritten digits can be learnt unsupervised, that is without any form of labels.



# The idea is the following: Each image consists of 28 * 28 = 784 pixels p_i with values ranging from 0 to 255. We interpret each pixel p_i as a word, and its intensity value corresponds to its frequency. This way, we are able to model the image data as a collection of "documents" each consisting of 784 different "words" with different frequencies.



# The reason I used terms like "documents" and "words" is that the topic modelling concept is best understood in the text data context. In short, it is a Bayesian learning model which assumes a generative process of the data. In the text context, it assumes that each document of a collection exhibits different proportions of "topics", and each word has a probability associated with each topic.

# For example, let's say we have 30 newspaper articles (collection), and the topics across the collection ranges from politics, to sports and to local news. We assume that each of the article (document) exhibits proportions for each topic, e.g. 80% sports, 10% politics, and 10% local news. Finally, the possible words have different probabilities for the different topics, so that "football", "nutrition", and "basketball" may have higher probabilities for the sports topic while "money", "government", and "elections" may have higher values for the politics topics.



# Finally, the inference of the topic structure is performed by reversing the generative process.



# In our case, the documents are the images, the words are the pixels p_i, their frequency are the intensity values, and we want to extract the underlying topic structure, while each topic corresponds to a number from 0 to 9.

# It is quite remarkable that this simple approach works although we only consider the intensity values of the pixels - we completely remove spatial information!



# If you are interested in topic modelling, the best way to start is to look at one of the creator's website: http://www.cs.columbia.edu/~blei/topicmodeling.html



import csv

import numpy as np

from sklearn.decomposition import LatentDirichletAllocation

from collections import Counter



###########################################################

# 1.) Training a model



# 1.1) Load training data

# Open the training data file and load it into the file_list variable. 

# The first line containing the headings is skipped.

with open('../input/train.csv') as csvfile:

    next(csvfile) # Skip header line

    file_list = [list(map(int, rec)) for rec in csv.reader(csvfile)]



# Split the training data into the image data (the image pixel values) and the labels y (the number the image describes)

data = np.array([rec[1:] for rec in file_list])

labels = [rec[0] for rec in file_list]



# 1.2) Training the model

# Initialise the LDA and learns an appropriate model

# The fit_transform method is called so that the topic predictions for the training data is returned

# It is interesting to note that we don't use the labels for the learning process - only the data! Therefore, it is a form of unsupervised learning



# One value that we need to specify is the number of topics we expect in the data - the standard value is 10, which we keep in this case (numbers from 0 to 9). This aspect is in a way cheating - we have to know the number of topics in advance! However, there is a solution for it: Nonparametric Bayesian topic modelling, where we learn the number of components from the data. This would be a nice extension of this work!

LDA = LatentDirichletAllocation(learning_method="online")

res = LDA.fit_transform(data)



# 1.3) Find the best topic id - number assignment

# In this step, we use the labels. The reason is a fundamental problem: The computer came up with a topic structure, but how should it be able to know how to call the topics? 

# However, the important part is that it learnt unsupervised - we only help with the names basically.



# 1.3.1) Collect all predicted topics for each number

predicted_topics = {}

for i, row in enumerate(res):

    predicted_topic = np.argmax(row) # Argmax returns the topic id with the highest proability

    if labels[i] in predicted_topics.keys():

        predicted_topics[labels[i]].append(predicted_topic)

    else:

        predicted_topics[labels[i]] = [predicted_topic]



# 1.3.2) Find the best assignment

assignment = {}

counters = []

# First, we collect the counts for each topic id for each number. For example, the topic id "1" might be selected 10 times to an image containing a 1, and 20 times to an image containing an 8.

for i in range(0, 10):

    counters.append(Counter(predicted_topics[i]))



# Next, for each number (0 - 9), we assign the the topic ID to it which has the highest count.

# Please note that this can be improved in many ways. A good approach to actually tackle this assignment problem could be the employment of a genetic algorithm. However, this is just a short demo how a topic model can be used for this problem, so I sticked to the most basic approach. 

for i in range(0, 10):

    num = 0

    perc = 0

    for j in range(0, 10):

        if counters[j][i] > perc:

            perc = counters[j][i]

            num = j

    assignment[i] = num



#########################################################################

# 2.) Testing



# 2.1) Import test data

with open('../input/test.csv') as csvfile:

    next(csvfile) # Skip the header line

    file_list = [list(map(int, rec)) for rec in csv.reader(csvfile)]



# Convert the data to a numpy ar

data = np.array([rec for rec in file_list])



# 2.2) Predict the topics

res = LDA.transform(data)



# 2.3) Produce string for results file and write to the file 'result.csv'

txt = 'ImageId,Label\n'

for i, row in enumerate(res):

    number = assignment[np.argmax(row)]

    txt += str(i+1) + ',' + str(number) + '\n'



with open('result.csv', 'w+') as res_file:

    res_file.write(txt)
