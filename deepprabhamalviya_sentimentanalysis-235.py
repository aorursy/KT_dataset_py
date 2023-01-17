# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import warnings

dt = pd.read_csv("../input/sentiment-train/sentiment_train.csv")

dt
dt.head()
#So, lets begin. Today

#Sir had mentioned a few points that we needed to understand and show him

#Is it readable ?yes. ok :-)
# Add Dataset sentiment  - DONE
#FInd the number of positive and negative sentiments

#Create a counting plot with the SEABORN library functions
#Create Bag of Word Modelling with Count Vectorizer
#This is a very important area, and I have understood the concept from Sir
#The remaning points will be done once these are completed. Ok ?:-) 
#ok, to shuru korchi.
#FInd the number of positive and negative sentiments

#Create a counting plot with the SEABORN library functions
#see, for this we will import a few libraries from Seaborn
import seaborn as sns

import matplotlib.pyplot as plt

sns.set(color_codes = True)  #ok, this step is optional, ok
dt.head()  #I need to check this data for reference
#Lets shape the data so that we get a different plotting point

dt.shape
#Now, is the part that you had discussed with Sir and we listened.

#Differenciating the Sentiment Counts - Positive and Negative
pos_sentiment_count = len(dt[dt.label == 1])

#Remember, sir had told us that for each word (and later, sentence)

#We had to scan and change them to 0s and 1s, Yes. So this is the part.
neg_sentiment_count = len(dt[dt.label == 0])

#Same for negative. And I think 1 and 0 can be either of them, as long as the

#graph index is ok.
print(pos_sentiment_count ,neg_sentiment_count,sep="\n" ) 
#ok
#Lets plot this using matplot lib now
plt.figure(figsize=(6,5))

ax = sns.countplot(x='label',data = dt)
#First Initial Part regarding DATA Handling is done uptill this

#Now we being the most important part (that is) the Count Vectorizer



#ok



#So what it does, is it scans every sentence, and every common word it takes and adds to a list

#For example see



# 1) You are my friend
# 2) They are my friends



#Now, this count vectorizer function, if I have understood correctly,

#would work like this



# You are my friend They friends

# It counts whatever word is unique and gives a numerical representationhaan

#ok so lets try this
# oh ho, I am such a stupid

# I have not imported the sklearn library

#nope, forgot



from sklearn.feature_extraction.text import CountVectorizer
#Lets call the function

count_vectorizer = CountVectorizer() #<--- This is the function



feature_vector = count_vectorizer.fit(dt.sentence) #one second let me check run
#Works hopefully, lets print them values



features = feature_vector.get_feature_names()

#Sorry, it was a spelling error
#Now, the important part, lets see how many actual Letters were found and how many unique

#words we could find from the dataset

#I think I will remove this line, looks confusing here, lets make it cleaner
dt_features = count_vectorizer.transform(dt.sentence)





print ("Highlighted Words : ",(dt_features.shape))

print ("Features : ",len(features))
#Ok, so this is the duplicate filtering section.ok
#Now, to plot a graph we need 3 things

#1) X and Y values - sir hsa guided with a matrix to dataframe concept

#2) Setting the column names to each feature extracted for comparison

#3) Guiding the program to proceed with the rest by giving one TRAINING TEST DATA so that it can understand

# what to do with the next upcoming data set information



#SEE
#Matrix to Dataframe



train_ds_dt = pd.DataFrame(dt_features.todense())  # I asked sir what this todense() function does, so it generates a matrix with 0s and 1s depending on feature values. Yes.ok
print(train_ds_dt)
#Sorry typo again.

#So this is the priority-non-priority map for analysis the data.

#Lets work on it.
train_ds_dt.columns = features #just a declaration for the interpreter to understand
#LEts print the indexes as a test case
print(dt[4:12])

print(train_ds_dt.iloc[4:12,204:210])

print(train_ds_dt[['brokeback','mountain','is','such','horrible','movie']][0:1])
#This is the data for all the features computed.

#Next step is to remove the non useful feature words