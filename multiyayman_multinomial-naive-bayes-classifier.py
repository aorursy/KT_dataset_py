import csv

import numpy as np

import re

import pandas as pd

import math

import nltk

from nltk.corpus import stopwords

from collections import Counter

#The following line has to run if you have not used nltk before

#nltk.download()
def pre_process_text_csv(data, encoding="utf-8"):

    #processing data and putting into a pandas dataframe

    print("processing data")

    regex = re.compile('[^a-zA-Z ]')

    processed_data = {"tweet":[], "class":[]}

    line_count = 0

    

    with open(data, encoding = "utf-8") as csv_file:

        #opening the csv

        csv_reader = csv.reader(csv_file, delimiter=",")

        classes = []

        for row in csv_reader:

            if line_count > 1:

                if row[1] not in classes:

                    classes.append(row[1])

                    line_count +=1

                    continue

            #filtering the tweets with regex and removing stopwords

            reviewsentence = remove_stopwords(regex.sub("",row[10].encode('ascii', 'ignore').decode().lower()))

            #Filtering the tweets without removing the stopwords will make the program run faster, 

            #but will make the accuracy more inconsistent

            #reviewsentence = regex.sub("",row[10].encode('ascii', 'ignore').decode().lower()))

            #appending tweets and classes to the pandas dataframe

            processed_data["tweet"].append(reviewsentence)

            processed_data["class"].append(row[1])



                

            line_count +=1

        

        #converting the data into a pandas Data frame

        pandas_processed_data = pd.DataFrame(processed_data)

        #splitting the data randomly into a test and training set with np.random.rand

        data_split = np.random.rand(len(pandas_processed_data)) < 0.8

        train_set = pandas_processed_data[data_split]

        test_set = pandas_processed_data[~data_split]

        #returning the list of classes and the train and test sets



    

    return train_set, test_set, classes
def make_vocab(

    data):

    

    # Create an empty dictionary

    allwords = set()

    for tweet in data:

         for word in tweet.split():

            allwords.add(word)

    return allwords

      
def add_negation(data):

    #using regex to add negations to tweets

    new_string = re.sub(r'(?:not|never|no|n\'t)[\w\s]+[^\w\s]', 

  		lambda match: re.sub(r'(\s+)(\w+)', r'\1not_\2', match.group(0)), 

			data,

      flags=re.IGNORECASE)

    

    return new_string
def remove_stopwords(data):

    #removing stopwords from tweets

    stop_words = set(stopwords.words('english'))

    

    filtered_sentence = []

    for word in data.split():

        #if a word is not in stop_words vocab and not 

        # in airlines the word will be removed from the tweets

        if word not in stop_words:

            filtered_sentence.append(word)



    return " ".join(filtered_sentence)
def train_bayes(data, Classes):

    #this function trains the classifier

    print("training classifier")

    #using the make_vocab function make a vocabulary of words

    vocab = make_vocab(data["tweet"])

    

    all_docs = len(data["tweet"])

    all_word_likelihoods = {}

    c_priors = {}

    #looping for the three classes

    for c in Classes:

        #finding all tweet with a certain class

        class_doc = data[data["class"] == c]

        #finding the prior probabilities of the classes

        c_prior = len(class_doc)/all_docs

        c_priors[c] = math.log(c_prior)

        #finding how many times a word is used in the given class

        word_freq_count = pd.Series(' '.join(class_doc["tweet"]).split()).value_counts()

        #finding the all the words used in all tweets in the given class

        words_total_count = len(' '.join(class_doc["tweet"].tolist()).split())

        

        for word in vocab:

            #looping through the words in the vocabulary to find the probabillities of the words

            if word not in all_word_likelihoods:

                all_word_likelihoods[word] = {}

            #if a word is not used in the given class the value is set to zero

            if word not in word_freq_count:

                word_freq = 0

            else:

                #finding the frequency of the given word

                word_freq = word_freq_count[word]

                

            #finding the probabillity of the given word

            word_likelihood = math.log((word_freq + 1)/(words_total_count + len(vocab)))

            #adding the word to a dictionary where the word is the key and the class and probabillity is the values

            all_word_likelihoods[word].update({c:word_likelihood})



    return all_word_likelihoods, c_priors, vocab
def test_bayes(testdata, log_prior, 

                log_likelihood, Classes, Vocab):

    #testing the classifier

   

    print("classifiying tweets")

    classified_tweets = {}

    final_preds = {"tweet":[], "class":[]}

    #looping through all classes

    for c in Classes:

        #finding the priors for the given class

        sum_c = log_prior[c]

        for tweet in testdata["tweet"]:

            if tweet not in classified_tweets:

                classified_tweets[tweet] = {}

            tweet_class_prob = []

            for word in tweet.split():

                if word in Vocab:

                    #putting the probabilities of the words into the a tweet_class_prob list

                    log_likelihood[word][c]

                    tweet_class_prob.append(log_likelihood[word][c])

            #adding the prior and the sum tweet_class_prob list of the words for the given class

            classified_tweets[tweet].update({c:sum_c + np.sum(tweet_class_prob)})

            



    for tweet in testdata["tweet"]:

        #finds the class with the highest probability for every tweet and lables the tweets accordingly

        final_preds["tweet"].append(tweet)

        final_preds["class"].append(Classes[np.argmax([classified_tweets[tweet]["positive"],classified_tweets[tweet]["neutral"], classified_tweets[tweet]["negative"]])])

        

    return pd.DataFrame(final_preds)

    ["positive", "negative", "neutral"]
def accuracy_score(label, target):

        #finding the accuracy score of the classifier by comparing the test set

        #and the data returned from the test_bayes function

        compare = []

        #compares the labels in the test set with the new labels

        for i in range(0,len(label)):

            if label.iloc[i] == target.iloc[i]:

                temp ='correct'

                compare.append(temp)

            else:

                temp ='incorrect'

                compare.append(temp)

        comparison = Counter(compare)

        accuracy = comparison['correct']/(comparison['correct']+comparison['incorrect'])

        return f"accuracy score: {accuracy * 100}%"

    
def take_tweet(C,vocab, 

               log_prior, log_likelihood, user_inp):

    #taking a tweet and classifying it 

    regex = re.compile('[^a-zA-Z ]')

    filtered_input = regex.sub("",user_inp.lower())

    word_probs = {}

    class_probs = []

    #the same process from the test bayes function is used here, 

    #except it has benn modified for one tweet

    for c in C:

        sum_c = log_prior[c]

        word_class_prob = []

        for word in filtered_input.split():

            if word not in word_probs:

                word_probs[word] = []

            if word in vocab:

                word_class_prob.append(log_likelihood[word][c])

                if (c,log_likelihood[word][c]) not in word_probs[word]:

                    word_probs[word].append((c,log_likelihood[word][c]))

            else:

                continue

        class_probs.append(sum_c + np.sum(word_class_prob))

        word_class_prob = []

            

            

    #returns the label with the highest probabillity, and word_probabillities

    return C[np.argmax(class_probs)], word_probs
def explanation_generator(word_probs, log_prior, vocab):

    #returns an explanation of why a tweet has been labeled with a class

    classes = ["positive", "neutral", "negative"]

    pos_prob = []

    neg_prob = []

    neu_prob = []

    #this loop prints the word probabilities for all classes

    for word in word_probs.keys():

        #finding the probabillities

        if word in vocab:

            print("P(" 

            + word 

            + "|positive) = " 

            + str(word_probs[word][0][1])

            + " | P(" 

            + word 

            + "|neutral) = " 

            + str(word_probs[word][1][1])

            + " | P(" 

            + word 

            + "|negative) = " 

            + str(word_probs[word][2][1])

            + "\n")

            pos_prob.append(word_probs[word][0][1])

            neu_prob.append(word_probs[word][1][1])

            neg_prob.append(word_probs[word][2][1])



    #summing all the probabillities to show the probabillities of all classes

    print("P(positive) + P(tweet|positive) = " 

        + str(log_prior["positive"]) 

        + " + " 

        + str(sum(pos_prob)) 

        + " = " 

        , log_prior["positive"] 

        + sum(pos_prob),"\n")

    print("P(negative) + P(tweet|negative) = " 

        + str(log_prior["negative"]) 

        + " + " 

        + str(sum(neg_prob)) 

        + " = "

        , log_prior["negative"] 

        + sum(neg_prob),"\n")

    print("P(neutral) + P(tweet|neutral) = " 

        + str(log_prior["neutral"]) 

        + " + " 

        + str(sum(neu_prob))

        + " = "

        , log_prior["neutral"] 

        +sum(neu_prob), "\n")

        

    all_probs = [log_prior["positive"] 

                 + sum(pos_prob),log_prior["neutral"] 

                 + sum(neu_prob), log_prior["negative"] 

                 + sum(neg_prob)]

    #returns the highest probabillity for the tweet

    return "this tweet is labeled " + classes[np.argmax(all_probs)] +" because the probabillity is highest for the " + classes[np.argmax(all_probs)] + " label."
train_set, test_set, classes = pre_process_text_csv("../input/twitter-airline-sentiment/Tweets.csv")

    



log_likelihood, log_priors, vocab = train_bayes(train_set, classes)



final_preds = test_bayes(test_set,log_priors,log_likelihood,classes,vocab)



print(accuracy_score(test_set["class"],final_preds["class"]))
#explanation for correctly labled tweet

    

tweet_label, word_probs = take_tweet(

classes, vocab, log_priors, 

log_likelihood,

"@VirginAmerica is anyone doing anything there today?  Website is useless and no one is answering the phone.")

print("label given by classifier: " + tweet_label + "\n")



print(explanation_generator(word_probs, log_priors, vocab))
#explanation for the second correctly labled tweet

tweet_label, word_probs = take_tweet(

classes, vocab, log_priors, 

log_likelihood,

"@VirginAmerica is the best airline I have flown on.Easy to change your reservation,helpful representatives a comfortable flying experience")

print("label given by classifier: " + tweet_label + "\n")

print(explanation_generator(word_probs, log_priors, vocab))
#explanation for wrongly labeled tweet

#this tweet is orginally classified as neutral

tweet_label, word_probs = take_tweet(

    classes, vocab, 

    log_priors, log_likelihood,

    "@VirginAmerica Can you find us a flt out of LAX that is sooner than midnight on Monday? That would be great customer service ðŸ˜ƒ")

print("label given by classifier: " + tweet_label + "\n")

print(explanation_generator(word_probs, log_priors,vocab))
#second wrongly labeled tweet

#originally labeled positive

tweet_label, word_probs = take_tweet(

    classes, vocab, 

    log_priors, log_likelihood,

    "@VirginAmerica  Flight Booking Problems last second flight for next week from SFO- to SAN any chance you want to gift me a promo code since I love you guys")

print("label given by classifier: " + tweet_label + "\n")

print(explanation_generator(word_probs, log_priors,vocab))