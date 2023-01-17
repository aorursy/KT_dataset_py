# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import spacy # NLP
from sklearn.svm import LinearSVC
import re # regular expressions
import html # HTML content, like &amp;
from spacy.lang.en.stop_words import STOP_WORDS # stopwords
from sklearn.model_selection import train_test_split # training and testing a model
from spacy.util import minibatch # batches for training
import random # randomizing for training

nlp = spacy.load('en_core_web_lg') #Load spacy, up here so I do not have to load it constantly

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("../input/covid-19-nlp-text-classification/Corona_NLP_train.csv", encoding = "ISO-8859-1") #Load the training set
train.head() #Take a peek at the training set
test = pd.read_csv("../input/covid-19-nlp-text-classification/Corona_NLP_test.csv", encoding = "ISO-8859-1") #Load the testing set
test.head() #Take a peek at the testing set
# Check for nulls in all columns in Train
print("Train CSV: \n")
print(train["UserName"].isnull().any())
print(train["ScreenName"].isnull().any())
print(train["Location"].isnull().any())
print(train["TweetAt"].isnull().any())
print(train["OriginalTweet"].isnull().any())
print(train["Sentiment"].isnull().any())
    
#Location has a null
train["Location"] = train["Location"].fillna("Unknown") #Fill the null values with "Unknown"
print("Location: ", train["Location"].isnull().any(), "\n") #Print the now fixed location to make sure it is truly fixed

# Check for nulls in all columns in Test
print("Test CSV: \n")
print(test["UserName"].isnull().any())
print(test["ScreenName"].isnull().any())
print(test["Location"].isnull().any())
print(test["TweetAt"].isnull().any())
print(test["OriginalTweet"].isnull().any())
print(test["Sentiment"].isnull().any())

#Location has a null
test["Location"] = test["Location"].fillna("Unknown") #Fill the null values with "Unknown"
print("Location: ", test["Location"].isnull().any(), "\n") #Print the now fixed location to make sure it is truly fixed
empty = train["OriginalTweet"].apply(lambda x: print("One") if not x else x) #Prints "One" if there are any empty strings
empty2 = test["OriginalTweet"].apply(lambda x: print("One") if not x else x) #Prints "One" if there are any empty strings
punctuations = """!()-![]{};:+'"\,<>./?@#$%^&*_~Â""" #List of punctuations to remove, including a weird A that will not process out any other way

#CleanTweets: parces the tweets and removes punctuation, stop words, digits, and links.
#Input: the list of tweets that need parsing
#Output: the parsed tweets
def cleanTweets(tweetParse):
    for i in range(0,len(tweetParse)):
        tweet = tweetParse[i] #Putting the tweet into a variable so that it is not calling tweetParse[i] over and over
        tweet = html.unescape(tweet) #Removes leftover HTML elements, such as &amp;
        tweet = re.sub(r"@\w+", ' ', tweet) #Completely removes @'s, as other peoples' usernames mean nothing
        tweet = re.sub(r'https\S+', ' ', tweet) #Removes links, as links provide no data in tweet analysis in themselves
        tweet = re.sub(r"\d+\S+", ' ', tweet) #Removes numbers, as well as cases like the "th" in "14th"
        tweet = ''.join([punc for punc in tweet if not punc in punctuations]) #Removes the punctuation defined above
        tweet = tweet.lower() #Turning the tweets lowercase real quick for later use
    
        tweetWord = tweet.split() #Splits the tweet into individual words
        tweetParse[i] = ''.join([word + " " for word in tweetWord if nlp.vocab[word].is_stop == False]) #Checks if the words are stop words
        
    return tweetParse #Returns the parsed tweets

#Jeez, this whole NLP project (plus the kaggle course) has thrown a lot of use of making a list via _ for _ if _

trainCopy = train["OriginalTweet"].copy() #Copies the train tweets, using a copy to ensure I do not screw it up
testCopy = test["OriginalTweet"].copy() #Copies the test tweets, using a copy to ensure I do not screw it up

trainTweets = cleanTweets(trainCopy) #Calls the cleanTweets method to clean the train tweets
testTweets = cleanTweets(testCopy) #Calls the cleanTweets method to clean the test tweets

train["CleanTweet"] = trainTweets #Puts the clean train tweets into a new column
test["CleanTweet"] = testTweets #Puts the clean test tweets into a new column
train.head() #Take a peek at the new addition to the data
print(trainTweets.loc[trainTweets == ""], "\n \n") #Print the row numbers with empty clean train tweets
print(testTweets.loc[testTweets == ""]) #Print the row number with empty clean test tweets
#RemoveBlanks: removes tweets that became blank after processing
#Input: the dataframe to look at
#Output: none
def removeBlanks(df):
    df["CleanTweet"] = df["CleanTweet"].apply(lambda x: np.nan if not x else x) #Changes blank strings to nan
    df.dropna(subset = ["CleanTweet"], inplace = True) #Drops the rows newly assigned to nan
    df.reset_index(drop=True, inplace=True) #Reset indecies so we can still loop through without error

removeBlanks(train) #Removes the blanks from the train set
removeBlanks(test) #Removes the blanks from the test set
train.head() #Opens up the train to take a peek, as the first one was blank in the training set
print(train["CleanTweet"].loc[train["CleanTweet"] == ""], "\n \n") #Print the row number that still has empty clean train tweets
print(test["CleanTweet"].loc[test["CleanTweet"] == ""]) #Print the row number that still has empty clean test tweets
# Sentiments: A function to turn the word sentiments into numerical values for the Train set, 0, 1, 2, 0 being negative, 2 being positive.
# This function also makes incorrect values in labels -1, as nothing else is -1
def sentiments(x):
    if x == "Negative":
        return 0
    if x == "Neutral":
        return 1
    return 2

def removeExtremes(x):
    if x == "Extremely Negative":
        return "Negative"
    if x == "Extremely Positive":
        return "Positive"
    return x

#Extremes were causing problems in the model, as it is hard to exemplify extreme to a computer
#These change the extremes to just their counterparts so it is not a necessary hurdle
train["Sentiment"] = train["Sentiment"].apply(removeExtremes)
test["Sentiment"] = test["Sentiment"].apply(removeExtremes)

train["NumSentiment"] = train["Sentiment"].apply(sentiments) #Add a row into train for numerical sentiment
test["NumSentiment"] = test["Sentiment"].apply(sentiments) #Add a row into test for numerical sentiment
test.head() #Display the test and see if it has numerical sentiment
#Pipe for processing, copied from the kaggle course
textcat = nlp.create_pipe(
              "textcat",
              config={
                "exclusive_classes": True,
                "architecture": "bow"})
try:
    nlp.add_pipe(textcat) #Add the pipe
    print("Pipeline loaded") #Print for if the pipeline is loaded
except:
    nlp.remove_pipe("textcat") #delete the pipe to reload
    nlp.add_pipe(textcat) #Add the pipe
    print("Pipeline now loaded") #Print for if the pipeline is loaded

#Adding labels for the tweets
textcat.add_label("Negative")
textcat.add_label("Neutral")
textcat.add_label("Positive")
#TrainData: a function to train the model to the train data. Modeled after the one in the kaggle class
#Input: the model, the training data, and an optimizer
#Output: losses
def trainData(model, data, optimize):
    losses = {} #A set for the losses data
    random.seed() #Randomizing the seed of shuffling data
    random.shuffle(data) #Shuffles the data
    
    batches = minibatch(data, size=10) #Creates batches of texts
    
    #For each batch of texts
    for batch in batches:
        text, label = zip(*batch) #Unzip the labels and text
        model.update(text, label, sgd = optimize, losses = losses) #Update the model with the new data
    
    return losses #Return the losses
#PredictTexts: predicts the sentiment of the tweet, from negative to positive
#Input: the model and the tweets
#Output: predictions
def predictTexts(model, texts):
    predicText = [model.tokenizer(text) for text in texts] #Tokenizes the test tweets
    model.get_pipe("textcat") #Gets the trained textcat pipe
    scores,_ = textcat.predict(predicText) #Gets the scores from the predictions, ignoring other outputs
    classes = scores.argmax(axis = 1) #Get the highest ranked prediction score for each tweet
    return classes #Returns the predictions
#CheckAccuracy: checks the accuracy compared to the predictions.
#Input: the NLP model, the tweets to predict, their pre-determined labels
#Output: the accuracy of the predictions
def checkAccuracy(model, texts, labels):
    predicted = predictTexts(model, texts) #Creates predictions on the tweets
    trueVal = [2*int(label["cats"]["Positive"]) + int(label["cats"]["Neutral"]) for label in labels] #Gets the actual value of the tweets provided
    correct = 0 #A holder variable for how many predictions are correct
    total = len(predicted) #The total number of analyzed tweets
    
    #For loop, comparing predictions to their values
    for i in range(0,total):
        if trueVal[i] == predicted[i]: #If the prediction is correct
            correct+=1  #Add a point to the correct pile
    
    accuracy = correct/total #Get the accuracy of the number correct over the number total
    return accuracy #Returns the accuracy of the model
labels = [] #Labels for the cleaned training tweet
labelsT = [] #The labels for the cleaned test tweet

#For loop to add true and false to classifications for the train set
for i in range(0,len(train)): 
    label = train["Sentiment"][i] #Get the sentiment
    
    #Categorize true false based on the labels
    if label == "Negative":
        cats = {"Negative" : True, "Neutral" : False, "Positive" : False}
    elif label == "Neutral":
        cats = {"Negative" : False, "Neutral" : True, "Positive" : False}
    else:
        cats = {"Negative" : False, "Neutral" : False, "Positive" : True}
    labels.append({'cats' : cats})

#For loop to add true and false to classifications for the test set
for i in range(0,len(test)):
    label = test["Sentiment"][i] #Get the sentiment
    
    #Categorize true false based on the labels
    if label == "Negative":
        cats = {"Negative" : True, "Neutral" : False, "Positive" : False}
    elif label == "Neutral":
        cats = {"Negative" : False, "Neutral" : True, "Positive" : False}
    else:
        cats = {"Negative" : False, "Neutral" : False, "Positive" : True}
    labelsT.append({'cats' : cats})

texts = train["CleanTweet"].copy() #Get the clean tweets
tokenTexts = [nlp.tokenizer(tweet) for tweet in texts] #Tokenize the training tweets
optimize = nlp.begin_training() #The optimizer, using spacy
data = list(zip(tokenTexts, labels)) #Zipping the labels and texts together

losses = trainData(nlp, data, optimize) #Train the model
accuracy = checkAccuracy(nlp, test["CleanTweet"].copy(), labelsT) #Gets the accuracy of predictions for the trained model
print("Losses: ", losses["textcat"], "Accuracy: ", accuracy) #Prints the loss when training and the accuracy