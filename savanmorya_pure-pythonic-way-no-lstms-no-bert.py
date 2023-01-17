import numpy as np

import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
trainData = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")

testData = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")

trainData.head()
print(trainData['text'].isnull().sum())

trainData = trainData.dropna()

#trainData[trainData['selected_text'].isnull()] #print row with nan
trainData.shape
myDict = {} #store all the value

myDictWordCount = {} #storing all the occurences of the word

for _,row in trainData.iterrows():

    text = row['text']

    sentiment = row['sentiment']

    wt = 0 #weight penalty ! 0 for neutral

    if(sentiment == "negative"):

        wt = -1 #weight penalty -1 for negative

    elif(sentiment == "positive"):

        wt = 1 #weight penalty +1 for negative

    for word in text.split(" "):

        key = word.lower()

        if key not in myDict:

            myDict[key] = wt

            myDictWordCount[key] = 1

        else:

            myDict[key] = myDict[key] + wt

            myDictWordCount[key] = myDictWordCount[key] + 1
#checking the sentiment on test data

#calculatin the score

#plot the score for all three sentiments



testData['neutral'] = 0

testData['positive'] = 0

testData['negative'] = 0

for idx,row in testData.iterrows():

    text = row['text']

    sentiment = row['sentiment']

    score = 0

    for word in text.split(" "):

        key = word.lower()

        if key in myDict:

            score = score + myDict[key]/myDictWordCount[key]

    testData.loc[idx,sentiment] = score
testData.head()
positiveCount = 0

negativeCount = 0

neutralCount = 0

right = 0

for idx,row in testData.iterrows():

    text = row['text']

    sentiment = row['sentiment']

    score = 0

    for word in text.split(" "):

        key = word.lower()

        if key in myDict:

            score = score + myDict[key]/myDictWordCount[key]

    if((score > -0.5 and score < 0.5)):

        neutralCount = neutralCount + 1

        if(sentiment == 'neutral'):

            right = right + 1

    elif(score >= 0.5):

        positiveCount = positiveCount + 1

        if(sentiment == 'positive'):

            right = right + 1

    else:

        negativeCount = negativeCount + 1

        if(sentiment == 'negative'):

            right = right + 1

print("+: "+str(positiveCount) +" -: "+ str(negativeCount) + " *: "+str(neutralCount))

print("right: "+ str(right)+ " Score: "+ str(right*100/testData.shape[0]))
for idx,row in testData.iterrows():

    text = row['text']

    sentiment = row['sentiment']

    score = 0

    for word in text.split(" "):

        key = word.lower()

        if key in myDict:

            score = score + myDict[key]/myDictWordCount[key]

    if(sentiment == "neutral"):

        print("Weight: "+ str(score) + "Actual: "+ sentiment)
testData['selected_text'] = ""

for idx,row in testData.iterrows():

    text = row['text']

    keysdata = {}

    for key in text.split(" "):

        if key.lower() in myDict and key != ' ':

            keysdata[key.lower()] = abs(myDict[key.lower()]/myDictWordCount[key.lower()])

    ans = []

    for v,k in keysdata.items():

        if(k > 0.1):

            ans.append(v)

    testData.loc[idx,'selected_text'] = (" ".join(ans))
testData[['textID','selected_text']].to_csv("submission.csv",index=False)