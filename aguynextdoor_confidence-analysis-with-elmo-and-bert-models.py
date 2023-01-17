import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import string

import re

from nltk.tokenize import word_tokenize

import json

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Utility function to handle the two classes' probability outputs of ELMo's model. 

def return_low(x):

    high = x["1"] - 1

    if(x["0"] > x["1"]): return (1-x["0"])

    else : return x["1"]
#Loading predictions from ELMo and BERT models.

#Loading training and test datasets.

import pandas as pd



test = pd.read_csv("../input/predictions-datasets/test.csv")

train = pd.read_csv("../input/predictions-datasets/train.csv")
def clean_tweets(tweet):

    """Removes links and non-ASCII characters"""

    

    tweet = ''.join([x for x in tweet if x in string.printable])

    

    # Removing URLs

    tweet = re.sub(r"http\S+", "", tweet)

    

    return tweet



def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



def remove_punctuations(text):

    punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"

    

    for p in punctuations:

        text = text.replace(p, f' {p} ')



    text = text.replace('...', ' ... ')

    

    if '...' not in text:

        text = text.replace('..', ' ... ')

    

    return text



def convert_abbrev(word):

    return abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word



def convert_abbrev_in_text(text):

    tokens = word_tokenize(text)

    tokens = [convert_abbrev(word) for word in tokens]

    text = ' '.join(tokens)

    return text
ids_with_target_error = [328,443,513,2619,3640,3900,4342,5781,6552,6554,6570,6701,6702,6729,6861,7226]

train.at[train['id'].isin(ids_with_target_error),'target'] = 0

train[train['id'].isin(ids_with_target_error)]



train = train.drop(train[train["text"].duplicated()].index)



with open('../input/abbreviation/abbreviation.json') as json_file:

    abbreviations = json.load(json_file)



train["text"] = train["text"].apply(lambda x: clean_tweets(x))

test["text"] = test["text"].apply(lambda x: clean_tweets(x))



train["text"] = train["text"].apply(lambda x: remove_emoji(x))

test["text"] = test["text"].apply(lambda x: remove_emoji(x))



train["text"] = train["text"].apply(lambda x: remove_punctuations(x))

test["text"] = test["text"].apply(lambda x: remove_punctuations(x))



train["text"] = train["text"].apply(lambda x: convert_abbrev_in_text(x))

test["text"] = test["text"].apply(lambda x: convert_abbrev_in_text(x))

ELMo_full_proba = pd.read_csv("../input/predictions-datasets/ELMo_full_proba.csv")[["0", "1"]]

elmo_full = ELMo_full_proba.apply(return_low, axis = 1)

elmo_full_around = elmo_full.apply(lambda x: np.int(np.around(x)))



bert_full = pd.read_csv("../input/predictions-datasets/newTrainPredict.csv")["0"]

bert_full_around = bert_full.apply(lambda x: np.around(x))



bert_test = pd.read_csv("../input/predictions-datasets/newTestPredict.csv")["0"]

bert_test_around = bert_test.apply(lambda x: np.around(x))
Predict = bert_full_around

Predict.index = train.index

bert_evaluationOnes = (Predict.loc[Predict == 1] == train.loc[Predict == 1]["target"])

bert_evaluationZeros = (Predict.loc[Predict == 0] == train.loc[Predict == 0]["target"])



tot_false = (bert_evaluationOnes[bert_evaluationOnes == False]).shape[0] + (bert_evaluationZeros[bert_evaluationZeros == False]).shape[0]

bert_acc = 1-tot_false/train.index.shape[0]

print("accuracy is ", bert_acc)



FP_bert = train.loc[bert_evaluationOnes[bert_evaluationOnes == False].index]

FN_bert = train.loc[bert_evaluationZeros[bert_evaluationZeros == False].index]

print(np.shape(FP_bert), np.shape(FN_bert))
Predict = elmo_full_around

Predict.index = train.index

elmo_evaluationOnes = (Predict.loc[Predict == 1] == train.loc[Predict == 1]["target"])

elmo_evaluationZeros = (Predict.loc[Predict == 0] == train.loc[Predict == 0]["target"])



tot_false = (elmo_evaluationOnes[elmo_evaluationOnes == False]).shape[0] + (elmo_evaluationZeros[elmo_evaluationZeros == False]).shape[0]

elmo_acc = 1-tot_false/train.index.shape[0]

print("accuracy is ", elmo_acc)



FP_elmo = train.loc[elmo_evaluationOnes[elmo_evaluationOnes == False].index]

FN_elmo = train.loc[elmo_evaluationZeros[elmo_evaluationZeros == False].index]

print(np.shape(FP_elmo), np.shape(FN_elmo))
combined_pd = pd.DataFrame({'bert': bert_full, 'elmo':elmo_full})

combined_pd = combined_pd.apply(lambda x: np.average(x), axis = 1)

combined_around_pd = combined_pd.apply(lambda x: np.int(np.around(x)))



Predict = combined_around_pd

Predict.index = train.index

combined_evaluationOnes = (Predict.loc[Predict == 1] == train.loc[Predict == 1]["target"])

combined_evaluationZeros = (Predict.loc[Predict == 0] == train.loc[Predict == 0]["target"])



tot_false = (combined_evaluationOnes[combined_evaluationOnes == False]).shape[0] + (combined_evaluationZeros[combined_evaluationZeros == False]).shape[0]



FP_comb = train.loc[combined_evaluationOnes[combined_evaluationOnes == False].index]

FN_comb = train.loc[combined_evaluationZeros[combined_evaluationZeros == False].index]



comb_acc = 1-tot_false/train.index.shape[0]

print("accuracy is ", comb_acc)

print(np.shape(FP_comb), np.shape(FN_comb))

fig = plt.figure(figsize = (30,10))



ax = fig.add_subplot(131)

ax.axis([0, 10, 0, 18])

plt.xticks(size = 20)

plt.yticks(size = 15)

plt.title('FP_elmo')

FP_elmo["keyword"].value_counts().head(10).plot.bar(ax = ax)



ax = fig.add_subplot(132)

ax.axis([0, 10, 0, 18])

plt.xticks(size = 20)

plt.yticks(size = 15)

plt.title('FP_bert')

FP_bert["keyword"].value_counts().head(10).plot.bar(ax = ax)



ax = fig.add_subplot(133)

ax.axis([0, 10, 0, 18])

plt.xticks(size = 20)

plt.yticks(size = 15)

plt.title('FP_comb')

FP_comb["keyword"].value_counts().head(10).plot.bar(ax = ax)
#Don't hesitate to test different keywords such as [detonate, pandemonium, tsunami, ...]

keyword = "windstorm"



for idx in FP_bert.index:  

    if(FP_bert["keyword"][idx] == keyword):

        print(FP_bert["id"][idx], FP_bert["keyword"][idx], " : ")

        print(FP_bert["text"][idx])

        print('--'*20)
train[train["keyword"] == "windstorm"]["target"].value_counts()
#Don't hesitate to test different keywords such as [eyewitnessed, destroyed, demolition, ...]

keyword = "trapped"



for idx in FP_elmo.index:  

    if(FP_elmo["keyword"][idx] == keyword):

        print(FP_elmo["id"][idx], FP_elmo["keyword"][idx], " : ")

        print(FP_elmo["text"][idx])

        print('--'*20)

        

print(train[train["keyword"] == keyword]["target"].value_counts())
keyword = "windstorm"



for idx in FP_elmo.index:  

    if(FP_elmo["keyword"][idx] == keyword):

        print(FP_elmo["id"][idx], FP_elmo["keyword"][idx], " : ")

        print(FP_elmo["text"][idx])

        print('--'*20)
keyword = "windstorm"



for idx in FP_comb.index:  

    if(FP_comb["keyword"][idx] == keyword):

        print(FP_comb["id"][idx], FP_comb["keyword"][idx], " : ")

        print(FP_comb["text"][idx])

        print('--'*20)
fig = plt.figure(figsize = (30,10))



ax = fig.add_subplot(131)

ax.axis([0, 10, 0, 12])

plt.title('FN_elmo')

plt.xticks(size = 20)

plt.yticks(size = 15)

FN_elmo["keyword"].value_counts().head(10).plot.bar(ax = ax)



ax = fig.add_subplot(132)

ax.axis([0, 10, 0, 12])

plt.xticks(size = 20)

plt.yticks(size = 15)

plt.title('FN_bert')

FN_bert["keyword"].value_counts().head(10).plot.bar(ax = ax)



ax = fig.add_subplot(133)

ax.axis([0, 10, 0, 12])

plt.xticks(size = 20)

plt.yticks(size = 15)

plt.title('FN_comb')

FN_comb["keyword"].value_counts().head(10).plot.bar(ax = ax)
def confidenceCalc(x):

    if x<0.5 : 

        return (0.5-x)*2

    else : 

        return (x-0.5)*2
Predict = elmo_full_around

Predict.index = train.index

elmo_full.index = train.index

TrueValues = elmo_full[train["target"] == Predict]

FalseValues = elmo_full[train["target"] != Predict]

meanTrueConf = TrueValues.apply(confidenceCalc).mean()

meanFalseConf = FalseValues.apply(confidenceCalc).mean()



print("ELMo's accuracy :", elmo_acc)

print("True confidence :", meanTrueConf, "False confidence :", meanFalseConf)

print("Value diff :", meanTrueConf - meanFalseConf)
Predict = bert_full_around

Predict.index = train.index

bert_full.index = train.index

TrueValues = bert_full[train["target"] == Predict]

FalseValues = bert_full[train["target"] != Predict]

meanTrueConf = TrueValues.apply(confidenceCalc).mean()

meanFalseConf = FalseValues.apply(confidenceCalc).mean()



print("BERT's accuracy :", bert_acc)

print("True confidence :", meanTrueConf, "False confidence :", meanFalseConf)

print("Value diff :", meanTrueConf - meanFalseConf)
Predict = combined_around_pd

Predict.index = train.index

combined_pd.index = train.index

TrueValues = combined_pd[train["target"] == Predict]

FalseValues = combined_pd[train["target"] != Predict]

meanTrueConf = TrueValues.apply(confidenceCalc).mean()

meanFalseConf = FalseValues.apply(confidenceCalc).mean()



print("combined's accuracy :", comb_acc)

print("True confidence :", meanTrueConf, "False confidence :", meanFalseConf)

print("Value diff :", meanTrueConf - meanFalseConf)
submission_combined = pd.read_csv("../input/submission/submission.csv")
submission_combined.to_csv('/kaggle/working/submission.csv', index=False)