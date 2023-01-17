import pandas as pd
import numpy as np
data = pd.read_csv('../input/spam.csv',encoding="latin-1")
total = 0
num_spam = 0
trainPositive = {}
trainNegative = {}
positiveTotal = 0
negativeTotal = 0
for sms in data.values:
    if sms[0] == "spam":
        num_spam+=1
    total+=1
    for word in sms[1]:
        if sms[0] == "ham":
            trainPositive[word] = trainPositive.get(word,0) + 1
            positiveTotal+=1
        else:
            trainNegative[word] = trainNegative.get(word,0) + 1
            negativeTotal+=1
p_spam = num_spam/float(total)
p_ham = (total - num_spam)/float(total)
def Word(word, spam):
    if spam:
        return trainPositive[word]/float(positiveTotal)
    return trainNegative[word]/float(negativeTotal)
def SMS(body, spam):
    result = 1.0
    for word in body:
        r = Word(word, spam)
        if(r>0):
            result *= r
    return result
def classify(sms):
    isSpam = p_spam * SMS(sms, True) # P (A | B)
    notSpam = p_ham * SMS(sms, False) # P(¬A | B)
    if(isSpam > notSpam):
        print("Spam")
    else:
        print("Not Spam")
classify("Random Text")

