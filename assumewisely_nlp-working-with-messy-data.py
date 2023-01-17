import pandas as pd

#read in the data
##!!!! this is for edu purposes, to learn how to deal with messy data. This wasn't really messy.
rawData_file = "../input/SMSSpamCollection.tsv"

dataSet =pd.read_csv(rawData_file, sep="\t", header=None)
dataSet.head(10)
import nltk
nltk.download()
#read in the data
rawData = open("../input/SMSSpamCollection.tsv").read()

#display the first 500 characters in this string of data
rawData[0:500]
#replace '\t' with '\n' and then split the data @ '\n' puts the tag and text on every other line.
parsedData = rawData.replace('\t','\n').split('\n')
parsedData[0:5]
#create new lists from parsedData[starting position:end(blank):2 indicates every other item]
labelList = parsedData[0::2]
textList = parsedData[1::2]

print(len(labelList))
print(len(textList))
#print last 5 items[start at the end and count backwards]
print(labelList[-5:])
import pandas as pd
fullCorpus = pd.DataFrame({
    'label' : labelList[:-1], ## [start at the begining: ignore the last item]
    'body_list' : textList
    
})

print (fullCorpus.head(10))
