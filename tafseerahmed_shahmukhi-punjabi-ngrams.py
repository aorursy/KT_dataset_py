import csv

with open('/kaggle/input/punjabi-shahmukhi-ngrams/unigram-p.csv/unigram-p.csv', encoding = 'utf-8') as csvfile:

    data = list(csv.reader(csvfile, delimiter='\t'))



print("Count of unigrams: ", len(data))

print("First 50 unigrams:\n", data[0:50])

import csv

with open('/kaggle/input/punjabi-shahmukhi-ngrams/bigram-p.csv/bigram-p.csv', encoding = 'utf-8') as csvfile:

    data = list(csv.reader(csvfile, delimiter='\t'))



print("Count of bigrams: ", len(data))

print("First 50 bigrams:\n", data[0:50])