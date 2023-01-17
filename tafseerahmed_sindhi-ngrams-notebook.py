import csv

with open('/kaggle/input/sindhi-ngrams/n1-s.csv/n1-s.csv', encoding = 'utf-8') as csvfile:

    data = list(csv.reader(csvfile, delimiter='\t'))

print("Count of unigrams: ", len(data))

print("First 50 unigrams:\n", data[0:50])

import csv

with open('/kaggle/input/sindhi-ngrams/n2-s.csv/n2-s.csv', encoding = 'utf-8') as csvfile:

    data = list(csv.reader(csvfile, delimiter='\t'))

print("Count of bigrams: ", len(data))

print("First 50 bigrams:\n", data[0:50])