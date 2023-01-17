import csv

with open('/kaggle/input/urdu-ngrams/eins-s.csv/eins-s.csv', encoding = 'utf-8') as csvfile:

    data = list(csv.reader(csvfile, delimiter='\t'))

print("Count of unigrams: ", len(data))

print("First 50 unigrams:\n", data[0:50])
import csv

with open('/kaggle/input/urdu-ngrams/vier-s.csv/vier-s.csv', encoding = 'utf-8') as csvfile:

    data = list(csv.reader(csvfile, delimiter='\t'))

print("Count of 4grams: ", len(data))

print("First 50 4grams:\n", data[0:50])