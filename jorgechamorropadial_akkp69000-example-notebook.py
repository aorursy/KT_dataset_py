import os

import random

import csv
PATH_TO_RAW_COLLECTION = '/kaggle/input/akkp69000/AKKP69000/raw'

PATH_TO_FILTERED_COLLECTION = '/kaggle/input/akkp69000/AKKP69000/filtered'

PATH_TO_PRE_PROCESSED_COLLECTION = '/kaggle/input/akkp69000/AKKP69000/pre_processed'
i = 0

for dirname, _, filenames in os.walk(PATH_TO_RAW_COLLECTION):

    filename = filenames[0]       

    with open(dirname+"/"+filename) as csv_file:

        csv_reader = csv.DictReader( (line.replace('\0','') for line in csv_file), delimiter="\t" )

        for article in csv_reader:

            aks = article['DE']

            kps = article['ID']

            if len(aks) > 0 and len(kps) > 0:

                print("Article {}".format(i))

                print(article['DE'])

                print(article['ID'])

            i +=1

            if i > 50:

                break
random_article_index = random.randint(0,28800)



ak_filtered_lines = open(PATH_TO_FILTERED_COLLECTION + '/' + 'ak.txt', 'r').readlines()

ak_preprocessed_lines = open(PATH_TO_PRE_PROCESSED_COLLECTION + '/' + 'ak.txt', 'r').readlines()

kp_filtered_lines = open(PATH_TO_FILTERED_COLLECTION + '/' + 'kp.txt', 'r').readlines()

kp_preprocessed_lines = open(PATH_TO_PRE_PROCESSED_COLLECTION + '/' + 'kp.txt', 'r').readlines()



print("FILTERED COLLECTION")

print("---> AK: {}".format(ak_filtered_lines[random_article_index]))

print("---> KP: {}".format(kp_filtered_lines[random_article_index]))



print("PRE_PROCESSED COLLECTION")

print("---> AK: {}".format(ak_preprocessed_lines[random_article_index]))

print("---> KP: {}".format(kp_preprocessed_lines[random_article_index]))


