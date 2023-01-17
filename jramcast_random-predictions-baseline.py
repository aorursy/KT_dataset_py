

import os

import csv

import random

import pandas as pd



TRAIN_FILEPATH ="../input/train_curated.csv"

TEST_FOLDER = "../input/test"

def get_classes():

    data = pd.read_csv(TRAIN_FILEPATH)

    labels = set()

    for labels_per_example in data["labels"].tolist():

        for label in labels_per_example.split(","):

            labels.add(label)



    labels_list = list(labels)

    labels_list.sort()

    return labels_list





def discover_test_filenames(datadir):

    for root, dirs, files in os.walk(datadir):

        for filename in files:

            yield os.path.join(datadir, filename), filename



labels = get_classes()





with open('submission.csv', 'w') as csvfile:

    filewriter = csv.writer(csvfile)

    filewriter.writerow(['fname'] + labels)



    for full_filename, filename in discover_test_filenames(TEST_FOLDER):

        row = [filename]

        for label in labels:

            row.append(random.uniform(0, 1))

        filewriter.writerow(row)
