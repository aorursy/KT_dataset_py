# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



import random



dataset = []

labels = []



with open("../input/Iris.csv") as f:

    data = f.readlines()[1:]



    random.shuffle(data)



    for i in range(0, len(data)):

        formatted = data[i].strip('\n')

        formatted_list = formatted.split(",")

        

        label = formatted_list[-1]

        labels.append(label)

        values = formatted_list[:len(formatted_list)-1]

        

        for j in range(len(values)):

            values[j] = float(values[j])

        dataset.append(values)
label_names = []



for i in range(len(labels)):

    label = labels[i]

    if label not in label_names:

        label_names.append(label)



    labels[i] = label_names.index(label)
# Separate data into train and test

train_count = int(0.8 * len(dataset))

train, test = dataset[0:train_count], dataset[train_count:]

train_labels, test_labels = labels[0:train_count], labels[train_count:]
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()

clf.fit(train, train_labels)

predictions = clf.predict(test)
correct = 0

total = len(predictions)



for i in range(total):

    if predictions[i] == test_labels[i]:

        correct += 1



print(float(correct)/total)