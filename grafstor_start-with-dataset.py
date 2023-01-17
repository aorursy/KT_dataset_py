import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

def load_data(path):

    data = open(path,'r').read()

    data = data.split('\n')[:-1]

    

    x_train = []

    y_train = []

    

    for line in data:

        x, y = line.split('\t')

        

        x_train.append(x)

        y_train.append(y)

        

    return (x_train, y_train)
x_train, y_train = load_data('/kaggle/input/simple-dialogs-for-chatbot/dialogs.txt')
print(f'Question: {x_train[0]}')

print(f'Answer: {y_train[0]}')