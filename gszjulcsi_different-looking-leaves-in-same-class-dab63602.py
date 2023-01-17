import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import matplotlib.image as mpimg

from sklearn.preprocessing import LabelEncoder
#Reading data

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

#tr = np.matrix(tr)

#ts = np.matrix(ts)

import glob

images_by_id ={}

for file in glob.glob('../input/images/*.jpg'):

    im = mpimg.imread(file)

    idx = int(file.replace('../input/images/', '').replace('.jpg', ''))

    images_by_id[idx] = im

def encode(train, test):

    le = LabelEncoder().fit(train.species) 

    labels = le.transform(train.species)           # encode species strings

    classes = list(le.classes_)                    # save column names for submission

    test_ids = test.id                             # save test ids for submission

    

    #train = train.drop(['species', 'id'], axis=1)  

    #test = test.drop(['id'], axis=1)

    

    return train, labels, test, test_ids, classes



train, labels, test, test_ids, classes = encode(train, test)
#Class corresponding to label = 1 is Acer_Circinatum

print(classes[1])

lab = np.where( labels == 1 )

print('Indices = ',lab[0])

ids = train['id'][lab[0]]

print('Corresponding image IDs:')

print(ids )

count = 0

for i in ids:

    plt.figure(count)

    plt.imshow( images_by_id[i] )

    count = count+1