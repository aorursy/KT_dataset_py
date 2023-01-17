# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!wget https://github.com/SayedMaheen/sg_PlanetEarth/archive/master.zip
!unzip master.zip
import random

data_dir = 'sg_PlanetEarth-master/smoke_data'

data = set()



size = 100



folder = 'smog'



upto = len(os.listdir(data_dir+'/'+folder)) - 1

print("Upto: ", upto)



while(len(data) < size):

    i = random.randint(0, upto)

    data.add(i)



len(data)
collection = []

validation = list()

# create smog first

for i in data:

    for j in os.listdir(data_dir):

        files = os.listdir(data_dir+"/"+j)

        name = files[i]

        file = data_dir+'/'+j+'/'+name

        collection.append(file)

        validation.append([name, j])



print("collection size: ", len(collection))
random.shuffle(validation)
random.shuffle(collection)
validation.insert(0, ['File name', 'Label'])

validation[:10]
import csv



with open('validation.csv', 'w') as valid_file:

    writer = csv.writer(valid_file)

    writer.writerows(validation)

    

print('validation write Complete!')
csv = pd.read_csv('validation.csv')

csv.iloc[:10]
if not os.path.exists('val'):

    os.makedirs('val')
import shutil

for i in collection:

    shutil.move(i, 'val/')
len(os.listdir('val'))
import shutil

shutil.make_archive("validation", 'zip', 'val')
os.path.exists('validation.zip')