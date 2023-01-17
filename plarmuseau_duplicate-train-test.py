import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder

import seaborn as sns

import matplotlib.pyplot as plt

# read datasets

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')







test['y'] = 102  # to make append possible

y_train = train["y"]

totaal=pd.DataFrame(train.append(test))



#find unique ?

kolom=train.columns 

kolom=[k for k in kolom if k not in ['ID','y']]

tot_u = train.sort_values(by='y').duplicated(subset=kolom)

print(train[tot_u==True])