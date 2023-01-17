import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
%matplotlib inline
os.listdir()
folder = '../input/'
os.listdir(folder)
user = pd.read_csv('../input/User Data.csv')
y = user['Average Rating (2017)']
books2017 = user['User Read Books (2017)']
splits2017 = books2017.str.split(', ')
n2017 = books2017.str.le



n()
Y = y*n2017
from sklearn.feature_extraction.text import CountVectorizer
counter = CountVectorizer(lowercase = False, tokenizer = lambda x: x)
counter
counter.fit(splits2017)
bagwords2017 = counter.transform(splits2017)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(bagwords2017, Y)

preds = model.predict(bagwords2017)
plt.scatter(preds, Y)
X = np.hstack((
    bagwords2017.todense(),
    np.eye(len(user))
))
from scipy.sparse import csr_matrix
newX = csr_matrix(X)
model.fit(newX, Y)
preds = model.predict(newX)
unswdata@gmail.com
read2018 = user['User Read Books (2018)'].str.split(', ')
n2018 = read2018.str.len()
total = n2017 + n2018
bag2018 = counter.transform(read2018)

TEST = np.hstack((
    bag2018.todense(),
    np.eye(len(user))
))
newY = (model.predict(TEST) + Y)/total
newY[newY > 10] = 10
newY[newY < 0] = 0

pd.DataFrame({'User ID':user['User ID'],
             'Average Rating (2018)':newY})