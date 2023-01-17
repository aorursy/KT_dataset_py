# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import time



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DATA = "/kaggle/input/"

class Poisson:

    def __init__(self, eps=1e-15, alpha=1e-12, iterations=1e7):

        self.eps=eps

        self.theta=0

        self.alpha = alpha

        self.iterations = iterations

        self.h = lambda x: np.exp(x)

        

    def norm(self, x):

        ans=0

        for i in x: ans = ans+(i*i)

        return np.sqrt(ans)

    

    def fit(self, x, y):

        start = time.time()

        m,n = x.shape

        self.theta = np.zeros(n)

        for i in range(int(self.iterations)):

            theta = self.theta

            tt = self.alpha*(y - self.h(x.dot(theta))).dot(x)

            self.theta = theta + tt

            if self.norm(tt)<self.eps: 

                print(i+1, "Iterations in", time.time()-start, "seconds")

                break

                

    def predict(self, x):

        return self.h(x.dot(self.theta))
train_df = pd.read_csv(DATA+"ds4_train.csv")

train_df['x_0'] = [1 for _ in range(len(train_df))]

x, y = train_df.drop(['y'], axis=1), train_df['y']

x = train_df[['x_0', 'x_1', 'x_2', 'x_3', 'x_4']]

x = np.array(x)

y = np.array(y)



model = Poisson()

model.fit(x,y)



test_df = pd.read_csv(DATA+"ds4_valid.csv")

test_df['x_0'] = [1 for _ in range(len(test_df))]

x_test, y_test = test_df.drop(['y'], axis=1), test_df['y']

x_test = test_df[['x_0', 'x_1', 'x_2', 'x_3', 'x_4']]

x_test = np.array(x_test)

y_test = np.array(y_test)

pred = model.predict(x_test)



model.theta
cnt = 0

for i in range(len(pred)):

    if abs(pred[i]-y_test[i])>1000: cnt+=1

cnt, len(pred)