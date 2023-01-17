# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from sys import stdin

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression

from matplotlib import pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



#Load Data

data = pd.read_csv("../input/kc_likes_data.csv")

data.head() # prodce a header from the first data row



#Features:

#filter

#hour

#dow - day of the week

#loc - transformed location

#article1 - heat, short, etc

#article2

#article3

#ndp - nudity detection probability

#hash1 - hashtags

#hash2

#hash3

#hash4

#hash5

#hash6

#hash7

#hash8

#hash9

#hash10







#Create arrays - targets into training

X = data["article1"].values.reshape(-1, 1)

y = data["likes"].values



# Create linear regression object

model = LinearRegression()

# Train the model using the training sets

model.fit(X, y)



predictX = model.predict(X)



plt.scatter(X, y,color='g')

plt.plot(X, predictX,color='k')

plt.show()



#Predict a specific case

d = 5

pred = model.predict(d)

print("Predicted Likes based on article1: " + str(int(pred)))