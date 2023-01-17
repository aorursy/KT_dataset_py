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
data = pd.read_csv("../input/tennis.csv")
data.info()
data.columns
data.head(14)
# outlook_count = data.groupby(['outlook', 'play']).size()

# outlook_total = data.groupby(['outlook']).size()

# temp_count = data.groupby(['temp', 'play']).size()

# temp_total = data.groupby(['temp']).size()

# humidity_count = data.groupby(['humidity', 'play']).size()

# humidity_total = data.groupby(['outlook']).size()

# windy_count = data.groupby(['windy', 'play']).size()

# windy_total = data.groupby(['windy']).size()

# print(outlook_count)

# print(windy_total)

# print(outlook_total)

# print(temp_count)

# print(temp_total)

# print(humidity_count)

# print(humidity_total)

# print(windy_count)

# print(windy_total)



# p_over_yes = outlook_count['overcast','yes']

# p_over_no = 0

# p_rainy_yes = outlook_count['rainy','yes']

# p_rainy_no = outlook_count['rainy','no']

# p_rainy_yes = outlook_count['sunny', 'yes']

X_train = pd.get_dummies(data[['outlook', 'temp', 'humidity', 'windy']])

y_train = pd.DataFrame(data['play'])



#assigning predictor and target variables

#x= np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])

#Y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])

print(X_train.info())

print(X_train.head())
#Import Library of Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB

import numpy as np
#Create a Gaussian Classifier

model = GaussianNB()



# Train the model using the training sets 

model.fit(X_train, y_train)



#Predict Output 

predicted= model.predict([[False,1,0,0,0,1,0,1,0]])

print (predicted)