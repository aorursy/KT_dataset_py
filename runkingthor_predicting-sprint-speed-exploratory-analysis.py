# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import scipy.stats  as stats

import os

fifa_dataset = pd.read_csv("../input/data.csv")

#fifa_dataset['Overall Range'] =  fifa_dataset['Overall'].apply(groups)

fifa_dataset['Overall Range'] = fifa_dataset.Overall.map(lambda x: '40-49' if x in range(40,49) 

                                                         else ('50-59' if x in range(50,59) 

                                                               else ('60-69' if x in range (60,69) 

                                                                     else('70-79' if x in range(70,79) 

                                                                          else('80-89' if x in range(80,89) 

                                                                               else'90-94')))))

#visualizing overall range distribution by age

fifa_dataset.groupby(['Age','Overall Range']).size().unstack().plot.bar(stacked=True)



#for easier visualization of speed distribution by age

fifa_dataset['Speed Range'] =  fifa_dataset.SprintSpeed.map(lambda x: '10-19' if x in range(10,19) 

                                                         else ('20-29' if x in range(50,59) 

                                                               else ('30-39' if x in range (60,69) 

                                                                     else('40-49' if x in range(70,79) 

                                                                          else('50-59' if x in range(80,89) 

                                                                               else('60-69' if x in range(60,69)

                                                                                   else('70-79' if x in range(70,79)

                                                                                       else('80-89' if x in range(80,89)

                                                                                           else '90-97'))))))))

#visualizing speed range distribution by age

fifa_dataset.groupby(['Age','Speed Range']).size().unstack().plot.bar(stacked=True)



#converting height and weight from string to int

fifa_dataset['Height']= fifa_dataset.Height.str.split("'").str.join('.').apply(lambda x: float(x)*30.48).dropna()

fifa_dataset['Height']= fifa_dataset['Height'].fillna(fifa_dataset['Height'].mean()).astype(np.int64)

fifa_dataset['Weight'] = fifa_dataset.Weight.str.replace("lbs", "").apply(lambda x: float(x)*0.45359237).dropna()

fifa_dataset['Weight']  = fifa_dataset['Weight'].fillna(fifa_dataset['Weight'].mean()).astype(np.int64)



#visualizing speed range distribution by weight

fifa_dataset.groupby(['Weight','Speed Range']).size().unstack().plot.bar(stacked=True)



#visualizing speed range distribution by height

fifa_dataset.groupby(['Height','Speed Range']).size().unstack().plot.bar(stacked=True)



#converting relevant columns to ints

def func(x):

  x = x.fillna(x.mean()).astype(np.int64)

  return x

fifa_dataset[['Agility','Acceleration','Balance','Positioning','Skill Moves','BallControl','Crossing','Finishing','Reactions','SprintSpeed']] = func(fifa_dataset[['Agility','Acceleration','Balance','Positioning','Skill Moves','BallControl','Crossing','Finishing','Reactions','SprintSpeed']])



#Testing for moderate to strong correlation with sprintspeed

def corr_test(x):

  x_corr = stats.spearmanr(x, fifa_dataset['SprintSpeed'])

  return x_corr



#Multivariable Linear Regression

#80/20 split- 20% training data

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score

train, test = train_test_split(fifa_dataset, test_size=0.2)



#independent and dependent variables

features= ['Agility', 'Acceleration', 'Balance','Reactions','Positioning','Skill Moves','BallControl','Crossing','Finishing']

target = 'SprintSpeed'



#model we are using

model = LinearRegression()



#training process

model.fit(train[features], train[target])

model.fit(test[features], test[target])



#mean absolute value for training data

data = train[target]

predict =  model.predict(train[features])

training_error = mean_absolute_error(data, predict)



#mean absolute value for test data

test_data = test[target]

predict_test = model.predict(test[features])

test_data_error = mean_absolute_error(test_data, predict_test)



#we need some metric to measure the accuracy of our regression model

from sklearn.metrics import r2_score



#on training data

true_value = train[target]

predicted_val =  model.predict(train[features])

accuracy = r2_score(true_value, predicted_val)



#on test data

true_value2 = test[target]

predicted_val2 =  model.predict(test[features])

accuracy2 = r2_score(true_value2, predicted_val2)



print('This model accounts for {}% of the training data with mean data error of {}'.format(round(accuracy2*100,2), round(training_error,2)))

print('This model accounts for {}% of the testing data with mean data error of {}'.format(round(accuracy*100,2), round(test_data_error,2)))





# Any results you write to the current directory are saved as output.