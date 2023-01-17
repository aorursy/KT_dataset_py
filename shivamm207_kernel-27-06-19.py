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
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn import preprocessing

from sklearn.metrics import mean_absolute_error, mean_squared_error
# Load the data_frame from the source file.



data_frame = pd.read_csv('../input/insurance.csv')

x = [i for i in range(1338)]

y = data_frame['expenses'].values

plt.scatter(x, y)

plt.show()
# Scattered plot wasn't much helpful for data visualisation. To better visualise the data, data is sorted based on 

# its expense values. 



y = np.sort(y)

plt.scatter(x, y)

plt.show()
# On sorting the data, it is noticed that the expense in this data-set is a combination of two linear curves.

# One linear curve with slope_1 to the expense value of around 15,000. And the other linear curve with a greater 

# slope than the previous one.

# So, while predicting, we first predict the expense values with the help of linear_model_1, which is based on the

# training data for expenses less than 15,000. If the prediction from this model surpasses 15,000 we update our 

# prediction with linear_model_2, which is trained on the data-set having expense values greater than 15000.

value_1 = 15000



# Segregating data_frames on the basis of expenses.

data_frame_1 = data_frame[data_frame['expenses'] < value_1]

data_frame_2 = data_frame[data_frame['expenses'] > value_1]



x1 = data_frame_1[['age', 'sex', 'bmi', 'children', 'smoker', 'region']].values

y1 = data_frame_1['expenses'].values



x2 = data_frame_2[['age', 'sex', 'bmi', 'children', 'smoker', 'region']].values

y2 = data_frame_2['expenses'].values

# changing classification variables, to dummy variables.

le_sex = preprocessing.LabelEncoder()

le_sex.fit(['female', 'male'])

x1[:, 1] = le_sex.transform(x1[:, 1])

x2[:, 1] = le_sex.transform(x2[:, 1])



le_smoker = preprocessing.LabelEncoder()

le_smoker.fit(['no', 'yes'])

x1[:, 4] = le_smoker.transform(x1[:, 4])

x2[:, 4] = le_smoker.transform(x2[:, 4])



le_region = preprocessing.LabelEncoder()

le_region.fit(['southwest', 'southeast', 'northwest', 'northeast'])

x1[:, 5] = le_region.transform(x1[:, 5])

x2[:, 5] = le_region.transform(x2[:, 5])
# Creating training and testing data sets for expenses < 15000 and expenses > 15000.

train_x1, test_x1, train_y1, test_y1 = train_test_split(x1, y1, test_size=0.2, random_state=4)

train_x2, test_x2, train_y2, test_y2 = train_test_split(x2, y2, test_size=0.2, random_state=4)
# adding the two test data-sets, so as to use our model to predict any random value in the test-dataset.

# similiar thing is done with the training data-set.

test_data_set = np.vstack((test_x1, test_x2))

actual_test_results = np.hstack((test_y1, test_y2))

actual_test_results = actual_test_results.transpose()



train_data_set = np.vstack((train_x1, train_x2))

actual_train_results = np.hstack((train_y1, train_y2))

actual_train_results = actual_train_results.transpose()
# model_1 based on data-set having expenses < 15000

model_1 = linear_model.LinearRegression()

model_1.fit(train_x1, train_y1)



# model_2 based on data-set having expenses > 15000

model_2 = linear_model.LinearRegression()

model_2.fit(train_x2, train_y2)
# Prediction of test data-set.

prediction_array = np.empty([len(test_data_set), 1])

for i1 in range(len(test_data_set)):

    predict = model_1.predict([test_data_set[i1]])

    if predict > 15000:

        predict = model_2.predict([test_data_set[i1]])



    prediction_array[i1, 0] = predict
# Prediction of train data-set.

prediction_array_train = np.empty([len(train_data_set), 1])

for i2 in range(len(train_data_set)):

    predict_train = model_1.predict(([train_data_set[i2]]))

    if predict_train > 15000:

        predict_train = model_2.predict([train_data_set[i2]])



    prediction_array_train[i2, 0] = predict_train
# mean absolute and mean square error for predicted and actual values for test data-set.

test_MSE = mean_squared_error(actual_test_results, prediction_array)

test_MAE = mean_absolute_error(actual_test_results, prediction_array)



print('test_MSE', test_MSE,'     ' ,'test_MAE', test_MAE)
# mean absolute and mean square error for predicted and actual values for train data-set.

train_MSE = mean_squared_error(actual_train_results, prediction_array_train)

train_MAE = mean_absolute_error(actual_train_results, prediction_array_train)

print('train_MSE', train_MSE,'     ','train_MAE', train_MAE)

# Plot for predicted and actual values of test set, so as to visualise the error in the predicted values.

print("Visualization")



x_plot = [j for j in range(len(actual_test_results))]

plt.plot(x_plot, actual_test_results, 'r', x_plot, prediction_array, 'g')

plt.show()
# From the graph, it can be noted that the prediction works extremely well for the cases having expenses less than 

# 15000.

# The errors for the prediction are mostly significant for the samples having expense values > 60000.
# This is to find out about the reliability of the prediction model. Percentage of predictions in the test data-set 

# having an absolute error greater than the acceptable error limit (taken to be 1000 here) is calculated.



acceptable_error = 1000

count = 0

for i in range(len(actual_test_results)):

    error = actual_test_results[i] - prediction_array[i]

    if abs(error) > acceptable_error:

        count = count+1

percentage = (count*100)/1338

percentage = format(percentage, '.2f')



print("Percentage error","   ",percentage, "%")