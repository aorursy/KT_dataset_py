# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/housesalesprediction/kc_house_data.csv'):

    for filename in filenames:

        print(os.path.join(dirname, kc_house_data.csv))



# Any results you write to the current directory are saved as output.
#os.file.('/kaggle/input/kc_house_data.csv')

#Step 2 : Data import

# Use pandas to read in csv file

#os.chdir(r'C:\Users\kundanm\Desktop')

os.path.isfile('/kaggle/input/housesalesprediction/kc_house_data.csv')

#train = pd.read_excel('kc_house_data.csv')

#this is just a comment

#train.head(5)
#Step 2 : Data import

# Use pandas to read in csv file

#os.chdir(r'C:\Users\dadhict\Desktop')

#os.chdir('C:\\Users\\dadhict\\Desktop')



train = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')

#this is just a comment

train.head(10000)
train.dtypes
train.describe() # To check Mean, Mod, Std, etc.
missing_values = train.isnull()

missing_values.head(10000)
import seaborn as sns
sns.heatmap(data = missing_values, yticklabels=False, cbar=False, cmap='viridis')
output = train ['sqft_living']

input = train ['sqft_lot15']
sns.scatterplot (input, output)
g = sns.jointplot(input,output)
input= train[['sqft_above']]

output= train ['sqft_living'] 

# list is defined, so we have used single braces

#Split data into Train and test



# Import module to split dataset

from sklearn.model_selection import train_test_split

# Split data set into training and test sets

input_train, input_test, output_train, output_test = train_test_split(input, output, test_size=0.25, random_state=100)
#Checking file types created



print(input_train.shape)

#print(input_test.head())

#print(output_train.head())

print(output_test.shape)
#Run the model



# Import model for fitting

from sklearn.linear_model import LinearRegression

# Create instance (i.e. object) of LogisticRegression

#model = LogisticRegression()



#You can try follwoing variation on above model, above is just default one

model = LinearRegression()

# Fit the model using the training data

# input_train -> parameter supplies the data features

# output_train -> parameter supplies the target labels

output_model=model.fit(input_train, output_train)

#output =input_test

#output['vehicleTypeId'] = output_test

output_model

#The input X, when you try to fit/etc the model will, if the flag is set to true (default), be copied for use within the function. 

#This means that the original X you passed as a parameter will be the same after the fit/etc has been performed.If you set the flag to false, there is a chance that the X you pass in will NOT be the same as it was before you ran fit/etc.
import pickle

#Save the model in pickle

#Save to file in the current working directory

pkl_filename = "pickle_model.pkl"

with open(pkl_filename, 'wb') as file:

    pickle.dump(model, file)



# Load from file

with open(pkl_filename, 'rb') as file:

    pickle_model = pickle.load(file)



# Calculate the accuracy score and predict target values

score = pickle_model.score(input_test, output_test)

print("Test score: {0:.2f} %".format(100 * score))

outputpredict = pickle_model.predict(input_test)
df = pd.DataFrame({'Actual': output_test, 'Predicted': outputpredict.flatten()})

df
#Understanding accuracy

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

predictions = model.predict(input_test)

#print("",classification_report(y_test, predictions))

#print("confusion_matrix",confusion_matrix(y_test, predictions))

#print("accuracy_score",accuracy_score(y_test, predictions))

##**Accuracy is a classification metric. You can't use it with a regression. See the documentation for info on the various metrics.

#For regression problems you can use: R2 Score, MSE (Mean Squared Error), RMSE (Root Mean Squared Error).

#print("Score",score(y_test, X_test))

#score(self, X, y, sample_weight=None)

## setting plot style 

plt.style.use('fivethirtyeight') 

  

## plotting residual errors in training data 

plt.scatter(model.predict(input_train), model.predict(input_train) - output_train, 

            color = "green", s = 1, label = 'Train data' ,linewidth = 5) 

  

## plotting residual errors in test data 

plt.scatter(model.predict(input_test), model.predict(input_test) - output_test, 

            color = "blue", s = 1, label = 'Test data' ,linewidth = 4) 

  

## plotting line for zero residual error 

plt.hlines(y = 0, xmin = 0, xmax = 4, linewidth = 2) 

  

## plotting legend 

plt.legend(loc = 'upper right') 

  

## plot title 

plt.title("Residual errors") 

  

## function to show plot 

plt.show() 

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(output_test, outputpredict))  

print('Mean Squared Error:', metrics.mean_squared_error(output_test, outputpredict))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(output_test, outputpredict)))
# plotting regression line

ax = plt.axes()

ax.scatter(input, output)

plt.title("Input Data and regression line ") 

ax.plot(input_test, outputpredict, color ='Red')

ax.set_xlabel('input')

ax.set_ylabel('output')

ax.axis('tight')

plt.show()
# Saving output file



os.path.isfile('/kaggle/input/housesalesprediction/kc_house_data.csv')

check = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')

print('Importing data to solve for')

print(check.head(5))

outputpredict = pickle_model.predict(check[['sqft_above']])

Result=check[['sqft_above']]

Result['Y_Predicted']=outputpredict

print(Result)

Result.to_csv('RML_28_Output.csv', index=False)

print("Output saved")