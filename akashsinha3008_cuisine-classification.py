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
#importing training dataset
training_dataset = pd.read_json('../input/train.json')
print(training_dataset.shape)
training_dataset.head()
#importing test dataset
test_dataset = pd.read_json('../input/test.json')
print(test_dataset.shape)
test_dataset.head()

#Checking if ay value is null in training set
sum(training_dataset['ingredients'].isnull())
#checking the number of unique cuisines
cuisines = training_dataset['cuisine'].unique()
print(cuisines.shape)
#function to extract food items from ingredients
def extract_items(ingredients):
    food_items = []
    for items in ingredients:
        for item in items:
            if item in food_items:
                pass
            elif item not in food_items:
                food_items.append(item)
            else:
                pass
    return food_items
#calling the function
ingredients = extract_items(training_dataset['ingredients'])
#count of unique ingredients
print(len(ingredients))
#Add each ingredient in training dataset with 0
for ingredient in ingredients:
    training_dataset[ingredient] = 0
#Add each ingredient in test dataset with 0
for ingredient in ingredients:
    test_dataset[ingredient] = 0
#Encoding the categorical values
#for ingredient present in cuisine value as 1 else 0
def find_item(ingredients_list, dataset):
    position = 0
    for items in ingredients_list:
        for item in items:
            if item in ingredients:
                dataset.loc[position , item] = 1
            else:
                pass
        position = position + 1
#Calling function for training dataset
find_item(training_dataset['ingredients'],training_dataset)
#Calling function for test dataset
find_item(test_dataset['ingredients'],test_dataset)
# Define X (Predictors)
#All encoded ingredients
X = training_dataset[ingredients]
#Define y (dependent variable)
y = training_dataset['cuisine']
y.head()
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
#Predicted result
y_pred = classifier.predict(X_test)
#Accuracy score
from sklearn.metrics import *
accuracy_score(y_test,y_pred)
#confusion matrix
cm = confusion_matrix(y_test,y_pred)
cm
#prediction on test data set
y_final_pred = classifier.predict(test_dataset[ingredients])

output = test_dataset['id']
output = pd.DataFrame(output)
output['cuisine'] = y_final_pred

output.to_csv('submission.csv',index=False)
