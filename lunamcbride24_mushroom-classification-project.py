# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split #Split the data into training and testing

from sklearn.tree import DecisionTreeClassifier #Get a tree to train



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
shroom = pd.read_csv("../input/mushroom-classification/mushrooms.csv") #Read the mushroom dataset

shroom.head() #Show the mushrooms
print(shroom.isnull().any()) #Check for any nulls
poisonous = shroom["class"].copy() #Get the characteristic we are looking for into one variable

characteristics = shroom.drop("class", axis = 1) #Get every other characteristic into another variable

characteristics.head() #Take a peek at every other characteristic
isPoison = pd.get_dummies(poisonous) #Changes the categorical poison or edible into 1 and 0 depending on the edibility

character = pd.get_dummies(characteristics) #Changes the character data into 1 and 0 for each characteristic

print(isPoison) #Prints isPoison to show the dummy structure
charaTrain, charaTest, poiTrain, poiTest = train_test_split(character, isPoison) #Splits the data into train and test sets
model = DecisionTreeClassifier() #Load our SVC

model.fit(charaTrain, poiTrain) #Train it to the training data
poiTest.reset_index(drop=True, inplace=True) #Reset the poiTest index for iteration

poison = [] #A list used to hold poiTest data in the form [[e,p]]

score = 0 #A score count for the model predictions

predict = model.predict(charaTest) #Use the model to predict edibility



#For loop to put poiTest into the poison list in the format [[e,p]], which is how the predictions come out

for i in range(0,len(poiTest)):

    poison.append([poiTest["e"][i], poiTest["p"][i]]) #Append edibility status into the poison list

    

predictLength = len(predict) #Move the length calculation so it is not calculated every time the loop is run

    

#For loop to compare the poisonous status to the predicted poisonous status

for i in range(0 , predictLength):

    if predict[i][0] ==  poison[i][0]: #If the prediction and actual have the same starting value, thus the same overall value

        score += 1 #Add one to the score



print("This tree model predicted {} out of {} correctly, which gives an accuracy of {}%".format(score, predictLength, int(score/predictLength * 100))) #Print the score