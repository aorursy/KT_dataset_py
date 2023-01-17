import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split



#open animalData and show the first values

animalData = pd.read_csv("../input/zoo-animal-classification/zoo.csv")



animalData.head()



# split the data into train and test data

features = ["hair","feathers","eggs","milk","airborne","aquatic","predator","toothed","backbone","breathes","venomous","fins","legs","tail","domestic","catsize"]

X = animalData[features]

y = animalData.class_type



train_X, valid_X, train_y, valid_y = train_test_split(X,y,random_state=0)
# assign the model

model = DecisionTreeClassifier()

fitted = model.fit(train_X, train_y)



# make predictions

predictions = model.predict((valid_X))



print(predictions)

print(valid_y)
# match data value to animal name and class type

names = animalData.animal_name

animalClass = pd.read_csv("../input/zoo-animal-classification/class.csv")

animalClass.head(7)
num = animalClass.Class_Number

types = animalClass.Class_Type

classDict = {}



for i in range(0,len(num)):

    classDict[i + 1] = types[i];

    

print(classDict)
# create a dictionary matching animal names to values and print the results



animalDict = {}



for i in range(0,len(animalData.animal_name)):

    animalDict[i] = animalData.animal_name[i]

prediction_names = [classDict[i] for i in predictions]

animal = [animalDict[n] for n in valid_y.index]

    #print(animalDict[n],":",prediction_names)

for j in range(0,len(predictions)):

    print(animal[j],":",prediction_names[j])