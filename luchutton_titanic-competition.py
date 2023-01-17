# Setup

import numpy as np

import pandas as pd

from randomforest import RandomForest

# Get Training Data

trainPath = "../input/titanic/train.csv";

trainData = pd.read_csv(trainPath);
# Create Forest

forest = RandomForest(trainData, "Survived", 5, ["PassengerId", "Name", "Ticket", "Fare", "Embarked", "Cabin"], 3);
testPath = "../input/titanic/test.csv";

testData = pd.read_csv(testPath);



predictions = forest.predict(testData);



output = pd.DataFrame({"PassengerId": testData.PassengerId, "Survived": predictions});

output.to_csv("submission.csv", index=False);