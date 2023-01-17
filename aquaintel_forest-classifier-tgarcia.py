# Import libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score



# Input data files 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Import input data

train = pd.read_csv("../input/learn-together/train.csv")

test = pd.read_csv("../input/learn-together/test.csv")
train.head()
train.describe()
print("Training Dimensions: %s" % str(train.shape)) 

print("Test Dimensions: %s" % str(test.shape)) 
# Drop ID in the training data

# Source: https://www.kaggle.com/kayveen/simple-notebook-to-make-a-first-submission



train = train.drop(["Id"], axis = 1)

test_ids = test["Id"]

test = test.drop(["Id"], axis = 1)

X_Test = test



# split data into training (80%) and validation data (20%)

X_Train, X_Val, Y_Train, Y_Val = train_test_split(train.drop(['Cover_Type'],axis=1), train['Cover_Type'], test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=10,criterion='entropy', random_state=42)

model.fit(X_Train, Y_Train)
# Model Score

model.score(X_Train, Y_Train)
# Prediction

predictions = model.predict(X_Val)

accuracy_score(Y_Val, predictions)
# Predictions in Test Set

Y_Test = model.predict(X_Test)
# Submision File

# Save test predictions to file

output = pd.DataFrame({'ID': test_ids,

                       'Cover_Type': Y_Test})

output.to_csv('submission.csv', index=False)