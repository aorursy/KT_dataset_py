import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs

column = train["label"]

xaxis = [0,1,2,3,4,5,6,7,8,9]
yaxis = [0,0,0,0,0,0,0,0,0,0]

for each in column:
    if each == 0:
        yaxis[0]=yaxis[0]+1
    if each == 1:
        yaxis[1]=yaxis[1]+1
    if each == 2:
        yaxis[2]=yaxis[2]+1    
    if each == 3:
        yaxis[3]=yaxis[3]+1    
    if each == 4:
        yaxis[4]=yaxis[4]+1    
    if each == 5:
        yaxis[5]=yaxis[5]+1    
    if each == 6:
        yaxis[6]=yaxis[6]+1    
    if each == 7:
        yaxis[7]=yaxis[7]+1   
    if each == 8:
        yaxis[8]=yaxis[8]+1
    if each == 9:
        yaxis[9]=yaxis[9]+1
        
plt.plot(xaxis,yaxis)
plt.ylabel('numbers')
plt.show()

# Import the random forest package
from sklearn.ensemble import RandomForestClassifier 

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train,test)

# Take the same decision trees and run it on the test data
output = forest.predict(test)


