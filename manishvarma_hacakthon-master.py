import numpy as np 

import pandas as pd 

import os

print(os.listdir("../input/intromlhack"))
dfTrain = pd.read_csv("../input/intromlhack/train.csv")

dfTrain.head()
dfTest = pd.read_csv("../input/intromlhack/test.csv")

dfTest.head()
#Submission:

submissionDF = pd.DataFrame({"Id": dfTest["Id"],"Demand":0})

submissionDF.to_csv('Submissionv1.csv',index=False)