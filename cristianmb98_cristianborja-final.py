import numpy as np
import pandas as pd 
import scipy 
import matplotlib.pyplot as plt

df = pd.read_csv("../input/multipleChoiceResponses.csv", low_memory=False) #Import data

Time = np.array(df)[1:,0]
Gender = np.array(df)[1:,1]
Age = np.array(df)[1:,3]
Country = np.array(df)[1:,4]
LevEd = np.array(df)[1:,5]
Exp = np.array(df)[1:,11]
Income = np.array(df)[1:,12]
ML = np.array(df)[1:,13]

print(np.array(df)[0][48])


