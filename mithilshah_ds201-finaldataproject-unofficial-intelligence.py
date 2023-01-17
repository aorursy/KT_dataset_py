import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
data = pd.read_csv('../input/cereal_cleaned.csv')
data.head()
#Averages

#Calories = 107
#
A = []
B = []

for index, row in data.iterrows():
    cal = row['calories']
    protein = row['protein']
    fat = row['fat']
    sodium = row['sodium']
    fiber = row['fiber']
    carbo = row['carbo']
    sugars = row['sugars']
    potass = row['potass']
    temp_arr = [cal, protein, fat, sodium, fiber, carbo, sugars, potass]
    A.append(temp_arr)
    
print(A)
        
for rat in data.rating:
    B.append(rat)
    
#print(B)
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np

clf = tree.DecisionTreeClassifier()
clf.fit(A, B)

#This is the first row in the CSV file. The values shown below are in the following order: calories	protein	fat	sodium	fiber	carbo	sugars	potass
#Based off these input values, the ml model predicts the rating.
clf.predict([[70, 4, 1, 130, 10, 5, 6, 280]])
