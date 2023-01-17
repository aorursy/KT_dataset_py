import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
accidata3 = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/accidents/accidents_2012_to_2014.csv")
low_memory = False
acci = accidata3.copy()
from sklearn.preprocessing import LabelEncoder 
#encoding function:
encoder = LabelEncoder()
for col in acci:
        if acci[col].dtype == 'object':
            acci[col] = encoder.fit_transform(acci[col].astype('str'))

print(acci.head(4))
selected = ['Number_of_Vehicles', 'Light_Conditions', 'Weather_Conditions', 'Number_of_Casualties', 'Local_Authority_(District)', '1st_Road_Class', '1st_Road_Number','2nd_Road_Class','2nd_Road_Number', 'Road_Type', 'Speed_limit', 'Junction_Control', 'Road_Surface_Conditions', 'Pedestrian_Crossing-Physical_Facilities', 'Urban_or_Rural_Area']
X = acci[selected]
Y = acci['Accident_Severity']

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

logreg=LogisticRegression(max_iter = 100000)
predicted = cross_val_predict(logreg, X, Y, cv=5)
print(metrics.accuracy_score(Y, predicted))




