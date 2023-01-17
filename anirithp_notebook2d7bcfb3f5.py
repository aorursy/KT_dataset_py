import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
accidata1 = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/accidents/accidents_2005_to_2007.csv")
low_memory = False
accidata2 = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/accidents/accidents_2009_to_2011.csv")
low_memory = False
accidata3 = pd.read_csv("../input/iiitb-ai511ml2020-assignment-1/Assignment/accidents/accidents_2012_to_2014.csv")
low_memory = False
accidata=pd.concat([accidata1,accidata2,accidata3])
reqd_cols = ['Number_of_Vehicles', 'Number_of_Casualties', 'Local_Authority_(District)', 'Local_Authority_(Highway)', '1st_Road_Class', '1st_Road_Number', 'Road_Type', 'Speed_limit', 'Junction_Control','2nd_Road_Class','2nd_Road_Number', 'Light_Conditions', 'Weather_Conditions', 'Road_Surface_Conditions', 'Special_Conditions_at_Site','Carriageway_Hazards', 'Urban_or_Rural_Area']
x = accidata3.copy()
for colname in x:
    if x[colname].dtype == 'object':
        x[colname] = x[colname].astype('category')
        x[colname] = x[colname].cat.codes
x.head()
X = x[reqd_cols]
Y = x['Accident_Severity']

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn import metrics
from sklearn import datasets
logreg=LogisticRegression(max_iter = 100000)
#logreg.fit(features_train,labels_train)
features_train, features_test, labels_train, labels_test = train_test_split(X, Y, test_size=0.1, random_state=42)
predicted = cross_val_predict(logreg, X, Y, cv=5)
print(metrics.accuracy_score(Y, predicted))



