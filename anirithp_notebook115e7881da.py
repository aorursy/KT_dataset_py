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
plt.figure(figsize=(35,35))
sns.heatmap(accidata.corr(), cmap="coolwarm", annot=True)
correlated_features = set()
correlation_matrix = accidata.corr()

for i in range(len(correlation_matrix .columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) >= 0.9:
            colname = correlation_matrix.columns[j]
            correlated_features.add(colname)
print(correlated_features)
selected = ['Number_of_Vehicles', 'Light_Conditions', 'Weather_Conditions', 'Number_of_Casualties', 'Local_Authority_(District)', '1st_Road_Class', '1st_Road_Number','2nd_Road_Class','2nd_Road_Number', 'Road_Type', 'Speed_limit', 'Junction_Control', 'Road_Surface_Conditions', 'Pedestrian_Crossing-Physical_Facilities', 'Urban_or_Rural_Area']
Accident = accidata['Accident_Severity']
Number = accidata['Number_of_Vehicles']
correlation = Accident.corr(Number)
print('Number of vehicles correlation: ', correlation)
print(len(selected))
for trait in selected:
    acci = pd.DataFrame(accidata.groupby(trait)['Accident_Severity'].mean())
    weatherconditions = list(acci.index)
    accident = list(acci.Accident_Severity)
    x = plt.figure(figsize = (10,10))
    plotting = x.add_axes([0,0,1,1])
    plotting.bar(weatherconditions, accident)
plt.show()