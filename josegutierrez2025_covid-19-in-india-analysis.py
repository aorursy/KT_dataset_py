import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline
data = pd.read_csv("../input/covid19-in-india/AgeGroupDetails.csv")

data2 = pd.read_csv("../input/covid19-in-india/IndividualDetails.csv")
print(data.columns)
print(data[['AgeGroup','Percentage']])
AgeGroups=['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80+','unknown']

Percentage=[3.18,3.9,24.86,21.1,16.18,11.13,12.86,4.05,1.45,1.30]
ypos = np.arange(len(AgeGroups))

print(ypos)
plt.xticks(ypos,AgeGroups)

plt.ylabel("Percentages")

plt.xlabel("Age Groups")

plt.title("Covid-19 Cases in India by Age Group")

plt.bar(ypos,Percentage)

plt.show()
plt.xticks(ypos,AgeGroups)

plt.ylabel("Percentages")

plt.xlabel("Age Groups")

plt.title("Covid-19 Cases in India by Age Group")

plt.plot(ypos,Percentage)

plt.show()
print(data2.shape)

data2
print(pd.value_counts(data2['current_status']))
case_status=['Deceased','Recovered','Active'] 

number_of_cases=[46,181,27662]
yposition = np.arange(len(case_status))

print(yposition)
plt.xticks(yposition,case_status)

plt.ylabel("Number of Cases")

plt.xlabel("Case Status")

plt.title("The Status of Covid-19 Cases in India")

plt.bar(yposition,number_of_cases)

plt.show()
case_status2=['Deceased','Recovered'] 

number_of_cases2=[46,181]
yposition2 = np.arange(len(case_status2))

print(yposition2)
plt.xticks(yposition2,case_status2)

plt.ylabel("Number of Cases")

plt.xlabel("Case Status")

plt.title("The Status of Covid-19 Cases in India")

plt.bar(yposition2,number_of_cases2)

plt.show()