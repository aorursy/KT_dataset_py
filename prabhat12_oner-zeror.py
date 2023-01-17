import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns, numpy as np
data = pd.read_csv("../input/diabetes-dataset/diabetes2.csv")

data.head()
data['Outcome'].value_counts()
sns.distplot(data['Outcome'],bins=3,kde=False)

plt.title("Analysing ZeroR")

plt.xticks([0,1])

plt.show()
# Our criteria:

# 0 = young, 1 = mid, 2 = old



column_age = []



for age in data['Age']:

    if(age < 25):

        column_age.append("0")

    elif(age>25 and age<45):

        column_age.append("1")

    else:

        column_age.append("2")



# adding a new column

data["Age_Categorical"] = column_age        
data.head()
for i in range(0,3):

    print("If Age Category: ", i, " , number of outcomes(0): ", len( data[ (data['Age_Categorical'] == str(i)) & (data['Outcome'] == 0) ]) )

    print("If Age Category: ", i, " , number of outcomes(1): ", len( data[ (data['Age_Categorical'] == str(i)) & (data['Outcome'] == 1) ] ) ,"\n")

    