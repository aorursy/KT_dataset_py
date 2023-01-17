import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import os

print(os.listdir())



import warnings

warnings.filterwarnings('ignore')
import pandas as pd

dataset = pd.read_csv("../input/heart.csv")
type(dataset)

dataset.shape

dataset.head(5)
dataset.sample(5)

dataset.describe()

dataset.info()

info = ["age","1: male, 0: female","chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"," maximum heart rate achieved","exercise induced angina","oldpeak = ST depression induced by exercise relative to rest","the slope of the peak exercise ST segment","number of major vessels (0-3) colored by flourosopy","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]







for i in range(len(info)):

    print(dataset.columns[i]+":\t\t\t"+info[i])
dataset["target"].describe()

dataset["target"].unique()

print(dataset.corr()["target"].abs().sort_values(ascending=False))

y = dataset["target"]



sns.countplot(y)





target_temp = dataset.target.value_counts()



print(target_temp)
print("Percentage of patience without heart problems: "+str(round(target_temp[0]*100/303,2)))

print("Percentage of patience with heart problems: "+str(round(target_temp[1]*100/303,2)))



#Alternatively,

# print("Percentage of patience with heart problems: "+str(y.where(y==1).count()*100/303))

# print("Percentage of patience with heart problems: "+str(y.where(y==0).count()*100/303))



# #Or,

# countNoDisease = len(df[df.target == 0])

# countHaveDisease = len(df[df.target == 1])
dataset["sex"].unique()

sns.barplot(dataset["sex"],y)

dataset["cp"].unique()

sns.barplot(dataset["cp"],y)

dataset["fbs"].describe()

dataset["fbs"].unique()

sns.barplot(dataset["fbs"],y)

dataset["restecg"].unique()

sns.barplot(dataset["restecg"],y)

dataset["exang"].unique()

sns.barplot(dataset["exang"],y)

dataset["slope"].unique()

sns.barplot(dataset["slope"],y)
