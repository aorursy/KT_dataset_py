import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



pd.options.display.max_columns = 100
train = pd.read_csv("../input/learn-together/train.csv")

test = pd.read_csv("../input/learn-together/test.csv")
train.head()
test.head()
train = train.drop(["Id"], axis = 1)



test_ids = test["Id"]

test = test.drop(["Id"], axis = 1)
print(f"Missing Values in train: {train.isna().any().any()}")

print(f"Missing Values in test: {test.isna().any().any()}")
print(f"Train Column Types: {set(train.dtypes)}")

print(f"Test Column Types: {set(test.dtypes)}")
for column in train.columns:

    print(column, train[column].nunique())
print("Soil_Type7: ", test["Soil_Type7"].nunique())

print("Soil_Type15: ", test["Soil_Type15"].nunique())
print("- - - Train - - -")

print(train["Soil_Type7"].value_counts())

print(train["Soil_Type15"].value_counts())

print("\n")

print("- - - Test - - -")

print(test["Soil_Type7"].value_counts())

print(test["Soil_Type15"].value_counts())
train = train.drop(["Soil_Type7", "Soil_Type15"], axis = 1)

test = test.drop(["Soil_Type7", "Soil_Type15"], axis = 1)
train.columns
train["Elevation"].plot(kind="hist", bins = 30)
train.plot(kind="scatter", x="Cover_Type", y="Elevation")
train["Aspect"].plot(kind="hist", bins = 30)
train.plot(kind="scatter", x="Cover_Type", y="Aspect")
sns.pairplot(train[['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',

       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',

       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',

       'Horizontal_Distance_To_Fire_Points']])
train["Slope"].plot(kind="hist", bins = 30)
sns.scatterplot(x=train["Slope"], y=train["Cover_Type"])
train["Horizontal_Distance_To_Hydrology"].plot(kind="hist", bins = 30)
train.plot(kind="scatter", x="Cover_Type", y="Horizontal_Distance_To_Hydrology")
train["Vertical_Distance_To_Hydrology"].plot(kind='hist', bins = 30)
train.plot(kind="scatter", x="Cover_Type", y="Vertical_Distance_To_Hydrology")
train["Horizontal_Distance_To_Roadways"].plot(kind='hist', bins = 30)
train.plot(kind="scatter", x="Cover_Type", y="Horizontal_Distance_To_Roadways")
train["Hillshade_9am"].plot(kind="hist", bins = 30)
train.plot(kind="scatter", x="Cover_Type", y="Hillshade_9am")
train["Hillshade_Noon"].plot(kind="hist", bins = 30)
train.plot(kind="scatter", x="Cover_Type", y="Hillshade_Noon")
train["Hillshade_3pm"].plot(kind="hist", bins = 30)
train.plot(kind="scatter", x="Cover_Type", y="Hillshade_3pm")
train["Horizontal_Distance_To_Fire_Points"].plot(kind="hist", bins = 30)
train.plot(kind="scatter", x="Cover_Type", y="Horizontal_Distance_To_Fire_Points")
sns.countplot(x="Wilderness_Area1", data=train)
sns.countplot(x="Wilderness_Area2", data=train)
sns.countplot(x="Wilderness_Area3", data=train)
sns.countplot(x="Wilderness_Area4", data=train)
res_soil_dict = {}

for col in train.columns[14:-1]:

    res_soil_dict[col] = train[col].value_counts().loc[1] 

    # .loc[1] – because in the resulting Series, the number of data points with col value == 1

    # is stored under the index 1

    # print(train[col].value_counts().loc[1])
sorted_d = sorted(res_soil_dict.items(), key=lambda kv: kv[1])
sorted_d