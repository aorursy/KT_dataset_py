import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#columns = ["ID","diagnosis","radius","texture","perimeter","area","smoothness","compactness","concavity","concave points","symmtery","fractal dimension",""]

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",sep=",",header=None)

df.columns = ["ID","Diagnosis","radius","texture","perimeter","area","smoothness","compactness","concavity","concave_points","symmtery","fractal_dimension",

              "radius_se", "texture_se","perimeter_se","area_se","smoothness_se","compactness_se","concavity_se","concave_point_se","symmetry_se","fractal_dimension_se",

              "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst","compactness_worst","concavity_worst","concave_point_worst","symmetry_worst","fractal_dimension_worst"]

df.head()

#Separating dependent and independent variables

drop_cols = ["Diagnosis","ID"]

X = df.drop(drop_cols, axis=1)

y = df["Diagnosis"]

X.head()
y.head()
# To know if the dataset is imbalance or not using visualizations

ax = sns.countplot(y,label = "count")   # countplot counts the no.of values of different classes in a column

B, M = y.value_counts()

print("Number of Benign Tumors :", B)

print("Number of Malignant Tumors :", M)
X.describe()   # descriptive statistics
data = X

data_standard = (data - data.mean())/data.std()    # standardizing the data
data = pd.concat([y, data_standard.iloc[:, 0:10]], axis=1) # Creating a new dataframe with only 10 features

data = pd.melt(data, id_vars = "Diagnosis", var_name = "features", value_name= "value") # Making the data from in to right format and structure for plotting

plt.figure(figsize=(10,10))  # Using matplotlib to set the figure size

sns.violinplot(x ="features", y ="value", hue="Diagnosis",data= data, split= True, inner= "quart")

plt.xticks(rotation=45);
data = pd.concat([y, data_standard.iloc[:, 10:20]], axis=1) # Creating a new dataframe with only 10 features

data = pd.melt(data, id_vars = "Diagnosis", var_name = "features", value_name= "value") # Making the data from in to right format and structure for plotting

plt.figure(figsize=(10,10))  # Using matplotlib to set the figure size

sns.violinplot(x ="features", y ="value", hue="Diagnosis",data= data, split= True, inner= "quart")

plt.xticks(rotation=45);
data = pd.concat([y, data_standard.iloc[:, 20:30]], axis=1) # Creating a new dataframe with only 10 features

data = pd.melt(data, id_vars = "Diagnosis", var_name = "features", value_name= "value") # Making the data from in to right format and structure for plotting

plt.figure(figsize=(10,10))  # Using matplotlib to set the figure size

sns.violinplot(x ="features", y ="value", hue="Diagnosis",data= data, split= True, inner= "quart")

plt.xticks(rotation=45);
# Box plots helps in identifying the outliers.

data = pd.concat([y, data_standard.iloc[:, 0:10]], axis=1) # Creating a new dataframe with only 10 features

data = pd.melt(data, id_vars = "Diagnosis", var_name = "features", value_name= "value")

plt.figure(figsize=(10,10))

sns.boxplot(x ="features", y="value", hue="Diagnosis",data=data)

plt.xticks(rotation=45);
data = pd.concat([y, data_standard.iloc[:, 10:20]], axis=1) # Creating a new dataframe with only 10 features

data = pd.melt(data, id_vars = "Diagnosis", var_name = "features", value_name= "value")

plt.figure(figsize=(10,10))

sns.boxplot(x ="features", y="value", hue="Diagnosis",data=data)

plt.xticks(rotation=45);
data = pd.concat([y, data_standard.iloc[:, 20:30]], axis=1) # Creating a new dataframe with only 10 features

data = pd.melt(data, id_vars = "Diagnosis", var_name = "features", value_name= "value")

plt.figure(figsize=(10,10))

sns.boxplot(x ="features", y="value", hue="Diagnosis",data=data)

plt.xticks(rotation=45);
sns.jointplot(X.loc[:, "concavity_worst"], 

              X.loc[:, "concave_point_worst"], 

              kind="regg",

              color= "#ce1414"

              );

# Swarm plot will show us all the data points while stacking up with the similar values

# It helps in observing the distribution of the values 

sns.set(style="whitegrid", palette="muted")

data = X

data_standard = (data - data.mean())/data.std()

data = pd.concat([y, data_standard.iloc[:, 0:10]], axis=1)

data = pd.melt(data, id_vars="Diagnosis",

               var_name="features",

               value_name="value")

plt.figure(figsize=(10,10))

sns.swarmplot(x="features", y="value", hue="Diagnosis", data=data)

plt.xticks(rotation=45);
sns.set(style="whitegrid", palette="muted")

data = X

data_standard = (data - data.mean())/data.std()

data = pd.concat([y, data_standard.iloc[:, 10:20]], axis=1)

data = pd.melt(data, id_vars="Diagnosis",

               var_name="features",

               value_name="value")

plt.figure(figsize=(10,10))

sns.swarmplot(x="features", y="value", hue="Diagnosis", data=data)

plt.xticks(rotation=45);
sns.set(style="whitegrid", palette="muted")

data = X

data_standard = (data - data.mean())/data.std()

data = pd.concat([y, data_standard.iloc[:, 20:30]], axis=1)

data = pd.melt(data, id_vars="Diagnosis",

               var_name="features",

               value_name="value")

plt.figure(figsize=(10,10))

sns.swarmplot(x="features", y="value", hue="Diagnosis", data=data)

plt.xticks(rotation=45);
f, ax = plt.subplots(figsize=(20,20))

sns.heatmap(X.corr(), annot=True, linewidth= .5, fmt ='.0%', ax=ax);
