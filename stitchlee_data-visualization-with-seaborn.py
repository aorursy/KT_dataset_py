#Setting up environment

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")

#Loading Data

iris_filepath = "../input/Iris.csv"

iris_data = pd.read_csv(iris_filepath, index_col="Id")

iris_data.head()
#Plotting the data

sns.scatterplot(x=iris_data['PetalLengthCm'], y=iris_data['PetalWidthCm'], hue=iris_data['Species'])
