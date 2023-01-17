import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
iris_filepath = "../input/IRIS.csv"

iris_data = pd.read_csv(iris_filepath)

iris_data.head()
sns.scatterplot(x=iris_data['sepal_length'], y=iris_data['sepal_width'])
sns.scatterplot(x=iris_data['petal_length'], y=iris_data['petal_width'])
sns.scatterplot(x=iris_data['sepal_length'], y=iris_data['petal_length'])
sns.scatterplot(x=iris_data['sepal_width'], y=iris_data['petal_width'])