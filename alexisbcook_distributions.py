

import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Path of the file to read

iris_filepath = "../input/iris.csv"



# Read the file into a variable iris_data

iris_data = pd.read_csv(iris_filepath, index_col="Id")



# Print the first 5 rows of the data

iris_data.head()
# Histogram 

sns.distplot(a=iris_data['Petal Length (cm)'], kde=False)
# KDE plot 

sns.kdeplot(data=iris_data['Petal Length (cm)'], shade=True)
# 2D KDE plot

sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind="kde")
# Paths of the files to read

iris_set_filepath = "../input/iris_setosa.csv"

iris_ver_filepath = "../input/iris_versicolor.csv"

iris_vir_filepath = "../input/iris_virginica.csv"



# Read the files into variables 

iris_set_data = pd.read_csv(iris_set_filepath, index_col="Id")

iris_ver_data = pd.read_csv(iris_ver_filepath, index_col="Id")

iris_vir_data = pd.read_csv(iris_vir_filepath, index_col="Id")



# Print the first 5 rows of the Iris versicolor data

iris_ver_data.head()
# Histograms for each species

sns.distplot(a=iris_set_data['Petal Length (cm)'], label="Iris-setosa", kde=False)

sns.distplot(a=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", kde=False)

sns.distplot(a=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", kde=False)



# Add title

plt.title("Histogram of Petal Lengths, by Species")



# Force legend to appear

plt.legend()
# KDE plots for each species

sns.kdeplot(data=iris_set_data['Petal Length (cm)'], label="Iris-setosa", shade=True)

sns.kdeplot(data=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", shade=True)

sns.kdeplot(data=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", shade=True)



# Add title

plt.title("Distribution of Petal Lengths, by Species")