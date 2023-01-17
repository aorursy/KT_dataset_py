# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Change the style of the figure to the "dark" theme
sns.set_style("dark")

# Path of the file to read
spotify_filepath = "../input/data-for-datavis/spotify.csv"

# Read the file into a variable spotify_data
spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)
spotify_data.sample(10)
# Line chart 
plt.figure(figsize=(12,6))
sns.lineplot(data=spotify_data);

# Path of the file to read
ign_filepath = "../input/data-for-datavis/ign_scores.csv"

# Fill in the line below to read the file into a variable ign_data
ign_data = pd.read_csv(ign_filepath, index_col="Platform")

# Set the width and height of the figure
plt.figure(figsize=(8, 6))

# Bar chart showing average score for racing games by platform
sns.barplot(x=ign_data['Racing'], y=ign_data.index)

# Add label for horizontal axis
plt.xlabel("")

# Add label for vertical axis
plt.title("Average Score for Racing Games on several platforms");

plt.figure(figsize=(12, 7))
# Heatmap showing average game score by platform and genre
sns.heatmap(data=ign_data, annot=True) ;

# Path of the file to read
ins_filepath = "../input/data-for-datavis/insurance.csv"

# Fill in the line below to read the file into a variable
insurance_data = pd.read_csv(ins_filepath)

plt.figure(figsize=(12, 7))
sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges']);
# Path of the file to read
candy_filepath = "../input/data-for-datavis/candy.csv"

# Fill in the line below to read the file into a variable ign_data
candy_data = pd.read_csv(candy_filepath)

# Analyzing Data
candy_data
plt.figure(figsize=(12, 7))
# Scatter plot showing the relationship between 'sugarpercent' and 'winpercent'
sns.scatterplot(x=candy_data['sugarpercent'],y=candy_data['winpercent']);

plt.figure(figsize=(12, 7))

# Scatter plot w/ regression line showing the relationship between 'sugarpercent' and 'winpercent'
sns.regplot(x='sugarpercent',y='winpercent', data = candy_data);

plt.figure(figsize=(12, 7))

# Scatter plot showing the relationship between 'chocolate' and 'winpercent'
sns.swarmplot(x=candy_data['chocolate'], y=candy_data['winpercent']);

# Color-coded scatter plot w/ regression lines
sns.lmplot(x='pricepercent', y='winpercent', hue= 'chocolate', data=candy_data, height=7, aspect=1.3);

plt.figure(figsize=(12, 7))

sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker']);

#fig,ax=plt.subplots(figsize=(12, 7))

sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data, height=7, aspect=1.3);
plt.figure(figsize=(12, 7))

sns.swarmplot(x=insurance_data['smoker'], y=insurance_data['charges']);
# Import pandas
import pandas as pd
import matplotlib.pyplot as plt
# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv('../input/gm2008/gm_2008_region.csv')
# Print the columns of df
print(df.columns)


df
# Create a boxplot of life expectancy per region
df.boxplot('life', 'Region', rot=60,figsize=(12,8))
# Show the plot
plt.show()

# Path of the file to read
iris_filepath = "../input/data-for-datavis/iris.csv"

# Fill in the line below to read the file into a variable
iris_data = pd.read_csv(iris_filepath)

iris_data
plt.figure(figsize=(9, 5))
# KDE plot 

plt.hist(iris_data['Species']);

plt.figure(figsize=(12, 7))
# Histogram 
sns.distplot(a=iris_data['Petal Length (cm)'], kde=False);

plt.figure(figsize=(12, 7))
# KDE plot 

sns.kdeplot(data=iris_data['Petal Length (cm)'], shade=True);

#plt.figure(figsize=(12, 7))
# 2D KDE plot
sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind="kde");

# Paths of the files to read
iris_set_filepath = "../input/data-for-datavis/iris_setosa.csv"
iris_ver_filepath = "../input/data-for-datavis/iris_versicolor.csv"
iris_vir_filepath = "../input/data-for-datavis/iris_virginica.csv"

# Read the files into variables 
iris_set_data = pd.read_csv(iris_set_filepath, index_col="Id")
iris_ver_data = pd.read_csv(iris_ver_filepath, index_col="Id")
iris_vir_data = pd.read_csv(iris_vir_filepath, index_col="Id")

# Print the first 5 rows of the Iris versicolor data
iris_ver_data.head()


plt.figure(figsize=(12, 7))
# Histograms for each species
sns.distplot(a=iris_set_data['Petal Length (cm)'], label="Iris-setosa", kde=False)
sns.distplot(a=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", kde=False)
sns.distplot(a=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", kde=False)

# Add title
plt.title("Histogram of Petal Lengths, by Species")

# Force legend to appear
plt.legend();

iris_data
df = iris_data

plt.subplots(figsize=(7,6), dpi=100)
sns.distplot( df.loc[df.Species=='Iris-setosa', "Sepal Length (cm)"] , color="dodgerblue", label="Setosa")
sns.distplot( df.loc[df.Species=='Iris-virginica', "Sepal Length (cm)"] , color="orange", label="Virginica")
sns.distplot( df.loc[df.Species=='Iris-versicolor', "Sepal Length (cm)"] , color="deeppink", label="Versicolor")

plt.title('Histogram of Sepal Lengths, by Species')
plt.legend();
plt.subplots(figsize=(7,6), dpi=100)

# Change line width
sns.violinplot( x=df["Species"], y=df["Sepal Length (cm)"], linewidth=5)
#sns.plt.show()
 
# Change width
sns.violinplot( x=df["Species"], y=df["Sepal Length (cm)"], width=0.3);

plt.figure(figsize=(12, 7))

# KDE plots for each species
sns.kdeplot(data=iris_set_data['Petal Length (cm)'], label="Iris-setosa", shade=True)
sns.kdeplot(data=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", shade=True)
sns.kdeplot(data=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", shade=True)

# Add title
plt.title("Distribution of Petal Lengths, by Species")

# Force legend to appear
plt.legend();

iris=iris_data.drop(['Id'], axis=1)
sns.pairplot(iris, hue="Species");
import missingno as msno 
vc = pd.read_csv('../input/vehicle-collisions/database.csv')

vc.describe()
msno.bar(vc)

msno.heatmap(vc);
msno.dendrogram(vc);