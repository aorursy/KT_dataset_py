# Importing numpy library and giving it a name "np" for fast access

import numpy as np



test_arr = np.array([1,5,2,6,8,3])
# Calculating the mean

mean = np.mean(test_arr)



print("The mean of the array = %f" % mean) # Must be 4.1667 Why wrong answer? HINT what does the %i in the format string do
# Calculating the midean

median = np.median(test_arr)



print("The median of the array = %0.2f" % median)
#Calculat the STD using the same array

std = np.std(test_arr)



print("The median of the array = %0.2f" % std)
#Calculat the mode using scipy

from scipy import stats



stats.mode([2,3,4,5])
# Importing the data set from sklearn library

from sklearn.datasets import fetch_covtype



cov = fetch_covtype()

columns = ['Elevation', 'Aspect', 'Slope',

       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',

       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',

       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',

       'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',

       'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3',

       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',

       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',

       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',

       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',

       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',

       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',

       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',

       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',

       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']



# ??? Why all of this?? np arrays doesn't have info about the features names
import pandas as pd



# Import the data into a dataframe for exploration

data = pd.DataFrame(cov.data, columns = columns)

data['Cover_Type'] = cov.target
data.head(5) # Default?
data.Soil_Type35.value_counts()
data['Elevation'] # Could be data.Elevation as well
data.Elevation.value_counts();

data.Cover_Type.value_counts()
data.info();
data.describe()
# Import matplotlib to show the graph

import matplotlib.pyplot as plt



# Why using bins??

data.Cover_Type.hist(bins=7)

plt.show()
data[['Elevation', 'Aspect', 'Slope', 'Cover_Type']].corr()
data.corr()
import seaborn as sns

import matplotlib.pyplot as plt





corr = data[['Elevation', 'Aspect', 'Slope',

       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',

       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',

       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']].corr()

f, ax = plt.subplots(figsize=(25, 25))



# Color Map, not mandatory

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Heat Map

sns.heatmap(corr, cmap=cmap, vmax=1, vmin = -1, center=0,

            square=True, linewidths=.5)
import seaborn as sns



Exploration_columns = data[['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology','Cover_Type']].head(1000)

sns.pairplot(Exploration_columns, hue='Cover_Type')


from pandas.plotting import scatter_matrix



# Can select multiple rows for exploration

scatter_matrix(data[['Elevation', 'Aspect', 'Slope']])



plt.show()
data.isna().sum()