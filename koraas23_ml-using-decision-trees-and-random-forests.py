import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from sklearn.model_selection import train_test_split



import warnings

warnings.filterwarnings('ignore')



from datetime import datetime
# Import the Data

data = pd.read_csv('../input/data.csv')
# Explore the Data

data.head(8)
data.describe()
# Check for missing values

sns.heatmap(data.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
# Delete the empty column

data.drop('Unnamed: 32', axis = 1, inplace = True)
# Visualize the data

sns.set(style = 'darkgrid')

g = sns.countplot(x = "diagnosis", data = data, palette = "Set3")

plt.ylabel("Number of Occurences")

plt.xlabel("Diagnosis")

plt.title("Diagnosis Distribution")
# drop ID column from our dataset

data.drop('id', axis = 1, inplace = True)
data.columns
# Create groups of some variables we want to visualize

means = data[['diagnosis', 'radius_mean', 'texture_mean', 'radius_worst', 'texture_worst']]



means2 = data[['diagnosis', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean']]



means3 = data[['diagnosis', 'concave points_mean', 'fractal_dimension_mean']]
# Use pd.melt to be able to visualize multiple variables at once

melt_means = pd.melt(means, id_vars = 'diagnosis', var_name = "Variables", value_name = "Value")

melt_means2 = pd.melt(means2, id_vars = 'diagnosis', var_name = "Variables", value_name = "Value")

melt_means3 = pd.melt(means3, id_vars = 'diagnosis', var_name = "Variables", value_name = "Value")
# Boxplots

sns.boxplot(x = "Variables", y = "Value", data = melt_means, hue = 'diagnosis', palette = 'pastel')
sns.boxplot(x = "Variables", y = "Value", 

            data = melt_means2, hue = 'diagnosis', 

            palette = 'pastel')



plt.xticks(rotation=25)
sns.boxplot(x = "Variables", y = "Value", data = melt_means3, hue = 'diagnosis', palette = 'pastel')
# We can also see the relationship between multiple variables at once

f = sns.PairGrid(means)

f = f.map_upper(plt.scatter)

f = f.map_lower(sns.kdeplot, cmap = "Purples_d")

f = f.map_diag(sns.kdeplot, lw = 3, legend = False)
c = sns.swarmplot(x = "Variables", y = "Value", data = melt_means2, hue = 'diagnosis', palette = 'pastel')

plt.xticks(rotation=25)
# Violin plots

cv = sns.violinplot(x = "Variables", y = "Value", data = melt_means2, hue = 'diagnosis', palette = 'spring')

plt.xticks(rotation=25)
cv = sns.violinplot(x = "Variables", y = "Value", data = melt_means2, hue = 'diagnosis', palette = 'seismic', split = True)

plt.xticks(rotation=25)
# If we want to see any specific relationships, we can use this:

sns.jointplot(x = 'texture_mean', y = 'radius_mean', data = means, kind = 'hex', color = "#4CB391")
corrmat = data.corr()

fig, ax = plt.subplots(figsize=(18, 18))

sns.heatmap(corrmat, square = True, cmap = "YlGnBu", annot = True, fmt = '.1f', linewidths = .25, linecolor = 'r')
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
# Prepare the data

X = data.drop('diagnosis', axis = 1)

y = data['diagnosis']
# Split the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
# Feature Selection

sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))

sel.fit(X_train, y_train)
# Check which features were selected to be the best to use

sel.get_support()
selected_feat = X_train.columns[(sel.get_support())]

print(selected_feat)
X = data[['radius_mean', 'perimeter_mean', 'area_mean', 'concave points_mean',

       'radius_worst', 'perimeter_worst', 'area_worst',

       'concave points_worst']]



y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
# Predict values based on selected model

predictions = dtree.predict(X_test)
# Check how the well the model did

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))

print('\n')

print(classification_report(y_test, predictions))
rfc = RandomForestClassifier(n_estimators = 100)
# Fit the model

rfc.fit(X_train, y_train)
# Predict values

rfc_pred = rfc.predict(X_test)
# Check accuracy

print(confusion_matrix(y_test, rfc_pred))

print('\n')

print(classification_report(y_test, rfc_pred))