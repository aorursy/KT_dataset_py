import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Decision Tree is a Flow Chart, that helps you make decisions based on previous experience.
# In the example, a person will try to decide if he/she should go to a comedy show or not.
# Luckily our example person has registered every time there was a comedy show in town,
# and registered some information about the comedian, and also registered if he/she went or not.
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
df = pandas.read_csv("../input/comedy-shows/shows.csv")
df
# To make a decision tree, all data has to be numerical.
# We have to convert the non numerical columns 'Nationality' and 'Go' into numerical values.
# Pandas has a map() method that takes a dictionary with information on how to convert the values.
# {'UK': 0, 'USA': 1, 'N': 2}
# Means convert the values 'UK' to 0, 'USA' to 1, and 'N' to 2.
# Example
# Change string values into numerical values:
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)
df
# Then we have to separate the feature columns from the target column.
# The feature columns are the columns that we try to predict from,
# and the target column is the column with the values we try to predict.
# Example
# X is the feature columns:
features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]
print(X)
# y is the target column:
y = df['Go']
print(y)
# Now we can create the actual decision tree, fit it with our details, and save a .png file on the computer:
# Example
# Create a Decision Tree, save it as an image, and show the image:
import graphviz
dtree = tree.DecisionTreeClassifier() # random_state = 1, max_depth = 7, min_samples_split=2
dtree = dtree.fit(X, y)
data = tree.export_graphviz(dtree, feature_names=features, out_file=None)
graph = graphviz.Source(data)
graph
# Predict Values
# We can use the Decision Tree to predict new values.
# Example: Should I go see a show starring a 40 years old American comedian,
# with 10 years of experience,
# and a comedy ranking of 7?
print(dtree.predict([[40, 10, 7, 1]]))
# What would the answer be if the comedy rank was 6?
print(dtree.predict([[40, 10, 6, 1]]))