import numpy as np 
import pandas as pd 
# Input data files are available in the read-only "../input/" directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
mushrooms = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
mushrooms.head()
mushrooms.describe()
# Making a new df of only the edible mushrooms 
is_edible = mushrooms["class"] == 'e'
edible_mushrooms = mushrooms[is_edible]

# Comparing the number of edible mushrooms to the number of all mushrooms
print(f'Amount of edible mushrooms: {len(edible_mushrooms)} \n Amount of all mushrooms: {len(mushrooms)}')
# Making a new df of only the poisonous mushrooms 
is_poisonous = mushrooms["class"] == 'p'
poisonous_mushrooms = mushrooms[is_poisonous]

len(poisonous_mushrooms)
# Proportion of poisonous mushrooms to all mushrooms
pois_mush_amount = len(poisonous_mushrooms)
all_mush_amount = len(mushrooms)

proportion_poisonous = round((pois_mush_amount / all_mush_amount), 2) * 100
print(f'Proportion of poisonous mushrooms: {proportion_poisonous}%')
# Proportion of edible mushrooms to all mushrooms
edibl_mush_amount = len(edible_mushrooms)
all_mush_amount = len(mushrooms)

proportion_edible = round((edibl_mush_amount / all_mush_amount), 2) * 100
print(f'Proportion of edible mushrooms: {proportion_edible}%')
poisonous_mushrooms.head()
# Taking another look at the data: trying to see what feaures are available
mushrooms.info()
mushrooms_columns = mushrooms.columns

def find_all_unique(): 
    for _ in mushrooms_columns: 
        print(mushrooms_columns[_].unique())

find_all_unique()
mushrooms.values
# The column 'veil-type' only has one value, so that is not useful for an analysis
# Remove 'veil-type' column 

mushrooms = mushrooms.drop("veil-type", axis=1)
# Using Seaborn to visualize the proportion of edible and poisonous mushrooms in the dataset
import seaborn as sns

x = mushrooms['class']
ax = sns.countplot(x=x, data=mushrooms)


import matplotlib.pyplot as plt
%matplotlib inline
# Using Seaborn to visualize the relationships between different features and the target class of edible or poisonous

# Define helper function to set up the graphs: 
def plot_data(hue, data): 
    for i, col in enumerate(data.columns): 
        plt.figure(i)
        sns.set(rc={'figure.figsize': (11.7, 8.27)})
        ax = sns.countplot(x=data[col], hue=hue, data=data)

# Using the function to visualize Mushrooms: 
#hue = mushrooms['class']
#data_for_plot = mushrooms.drop('class', 1)
#plot_data(hue, data_for_plot)
# Use label encoder to encode the label 
# The label is what we want to predict: whether poisonous or edible
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
mushrooms['class'] = le.fit_transform(mushrooms['class'])

mushrooms.head()
# Use OneHotEncoder for the rest of the features 
encoded_mushrooms = pd.get_dummies(mushrooms)

encoded_mushrooms.head()
# Split the train and test data 
from sklearn.model_selection import train_test_split 

X = encoded_mushrooms 
y = mushrooms['class'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Try out a model and see what happens
from sklearn.svm import LinearSVC

model = LinearSVC()

model.fit(X_train, y_train.ravel()) 

# In case you are wondering, y_train.ravel() returns "a contiguous 1d flattened array"
model.score(X_test, y_test)
# Well that's clearly overfitting, but at least all the syntax worked! 
