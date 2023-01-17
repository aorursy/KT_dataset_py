# import libraries

import pandas as pd

import seaborn as sns 

import matplotlib.pyplot as plt 

from scipy.stats import norm 
forest_data = pd.read_csv("../input/learn-together/train.csv")

forest_test = pd.read_csv("../input/learn-together/test.csv")
forest_data.info()
forest_data.describe()
forest_data.head()
forest_data.columns
forest_data.dtypes
forest_data.shape
# yet I didn't use any
#corrMatt = X_train[["","","","","","",""]].corr()

corrmat = forest_data[[ 'Elevation', 'Aspect', 'Slope',

       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',

       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',

       'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',

       'Cover_Type']].corr()



f, ax = plt.subplots(figsize =(11, 10)) 

sns.heatmap(corrmat, annot=True, ax = ax, cmap ="YlGnBu", linewidths = 0.1)
# Elevation in meters

sns.distplot(forest_data.Elevation, color="b")
f, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True, sharey=True)

sns.scatterplot(y='Hillshade_9am', x='Elevation', 

                 data=forest_data, ax=axes[0])

sns.scatterplot(y='Hillshade_Noon', x='Elevation', 

                 data=forest_data, ax=axes[1])

sns.scatterplot(y='Hillshade_3pm', x='Elevation', 

                 data=forest_data, ax=axes[2])
f, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)



sns.distplot(forest_data.Hillshade_9am, color="y", ax=axes[0])

sns.distplot(forest_data.Hillshade_Noon, color="b", ax=axes[1])

sns.distplot(forest_data.Hillshade_3pm, color="g", ax=axes[2])

f, axes = plt.subplots(2, 1, figsize=(15, 15), sharex=True, sharey=False)



sns.distplot(forest_data.Slope, color="y", ax=axes[0])

sns.distplot(forest_data.Aspect, color="b", ax=axes[1])

f, axes = plt.subplots(2, 1, figsize=(15, 15), sharex=True, sharey=False)

sns.scatterplot(y='Slope', x='Aspect', 

                 data=forest_data, ax=axes[0])

sns.scatterplot(y='Aspect', x='Slope', 

                 data=forest_data, ax=axes[1])
forest_train = forest_data.drop(["Id"], axis = 1)



forest_test_id = forest_test["Id"]

forest_test = forest_test.drop(["Id"], axis = 1)
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



# Split X and y in train and validation sets

X_train, X_val, y_train, y_val = train_test_split(forest_train.drop(['Cover_Type'], axis=1), forest_train['Cover_Type'], test_size=0.2, random_state = 50)



# Define model

forest_model = RandomForestClassifier(n_estimators=100, random_state=50)

# Fit the model to train data

forest_model.fit(X_train, y_train)
# Check the model accuracy

from sklearn.metrics import classification_report, accuracy_score

forest_model.score(X_train, y_train)
# Make prediction

forest_preds = forest_model.predict(X_val)



accuracy_score(y_val, forest_preds)
# Select features

features = ['Elevation', 'Aspect', 'Slope',

       'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',

       'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm','Horizontal_Distance_To_Fire_Points']



forest_data_reduced = forest_data[features]
X = forest_data_reduced.copy()

y = forest_data['Cover_Type']
from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



# Split X and y in train and validation sets

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state = 2)



# Define model

forest_model = RandomForestClassifier(n_estimators=100, random_state=2)

# Fit the model to train data

forest_model.fit(X_train, y_train)
# Check the model accuracy

from sklearn.metrics import classification_report, accuracy_score

forest_model.score(X_train, y_train)
# Make prediction

forest_preds = forest_model.predict(X_val)

accuracy_score(y_val, forest_preds)
# Define a function to calculate accuracy_score

def acc_calculate(max_leaf_nodes, X_train, X_val, y_train, y_val):

    model = RandomForestClassifier(n_estimators=100,max_leaf_nodes=max_leaf_nodes, random_state=50)

    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)

    acc = accuracy_score(y_val, val_preds)

    return(acc)
# compare accuracy_score with differing values of max_leaf_nodes

for max_leaf_nodes in [5, 50, 500, 5000, 10000]:

    my_acc = acc_calculate(max_leaf_nodes, X_train, X_val, y_train, y_val)

    print("Max leaf nodes: %d  \t\t accuracy_score:  %f" %(max_leaf_nodes, my_acc))
# Run the best model to be used for prediction

X_train, X_val, y_train, y_val = train_test_split(forest_train.drop(['Cover_Type'], axis=1), forest_train['Cover_Type'], test_size=0.2, random_state = 50)



# Define model

forest_model = RandomForestClassifier(n_estimators=100, random_state=50)

# Fit the model to train data

forest_model.fit(X_train, y_train)
test_preds = forest_model.predict(forest_test)
# To submit on kaggle

output = pd.DataFrame({'Id': forest_test_id,

                       'Cover_Type': test_preds})

output.to_csv('submission.csv', index=False)