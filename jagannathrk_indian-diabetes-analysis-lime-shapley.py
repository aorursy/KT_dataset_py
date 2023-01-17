# Loading necessary libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

%matplotlib inline





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings

warnings.filterwarnings("ignore")      
# Reading in the data

df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

df.head()

df.info()
# Creating the target and the features column and splitting the dataset into test and train set.



X = df.iloc[:, :-1]

y = df.iloc[:, -1]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# Training and fitting a Random Forest Model

my_model = RandomForestClassifier(n_estimators=100,

                                  random_state=0).fit(X_train, y_train)
# Calculating and Displaying importance using the eli5 library

import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(my_model, random_state=1).fit(X_test,y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
# training and fitting a Decision Tree

from sklearn.tree import DecisionTreeClassifier

feature_names = [i for i in df.columns]

tree_model = DecisionTreeClassifier(random_state=0).fit(X_train, y_train)
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI', 'DiabetesPedigreeFunction', 'Age']
# Let's plot a decision tree source : #https://www.kaggle.com/willkoehrsen/visualize-a-decision-tree-w-python-scikit-learn

# Since there are a lot of attributes, it is difficult to actually make sense of the decision tree graph in this notebook. 

# It is advised to export it as png and view it.



from sklearn import tree

import graphviz



tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=feature_names,filled = True)

graphviz.Source(tree_graph)


from pdpbox import pdp, get_dataset, info_plots



# Create the data that we will plot

pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=X_test, model_features=feature_names, feature='Glucose')



# plot it

pdp.pdp_plot(pdp_goals, 'Glucose')

plt.show()
# Create the data that we will plot

pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=X_test, model_features=feature_names, feature='Insulin')



# plot it

pdp.pdp_plot(pdp_goals, 'Insulin')

plt.show()
features_to_plot = ['Glucose','Insulin']

inter1  =  pdp.pdp_interact(model=tree_model, dataset=X_test, model_features=feature_names, features=features_to_plot)



pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour', plot_pdp=True)

plt.show()
row_to_show = 10

data_for_prediction = X_test.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired

data_for_prediction_array = data_for_prediction.values.reshape(1, -1)



tree_model.predict_proba(data_for_prediction_array)
import shap  # package used to calculate Shap values



# Create object that can calculate shap values

explainer = shap.TreeExplainer(tree_model)



# Calculate Shap values

shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
import shap  # package used to calculate Shap values



# Create object that can calculate shap values

explainer = shap.TreeExplainer(tree_model)



# calculate shap values. This is what we will plot.

# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.

shap_values = explainer.shap_values(X_test)



# Make plot. Index of [1] is explained in text below.

shap.summary_plot(shap_values[1],X_test)