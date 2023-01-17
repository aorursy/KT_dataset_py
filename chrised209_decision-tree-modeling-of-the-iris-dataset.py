# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import the needed matplotlib functionality for scatter plot visualization.

import matplotlib.pyplot as plt

# import the needed dataset.

from sklearn.datasets import load_iris

# Import the model and an additional visualization tool.

from sklearn.tree import DecisionTreeClassifier, plot_tree

# Define a variable to establish three classes/species.

class_count = 3

# Define standard RGB color scheme for visualizing ternary classification in order to match the color map used later.

plot_colors = 'brg'

# Define marker options for plotting class assignments of training data.

markers = 'ovs'

# We also need to establish a resolution for plotting.  I favor clean powers of ten, but this is not by any means a hard and fast rule.

plot_res = 0.01
# Load the iris dataset from scikit-learn (note the use of from [library] import [function] above)

iris = load_iris()
# Set the size of the figure used to contain the subplots to be generated.

plt.figure(figsize=(20,10))



# Create an empty list of models to store the results of each pairwise model fit.

models = []



# Use enumerate() to define the possible pairs of features available and iterate over each pair.

for pair_index, pair in enumerate([[0, 1], [0, 2], [0, 3], 

                                           [1, 2], [1, 3], 

                                                   [2, 3] ]):



    # We only take the two features corresponding to the pair in question...

    X, y = iris.data[:, pair] , iris.target

    

    # ... to fit the decision tree classifier model.

    model = DecisionTreeClassifier().fit(X, y)

    

    # Append the results to the models list

    models.append(model)

    

    # Establish a two row by three column subplot array for plotting.

    plt.subplot(2, 3, pair_index + 1)

    

    # Define appropriate x and y ranges for each plot...

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5

    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # ... and use each range to define a meshgrid to use as the plotting area.

    xx, yy = np.meshgrid(np.arange(x_min, 

                                   x_max, 

                                   plot_res),

                         np.arange(y_min, 

                                   y_max, 

                                   plot_res) )

    # Use plt.tight_layout() to establish spacing of the subplots.

    plt.tight_layout(h_pad = 0.5, 

                     w_pad = 0.5, 

                       pad = 4.0 )

    

    # Predict the classification of each point in the meshgrid based on the calculated model above.

    # The numpy methods .c_() and .ravel() reshape our meshgrid values into a format compatible with our model.predict() method,

    Z = model.predict(np.c_[xx.ravel(), 

                            yy.ravel() ])

    # Reshape the predictions to match xx...

    Z = Z.reshape(xx.shape)

    # ... and prepare a contour plot that reflects the predictions .

    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.brg)

    

    # Define the subplot axis labels after title casing while preserving case on the unit of measure 

    plt.xlabel(iris.feature_names[pair[0]].title()[0:-4] + iris.feature_names[pair[0]][-4:])

    plt.ylabel(iris.feature_names[pair[1]].title()[0:-4] + iris.feature_names[pair[1]][-4:])

    

    # Plot the training points for each species in turn

    for i, color, marker in zip(range(class_count), plot_colors, markers):

        # Subset the data to the class in question with the np.where() method

        index = np.where(y == i)

        # Plot the class in question on the subplot

        plt.scatter(X[index, 0], 

                    X[index, 1], 

                    c = color,

                    marker = marker,

                    label = iris.target_names[i],

                    cmap = plt.cm.brg, 

                    edgecolor = 'black', 

                    s = 15                       )



# Define a title for the overall collection of subplots after each subplot is fully defined

plt.suptitle('Decision Surface of a Decision Tree Using Paired Features',

             size = 24                                                   )



# Define the legend for the subplot collection

plt.legend(loc = 'lower right',

           fontsize = 16,

           borderpad = 0.1, 

           handletextpad = 0.1 )



# Set limits just large enough to show everything cleanly

plt.axis("tight")

# Apply the decision tree classifier model to the data using all four parameters at once.

model_all_params = DecisionTreeClassifier().fit(iris.data, iris.target)

# Prepare a plot figure with set size.

plt.figure(figsize = (20,10))

# Plot the decision tree, showing the decisive values and the improvements in Gini impurity along the way.

plot_tree(model_all_params, 

          filled=True      )

# Display the tree plot figure.

plt.show()
# Apply the decision tree classifier model to the data using all four parameters at once, but with a maximum tree depth of 3

model_all_params_max_depth_3 = DecisionTreeClassifier(max_depth = 3).fit(iris.data, iris.target)

# Prepare a plot figure with set size.

plt.figure(figsize = (16,10))

# Plot the decision tree.

plot_tree(model_all_params_max_depth_3,

          rounded = True,

          filled = True                )

# Display the tree plot figure.

plt.show()
# Apply the model to the data as before, but with a minimum impurity decrease of 0.01

model_all_params_min_imp_dec_001 = DecisionTreeClassifier(min_impurity_decrease = 0.01).fit(iris.data, iris.target)

# Prepare a plot figure with set size.

plt.figure(figsize = (16,10))

# Plot the decision tree.

plot_tree(model_all_params_min_imp_dec_001,

          rounded = True,

          filled = True                )

# Display the tree plot figure.

plt.show()
# Prepare a plot figure with set size.

plt.figure(figsize = (20,12))

# Plot the decision tree, showing the decisive values and the improvements in Gini impurity along the way.

plot_tree(models[5],

          rounded = True,

          filled = True  )

# Display the tree plot figure.

plt.show()