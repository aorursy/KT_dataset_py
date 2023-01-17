# For handling DataFrames and NumPy arrays
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Put imports for the libraries you will use here

# End of user imports

# Render and display charts in the notebook
%matplotlib inline
# Set a white theme for seaborn charts
sns.set_style('white')

param_grid = {'parameter_name' : 'parameter_search_space',
              'parameter_name2': 'parameter_search_space2'}
# and so on...
# If you want to use Grid search, change to GridSearchCV. Don't forget to import first!
# But be warned, it will take a long time
param_search = RandomizedSearchCV(estimator=clf, # Your model variable goes here,
                                  param_distributions=param_grid, # Your parameter search space
                                  n_iter=100, # How many times do you want it to run?
                                  cv=5,
                                  verbose=2, # Print messages. Set to 0 to silence it
                                  n_jobs=-1) # Use all available CPU cores