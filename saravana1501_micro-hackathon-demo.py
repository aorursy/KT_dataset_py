import pandas as pd

# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))

home_data = pd.read_csv('/kaggle/input/housingpricedemo/housing_demo.csv')

print('Setup Completed')

# Any results you write to the current directory are saved as output.
# split your data here!

# Specify your feature column here.

feature_columns = []

# Uncomment below line and add your code to split data.

# train_X, val_X, train_y, val_y = 
# specify and fit the model here!

# Uncomment below line and create your model.

# model = 
# Verify your model prediction