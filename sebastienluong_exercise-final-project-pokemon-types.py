import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex7 import *

print("Setup Complete")
# Check for a dataset with a CSV file

step_1.check()
# Fill in the line below: Specify the path of the CSV file to read

my_filepath = "../input/pokemon/Pokemon.csv"



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath, index_col="#")

print(list(my_data.columns))



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.head()
plt.figure(figsize=(14,6))



# Setup data frame with specific param - 1st Gen

df_original = my_data.loc[my_data['Generation'] == 1]

sns.countplot(x='Type 1',data=df_original)

# Get specific values that match the condition given -> Ice type

print(df_original.loc[df_original['Type 1'] == 'Ice'])

# .loc() is very useful



# Check that a figure appears below

step_4.check()