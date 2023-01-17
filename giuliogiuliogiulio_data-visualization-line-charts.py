import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
print("Setup Complete")
# Set up code checking
import os
if not os.path.exists("../input/museum_visitors.csv"):
    os.symlink("../input/data-for-datavis/museum_visitors.csv", "../input/museum_visitors.csv") 
from learntools.core import binder
binder.bind(globals())
from learntools.data_viz_to_coder.ex2 import *
# Path of the file to read
museum_filepath = "../input/museum_visitors.csv"

# Fill in the line below to read the file into a variable museum_data
museum_data = pd.read_csv(museum_filepath, index_col = 'Date', parse_dates = True)
# Print the last five rows of the data 
museum_data.tail()
# Fill in the line below: How many visitors did the Chinese American Museum 
# receive in July 2018?
ca_museum_jul18 = 2620 

# Fill in the line below: In October 2018, how many more visitors did Avila 
# Adobe receive than the Firehouse Museum?
avila_oct18 = (19280 - 4622)
# Line chart showing the number of visitors to each museum over time
plt.figure(figsize=(15,7))
sns.lineplot(data = museum_data)
plt.title("Monthly Visitors to Los Angeles City Museums")
# Line plot showing the number of visitors to Avila Adobe over time
plt.figure(figsize=(10,5))
plt.title('Avila visitors over time')
sns.lineplot(data = museum_data['Avila Adobe'])
plt.xlabel('Date')
#March-August