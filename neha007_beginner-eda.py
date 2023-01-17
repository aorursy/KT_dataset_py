import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Fill in the line below: Specify the path of the CSV file to read

my_filepath = "../input/insurance.csv"

# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath)
# Print the first five rows of the data

my_data.head()
sns.lmplot(x='age',y='charges',hue='smoker',data=my_data)
sns.lmplot(x='age',y='charges',hue='sex',data=my_data)
sns.swarmplot(x='children',y='charges',data=my_data)
sns.swarmplot(x='region',y='charges',data=my_data)

sns.lmplot(x='bmi',y='charges',hue='smoker',data=my_data)

sns.distplot(my_data['charges']) # Your code here