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
my_filepath = "../input/fivethirtyeight-comic-characters-dataset/marvel-wikia-data.csv"

# Check for a valid filepath to a CSV file in a dataset
step_2.check()
# Fill in the line below: Read the file into a variable my_data
my_data = pd.read_csv(my_filepath)

# Check that a dataset has been uploaded into my_data
step_3.check()
# Print the first five rows of the data
my_data.head().T
my_data.shape
my_data.info()
my_data.describe()
my_data['SEX'].value_counts(normalize=True)
# Create a plot
plt.figure(figsize=(14,8))
plt.title('Quantidade de Personagens por Sexo', fontsize=20, color='white')
sns.countplot(x="SEX", data=my_data)
plt.xlabel('Sexo', fontsize=16, color='white')
plt.ylabel('Quantidade', fontsize=16, color='white')

# Check that a figure appears below
step_4.check()