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
my_filepath = "../input/covid19-mx/casos_confirmados.csv"

# Check for a valid filepath to a CSV file in a dataset
step_2.check()
# Fill in the line below: Read the file into a variable my_data
my_data = pd.read_csv(my_filepath, index_col=0)

# Check that a dataset has been uploaded into my_data
step_3.check()
# Print the first five rows of the data
my_data.head()
state_wise_cnfmd = my_data.groupby(['State']).Confirmed.agg([len])
sex_wise_cnfmd = my_data.groupby(['Sex']).Confirmed.agg([len])
state_wise_cnfmd = state_wise_cnfmd.reset_index()
state_wise_cnfmd.head()
sex_wise_cnfmd = sex_wise_cnfmd.reset_index()
sex_wise_cnfmd.head()
# Create a plot
plt.figure(figsize=(20,10))
sns.barplot(x=state_wise_cnfmd['State'], y=state_wise_cnfmd['len']) # Your code here
#sns.distplot(a=my_data['Age'], kde=False)
# Check that a figure appears below
step_4.check()