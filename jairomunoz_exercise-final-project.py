import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style='darkgrid')
print("Setup Complete")
# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.data_viz_to_coder.ex7 import *
print("Setup Complete")
# Check for a dataset with a CSV file
step_1.check()
# Fill in the line below: Specify the path of the CSV file to read
my_filepath = '../input/fivethirtyeight-comic-characters-dataset/dc-wikia-data.csv'

# Check for a valid filepath to a CSV file in a dataset
step_2.check()
# Fill in the line below: Read the file into a variable my_data
my_data = pd.read_csv(my_filepath)

# Check that a dataset has been uploaded into my_data
step_3.check()
# Print the first five rows of the data
my_data.head()
my_data.info()
my_data.describe().T
# Create a plot
plt.figure(figsize=(10,6))
sns.countplot(x="EYE", data=my_data, order=my_data.EYE.value_counts().index)
plt.xticks(rotation=90)
# Check that a figure appears below
step_4.check()
# Alignments of the characters
sns.countplot(x="ALIGN", data=my_data, order=my_data.ALIGN.value_counts().index)
plt.xticks(rotation=45)
lgtb_data = my_data.loc[my_data.GSM.notna()]
print('There are {} characters belonging to LGTB group'.format(lgtb_data.size))
sns.countplot(x="GSM", data=lgtb_data, order=lgtb_data.GSM.value_counts().index)
plt.xticks(rotation=45)
lgtb_data.GSM.value_counts()
lgtb_data.loc[lgtb_data.GSM == 'Bisexual Characters'].name.tolist()
my_data.columns
top_appearances = my_data.sort_values(by='APPEARANCES', ascending=False).head()
top_appearances
sns.barplot(x=top_appearances.name ,y=top_appearances.APPEARANCES)
plt.xticks(rotation=45)
sns.distplot(a=my_data.YEAR, kde=False)