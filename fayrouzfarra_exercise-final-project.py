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

my_filepath = '../input/fivethirtyeight-comic-characters-dataset/dc-wikia-data.csv'







# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath,index_col='page_id')



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.head()
plt.subplots(1,2,figsize = (18,6))

plt.subplot(121)

sns.countplot(x= 'ALIGN',hue = 'SEX',data = my_data)

plt.legend(loc='upper right')

plt.subplot(122)

sns.countplot(x= 'ALIVE',hue = 'SEX',data = my_data)

plt.legend(loc='upper right')
plt.figure(figsize = (10,6))

dead_dc = my_data[my_data.ALIVE == 'Deceased Characters']

sns.countplot(x='ALIGN',data = dead_dc)
hero=my_data.sort_values(by='APPEARANCES', ascending=False)[:10][['name','SEX','APPEARANCES']]

hero
plt.figure(figsize=(20,16))

sns.boxenplot(x='name', y='APPEARANCES', data=hero, hue='SEX')
good_dc = my_data[my_data.ALIGN=='Good Characters'].sort_values(by='APPEARANCES', ascending=False)[:10]

good_dc
sns.boxenplot(x='APPEARANCES', y='HAIR', data=good_dc, hue='SEX')
draw=my_data.groupby('YEAR')['SEX'].value_counts()

draw
my_data.groupby('YEAR')['SEX'].value_counts().unstack().plot(figsize = (16,6))