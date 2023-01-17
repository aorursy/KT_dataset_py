# Kaggle Starter Code

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import the Kickstarter Dataset
ds = pd.read_csv("/kaggle/input/kickstarter-projects/ks-projects-201801.csv")
# Print Out General Information about the Dataset
ds.info()
# Filter the dataset to view only the "state" feature
ds["state"]
# Store Breakdown of State of Kickstarter Projects
projState = ds['state'].value_counts()

# Plot Pie Graph for State of Kickstarter Projects
projState.plot(kind='pie')
# Plot Pie Graph for State of Kickstarter Projects with Method Chaining
ds['state'].value_counts().plot(kind='pie')
# Bar Graph for State of Kickstarter Projects
projState.plot(kind='bar')
# Horizontal Bar Graph for State of Kickstarter Projects
projState.plot(kind='barh')
# Plot Line Graph Depicting Deadling vs Goal of Kickstarter Projects
ds.plot(x='deadline',y='goal', kind='line')
# Plot First 100 Data Points in a Line Graph Depicting Deadling vs Goal of Kickstarter Projects
ds.head(100).plot(x='deadline',y='goal', kind='line')
# Plot Scatter Plot Depicting Date Launched vs Goal
# This line of code may take awhile to run. If so, cancel the run and apply the head() method to get only a small amount of the dataset. 

ds.plot(x='launched', y='goal', kind='scatter')
# Plot First 100 Data Points in a Scatter Plot Depicting Date Launched vs Goal
ds.head(100).plot(x='launched', y='goal', kind='scatter')
# Plot Histogram for Goal Amounts
ds.hist(column='goal')
# Filter Dataset with Only Entities with a Goal Amounts under $5000
under5000 = ds[(ds['goal']<5000)]

# Plot Histogram for Goal Amounts on Filtered Dataset
under5000.hist(column='goal')
# Compare Histogram with Different Bin Sizes for Data Fitting
under5000.hist(column='goal', bins=20)
under5000.hist(column='goal')

# Import the Seaborn Python Library and Assign the 'sns' Nickname
import seaborn as sns

# Store Breakdown of State of Kickstarter Projects
state_freq = ds['state'].value_counts()

# Plot Bar Graph of Kickstarter Projects
sns.barplot(x=state_freq.index, y=state_freq)
# Plot Count Plot of Kickstarter Projects Based on "state" feature.
# This graph calculates the freqencies of each possible value in the state feature before plotting the graph.
# The countplot() method helps eliminate using the value_counts() to generate the frequencies before plotting.
# Notice the code in the previous code block generates the same result as this single line of code!

sns.countplot(x='state', data=ds)
# Plotting Count Plot Based on "state" features and also grouping by "main_category" value for each state. 
# After separating the data based on "state", we separate once more based on the hue parameter, which is "main_category".
# We can change the color palette of the graph by setting the palette parameter to the various color paalettes available in seaborn. 

sns.countplot(x='state', hue='main_category', data=ds, palette="Set3")
# Plotting Box Plot on the Distribution of Goal Amounts For Each Main Category
sns.boxplot(x="main_category", y="goal", data=ds)