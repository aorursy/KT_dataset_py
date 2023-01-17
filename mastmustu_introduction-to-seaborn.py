# Import the necessary libraries

import matplotlib.pyplot as plt

import pandas as pd



# Initialize Figure and Axes object

fig, ax = plt.subplots()



# Load in data

tips = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")



print(tips["total_bill"])



# Create violinplot

ax.violinplot(tips["total_bill"], vert=False)



# Show the plot

plt.show()
# Import the necessary libraries

import matplotlib.pyplot as plt

import seaborn as sns



# Load the data

tips = sns.load_dataset("tips")



# Create violinplot

sns.violinplot(x = "total_bill", data=tips)



# Show the plot

plt.show()
# Import necessarily libraries

import matplotlib.pyplot as plt

import seaborn as sns



# Load data

titanic = sns.load_dataset("titanic")



# Set up a factorplot

g = sns.factorplot("class", "survived", "sex", data=titanic, kind="bar", palette="muted", legend=False)

                   

# Show plot

plt.show()
df = pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')
df.head()
df.corr()
h = sns.heatmap(df.corr())
sns.heatmap(df.corr(),linewidths=0.1,vmax=1.0,square=True,linecolor='white',annot=True)
colormap = plt.cm.plasma

sns.heatmap(df.corr(),linewidths=0.1,vmax=1.0,square=True,linecolor='white',annot=True , cmap=colormap)