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

my_filepath = "../input/fivethirtyeight-comic-characters-dataset/dc-wikia-data.csv"



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath)



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

my_data.head()
my_data.isnull().sum()
plt.figure(figsize=(12,4))

plt.title("Graph of Aligns\n", fontsize=20)

ax = sns.countplot(x="ALIGN", data=my_data,

                   facecolor=(0, 0, 0, 0),

                   linewidth=4,

                   edgecolor=sns.color_palette("dark", 10))
import plotly.graph_objects as go

labels = my_data.ALIVE.unique()

values=[]

for each in labels:

    values.append(len(my_data[my_data.ALIVE==each]))



fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])

fig.show()

xx= my_data.nlargest(5, ['APPEARANCES'])

xx.name

goods = my_data.loc[my_data.ALIGN == 'Good Characters']

df = goods.SEX.value_counts()
goods = my_data.loc[my_data.ALIGN == 'Good Characters']

df = goods.SEX.value_counts()



plt.figure(figsize=(27,6))

sns.lineplot(data=df)

plt.xlabel("\nSex", fontsize=15)

plt.ylabel("Number of people", fontsize=15)

plt.title("Gender Chart of Good Characters\n", fontsize=25)
sexCount = my_data.SEX.value_counts()

sexCount
#yıllara göre hangi cinsiyetteki karakterlerin ne zaman çıktığını gösterir.

plt.figure(figsize=(27,6))

plt.xlabel("", fontsize=15)

plt.ylabel("", fontsize=15)

plt.title("Gender Distribution of Characters by Years\n", fontsize=25)



sns.scatterplot(x=my_data['YEAR'], y=my_data['SEX'], color="red")