# Import libraries

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns
# Configuring styles

sns.set_style("darkgrid")

matplotlib.rcParams['font.size'] = 14

matplotlib.rcParams['figure.figsize'] = (9, 5)

matplotlib.rcParams['figure.facecolor'] = '#00000000'
# Sample data

years = range(2000, 2012)

apples = [0.895, 0.91, 0.919, 0.926, 0.929, 0.931, 0.934, 0.936, 0.937, 0.9375, 0.9372, 0.939]

oranges = [0.962, 0.941, 0.930, 0.923, 0.918, 0.908, 0.907, 0.904, 0.901, 0.898, 0.9, 0.896, ]



# First line

plt.plot(years, apples, 'b-x', linewidth=4, markersize=12, markeredgewidth=4, markeredgecolor='navy')



# Second line

plt.plot(years, oranges, 'r--o', linewidth=4, markersize=12,);



# Title

plt.title('Crop Yields in Hoenn Region')



# Line labels

plt.legend(['Apples', 'Oranges'])



# Axis labels

plt.xlabel('Year'); plt.ylabel('Yield (tons)');
# Load data into a Pandas dataframe

data = sns.load_dataset("iris")



# View the data

data.sample(5)
# Create a scatter plot

sns.scatterplot(data.sepal_length, # X-axis

                data.sepal_width,  # Y-axis

                hue=data.species,  # Dot color

                s=100);



# Chart title

plt.title("Flowers");
plt.title("Distribution of Sepal Width")



sns.distplot(data.sepal_width, kde=False);
plt.title("Distribution of Sepal Width")



sns.distplot(data.sepal_width);
# Load the example flights dataset as a matrix

flights = sns.load_dataset("flights").pivot("month", "year", "passengers")



# Chart Title

plt.title("No. of Passengers (1000s)")



# Draw a heatmap with the numeric values in each cell

sns.heatmap(flights, 

            fmt="d", 

            annot=True, 

            linewidths=.5, 

            cmap='Blues',

            annot_kws={"fontsize":13});
plt.title("Flowers")



sns.kdeplot(data.sepal_length, data.sepal_width, shade=True, shade_lowest=False);
setosa = data[data.species == 'setosa']

virginica = data[data.species == 'virginica']



plt.title("Flowers (Setosa & Virginica)")



sns.kdeplot(setosa.sepal_length, setosa.sepal_width, shade=True, cmap='Reds', shade_lowest=False)

sns.kdeplot(virginica.sepal_length, virginica.sepal_width, shade=True, cmap='Blues', shade_lowest=False);
# Load the example tips dataset

tips = sns.load_dataset("tips");

tips

# Chart title

plt.title("Daily Total Bill")



# Draw a nested boxplot to show bills by day and time

sns.boxplot(tips.day, tips.total_bill, hue=tips.smoker);
sns.barplot(x="day", y="total_bill", hue="sex", data=tips);