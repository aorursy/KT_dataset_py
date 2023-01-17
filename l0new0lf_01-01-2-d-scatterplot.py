import matplotlib.pyplot as plt
import seaborn as sns

iris = sns.load_dataset("iris")
print(iris.sepal_length)
iris.head()
# Method 1
ax = sns.scatterplot(x=iris.sepal_length, y=iris.sepal_width,
                     hue=iris.species, style=iris.species)
# Method 2
sns.set_style("whitegrid");

sns.FacetGrid(iris, hue="species", height=4) \
   .map(plt.scatter, "sepal_length", "sepal_width") \
   .add_legend();
plt.show();
# example: check importance of all four features
# Unfortunately, as we can only take combinations of 2 features (cz, only two axis)
# For 4 features, 4C2 = 6 scatter plots

RANGE2 = [(1,0,1), (2,0,2), (3,0,3), (4,2,1), (5,2,3), (6,3,1)]

plt.figure(2)
for q, x_index, y_index in RANGE2:

    sns.FacetGrid(iris, hue="species") \
   .map(plt.scatter, iris.columns[x_index], iris.columns[y_index]);
    