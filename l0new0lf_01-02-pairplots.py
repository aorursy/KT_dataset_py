import matplotlib.pyplot as plt

import seaborn as sns
iris = sns.load_dataset("iris")



iris.head()
plt.close();

sns.set_style("whitegrid");

sns.pairplot(iris, hue="species", height=3);

plt.show()