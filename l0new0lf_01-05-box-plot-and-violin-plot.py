import matplotlib.pyplot as plt

import seaborn as sns

iris = sns.load_dataset("iris")



iris.head()
plt.figure(figsize=(7,7))



sns.boxplot(x='species',y='petal_length', data=iris)





plt.grid()

plt.show()
plt.figure(figsize=(7,7))



sns.violinplot(x='species',y='petal_length', data=iris)





plt.grid()

plt.show()