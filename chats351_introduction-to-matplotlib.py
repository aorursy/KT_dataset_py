import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.plot([1, 2, 3, 4]) # x-axis is associated with indexes and Y-axis is associated with list
plt.show()
plt.plot([2, 5, 6, 9], [1, 3, 5, 7], '-bo') # g is a green color followed by shape of point
plt.show()
# Draw Scatter Plot
plt.scatter([2, 5, 6, 9], [1, 3, 5, 7])
plt.show()
#n1 = np.array([1, 2, 3, 4, 5])
n1 = np.arange(1, 5.1, 0.1)
sq = n1**2
cb = n1**3
plt.figure(figsize = (15, 7))
plt.plot(n1, 'r')
plt.plot(sq, 'y')
plt.plot(cb, 'g')
plt.title("Comparison of squares and cubes")
plt.xlabel("Index")
plt.ylabel("Value")
plt.savefig("Graph.png")
plt.grid(True)
plt.legend(['Numbers', 'Squares', 'Cubes'])
plt.xticks(range(0, 50, 10))
plt.yticks(range(0, 130, 10))
plt.show()
df = pd.read_csv("../input/BlackFriday.csv")
df.head()
groupNames = []
counts = []
for group_name, subset in df.groupby('Gender'):
    groupNames.append(group_name)
    counts.append(len(subset))
    print(group_name, len(subset))
print(groupNames)
print(counts)
# Bar Charts
plt.bar([1, 2, 3], [10, 20, 30])
plt.xticks([1, 2, 3])
plt.show()
plt.bar(groupNames, counts)
plt.show()
l = [1, 2, 5, 62, 54, 234, 212, 55, 887, 445, 234, 11, 4325, 4345, 643, 451, 1, 3, 325, 87, 98, 3]
plt.hist(l, bins = [1, 100, 1000, 10000])
plt.xticks([1, 100, 1000, 10000])
plt.show()
dfOccupation = df['Occupation']
plt.hist(dfOccupation, bins = list(range(0, 21)))
plt.xticks(range(0, 21))
plt.show()
# Scatter Plot
plt.scatter([1, 2, 3], [4, 5, 6])
plt.show()
plt.figure(figsize = (7, 7))
plt.pie(counts, labels = groupNames)
plt.show()