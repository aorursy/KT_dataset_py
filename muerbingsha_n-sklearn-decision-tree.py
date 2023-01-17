# source: https://www.geeksforgeeks.org/python-decision-tree-regression-using-sklearn/





# Decision tree 

# supervised learning algorithm



import numpy as np

import matplotlib.pyplot as plt 

import pandas as pd 

dataset = np.array( 

[['Asset Flip', 100, 1000], 

['Text Based', 500, 3000], 

['Visual Novel', 1500, 5000], 

['2D Pixel Art', 3500, 8000], 

['2D Vector Art', 5000, 6500], 

['Strategy', 6000, 7000], 

['First Person Shooter', 8000, 15000], 

['Simulator', 9500, 20000], 

['Racing', 12000, 21000], 

['RPG', 14000, 25000], 

['Sandbox', 15500, 27000], 

['Open-World', 16500, 30000], 

['MMOFPS', 25000, 52000], 

['MMORPG', 30000, 80000] 

]) 



print(dataset)
x = dataset[:, 1:2].astype(int)

print(x)

y = dataset[:, 2].astype(int)

print(y)
from sklearn.tree import DecisionTreeRegressor



regressor = DecisionTreeRegressor(random_state=0)

regressor.fit(x, y)

y_pred = regressor.predict([[3750]])

print(y_pred)
# visualize 

x_grid = np.arange(min(x), max(x), 0.01)

x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x, y, color='red')

plt.plot(x_grid, regressor.predict(x_grid), color='blue')

plt.title('Profit to Production Cost (Decision Tree Regression)')

plt.xlabel('Production Cost')

plt.ylabel('Profit')

plt.show()
# The tree is finally exported and shown in the TREE STRUCTURE below, visualized using http://www.webgraphviz.com/ by copying the data from the ‘tree.dot’ file



from sklearn.tree import export_graphviz

export_graphviz(regressor, out_file='tree.dot', feature_names=['Production Cost'])