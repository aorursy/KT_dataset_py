# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals.six import StringIO
import numpy as np
import graphviz
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz

#Build a simple data set with y = x + random
nPoints = 100
#x values for plotting
xPlot = [(float(i)/float(nPoints) - 0.5) for i in range(nPoints + 1)]
#x needs to be list of lists.
x = [[s] for s in xPlot]
#y (labels) has random noise added to x-value
#set seed
numpy.random.seed(1)
y = [s + numpy.random.normal(scale=0.1) for s in xPlot]
plt.rcParams.update({'font.size': 14})
plt.rcParams['figure.figsize'] = (12.0, 7.0)
plt.plot(xPlot,y)
plt.axis('tight')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Figure 1: Simple dataset')
plt.show()
simpleTree = DecisionTreeRegressor(max_depth=1)
simpleTree.fit(x, y)
#draw the decision tree result with graphviz
dot_data = export_graphviz(simpleTree,out_file = None,rounded = True,filled = True)
graph = graphviz.Source(dot_data)
graph.render() 
graph
#compare prediction from tree with true values
yHat = simpleTree.predict(x)
plt.figure()
plt.plot(xPlot, y, label='True y')
plt.plot(xPlot, yHat, label='Tree Prediction ', linestyle='--');
plt.legend(bbox_to_anchor=(1,0.2))
plt.axis('tight')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Figure 2: Decision Tree Split Depth=1')
plt.show()
simpleTree2 = DecisionTreeRegressor(max_depth=2)
simpleTree2.fit(x, y);
#draw the tree
dot_data = export_graphviz(simpleTree2,out_file = None,rounded = True,filled = True)
graph = graphviz.Source(dot_data)
graph.render() 
graph
#compare prediction from tree with true values
yHat = simpleTree2.predict(x)
plt.figure()
plt.plot(xPlot, y, label='True y')
plt.plot(xPlot, yHat, label='Tree Prediction ', linestyle='--')
plt.legend(bbox_to_anchor=(1,0.2))
plt.axis('tight')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Figure 3: Decision Tree Split Depth=2')
plt.show()
#split point calculations - try every possible split point to
#find the best one
sse = []
xMin = []
for i in range(1, len(xPlot)):
     #divide list into points on left and right of split point
     lhList = list(xPlot[0:i])
     rhList = list(xPlot[i:len(xPlot)])
     #calculate averages on each side
     lhAvg = sum(lhList) / len(lhList)
     rhAvg = sum(rhList) / len(rhList)
     #calculate sum square error on left, right and total
     lhSse = sum([(s - lhAvg) * (s - lhAvg) for s in lhList])
     rhSse = sum([(s - rhAvg) * (s - rhAvg) for s in rhList])
     #add sum of left and right to list of errors
     sse.append(lhSse + rhSse)
     xMin.append(max(lhList))
#SSE is sum of squared error
plt.plot(range(1, len(xPlot)), sse)
plt.xlabel('Split Point Index')
plt.ylabel('Sum Squared Error')
plt.title('Figure 4: SSE resulting from every possible split point location')
plt.show()
#minSse = min(sse)
#idxMin = sse.index(minSse)
#print(xMin[idxMin])
#increase the depth to 6
simpleTree6 = DecisionTreeRegressor(max_depth=6)
simpleTree6.fit(x, y);
#compare prediction from tree with true values
yHat = simpleTree6.predict(x)
plt.figure()
plt.plot(xPlot, y, label='True y')
plt.plot(xPlot, yHat, label='Tree Prediction ', linestyle='--')
plt.legend(bbox_to_anchor=(1,0.2))
plt.axis('tight')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Figure 5: Decision Tree Split Depth=6')
plt.show()