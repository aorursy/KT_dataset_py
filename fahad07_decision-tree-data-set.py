import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Importing data set Positiion salaries 
dataset = pd.read_csv("../input/salary-datase-report/salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values
print(dataset)
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)
#Fit the regressor object to the dataset.
regressor.fit(X,y)
X_grid = np.arange(min(x), max(x), 0.01) 
Y_grid = X_grid.reshape((len(X_grid), 1)) 
plt.scatter(x, y, color = 'blue') 
plt.plot(X_grid, regressor.predict(X_grid), 
color = 'green') 
plt.title('Random Forest Regression') 
plt.xlabel('Position level') 
plt.ylabel('Salary') 
plt.show()

# adapted from https://plot.ly/python/plotly-express/
import plotly.express as px
iris = px.data.iris()
fig = px.scatter(iris, x="sepal_width", y="sepal_length")
fig.show()
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

train = pd.DataFrame(
    np.random.rand(100, 3),
    columns=['X1','X2','y'])

X = train[['X1','X2']]
y = train[['y']]

model = DecisionTreeRegressor().fit(X,y)
print("The first 5 predictions:",model.predict(X.head(5)))
# Adapted from https://scikit-learn.org/stable/modules/tree.html#tree
from sklearn.datasets import load_iris
from sklearn import tree
import graphviz
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")
dot_data = tree.export_graphviz(clf, out_file=None,
feature_names=iris.feature_names,
class_names=iris.target_names,
filled=True, rounded=True,
special_characters=True)
graph = graphviz.Source(dot_data)
graph
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# adapted from https://plot.ly/python/plotly-express/
import plotly.express as px
gapminder = px.data.gapminder()
fig = px.scatter(gapminder.query("year==2007"), x="gdpPercap", y="lifeExp", size="pop", color="continent",
           hover_name="country", log_x=True, size_max=60)
fig.show()