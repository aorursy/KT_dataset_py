import sys
!{sys.executable} -m pip install pydotplus
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  

from IPython.display import Image  
import pydotplus
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
y = iris.target
tree = DecisionTreeClassifier()
tree.fit(df, y)
dot_data = StringIO()
export_graphviz(tree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())