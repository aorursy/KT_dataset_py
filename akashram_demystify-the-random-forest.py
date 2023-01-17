import pandas as pd
import numpy as np
data = pd.read_csv("../input/kc_house_data.csv")
data.head()
from treeinterpreter import treeinterpreter as ti
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
dt = DecisionTreeRegressor()
y = data.iloc[:,2]
x = data.loc[:, data.columns != 'price']
x = x.drop('date',1)
x = x.drop('id', 1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
dt.fit(X_train, y_train)
instances = X_test.loc[[735]]
instances
prediction, bias, contributions = ti.predict(dt, instances)
ft_list = []
for i in range(len(instances)):
    #print("Instance", i)
    print("Bias (trainset mean)", bias[i])
    #print("Feature contributions:")
    for c, feature in sorted(zip(contributions[i], 
                                 x.columns), 
                             key=lambda x: -abs(x[0])):
       ft_list.append((feature, round(c, 2)))
    print("-"*50)
labels, values = zip(*ft_list)
ft_list
import numpy as np                                                               
import matplotlib.pyplot as plt
import seaborn as sns

from pylab import rcParams
rcParams['figure.figsize'] = 25, 25

xs = np.arange(len(labels)) 

sns.barplot(xs, values)

#plt.bar(xs, values, width, align='center')

plt.xticks(xs, labels)
plt.yticks(values)

plt.show()
contributions
prediction
bias
print(bias + np.sum(contributions, axis=1))
top50x = X_train.head(50)
top5x = X_train.head(5)
top50y = y_train.head(50)
top5y = y_train.head(5)
dt1 = DecisionTreeRegressor()
dt1.fit(top5x, top5y)
from sklearn.externals.six import StringIO  
from IPython.display import Image  
#from sklearn.tree import export_graphviz
#import pydotplus
#dot_data = StringIO()
#export_graphviz(dt1, out_file=dot_data,  
#                filled=True, rounded=True,
 #               special_characters=True)
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#Image(graph.create_png())
top5x
top5y
rf.fit(X_train, y_train)
rf_prediction, rf_bias, rf_contributions = ti.predict(rf, instances)
rf_ft_list = []
for i in range(len(instances)):
    print("Bias (trainset mean)", rf_bias[i])
    for c, feature in sorted(zip(rf_contributions[i], 
                                 x.columns), 
                             key=lambda x: -abs(x[0])):
       rf_ft_list.append((feature, round(c, 2)))
    print("-"*50)
rf_labels, rf_values = zip(*rf_ft_list)
rf_ft_list
import numpy as np                                                               
import matplotlib.pyplot as plt

from pylab import rcParams
rcParams['figure.figsize'] = 25, 25

rf_xs = np.arange(len(rf_labels)) 

plt.bar(rf_xs, rf_values, 0.8, align='center')

plt.xticks(rf_xs, rf_labels)
plt.yticks(rf_values)

plt.show()
rf_contributions
rf_prediction
rf_bias
print(rf_bias + np.sum(rf_contributions, axis=1))
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=10)
top5xrf = X_train.head(5)
top5yrf = y_train.head(5)
rf_model.fit(top5xrf, top5yrf)
estimator = rf_model.estimators_[5]
estimator1 = rf_model.estimators_[6]
from sklearn.externals.six import StringIO  
from IPython.display import Image
#from sklearn.tree import export_graphviz
#import pydotplus
#dot_data1 = StringIO()
#export_graphviz(estimator, out_file=dot_data1,  
 #               filled=True, rounded=True,
  #              special_characters=True)
#graph = pydotplus.graph_from_dot_data(dot_data1.getvalue())  
#Image(graph.create_png())
from sklearn.externals.six import StringIO  
from IPython.display import Image  
#from sklearn.tree import export_graphviz
#import pydotplus
#dot_data3 = StringIO()
#export_graphviz(estimator1, out_file=dot_data3,  
 #               filled=True, rounded=True,
  #              special_characters=True)
#graph = pydotplus.graph_from_dot_data(dot_data3.getvalue())  
#Image(graph.create_png())