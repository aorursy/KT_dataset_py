# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
data = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
data.head()
data.shape
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
data["Gender"] = label.fit_transform(data["Gender"])
data.head()
data.hist(figsize=(16, 8))
plt.show()
sns.heatmap(data.corr(), annot=True)
plt.show()
sns.violinplot(data['Age'], data['Gender'])
sns.despine()
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(data['Age'],data['Annual Income (k$)'], s=data['Spending Score (1-100)']) 
plt.show()
var=data.groupby(['Gender']).sum().stack()
temp=var.unstack()
type(temp)
x_list = temp['Annual Income (k$)']
label_list = temp.index
plt.axis("equal")
plt.pie(x_list,labels=label_list,autopct="%1.1f%%") 
plt.title("Pastafarianism expenses") 
plt.show()
fig = px.scatter(data, x="Age", y="Annual Income (k$)", 
                 color="Age",
                 size='Age', 
                 hover_data=[ 'Spending Score (1-100)'], 
                 title = "Age wise Annual Income")
fig.show()
fig2 = go.Figure(data=go.Scatter(x=data['Age'],
                                y=data['Annual Income (k$)'],
                                mode='markers',
                                marker_color=data['Age'],
                                text=data['Spending Score (1-100)'])) # hover text goes here

fig2.update_layout(title='Age wise Annual Income')
fig2.show()
X = data.iloc[:, 0:4]
y = data.iloc[:, 4]
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators = 5, 
                                        criterion ='entropy', max_features = 2)
clf.fit(X, y)
feature_importance = clf.feature_importances_ 
feature_importance_normalized = np.std([tree.feature_importances_ for tree in 
                                        clf.estimators_], 
                                        axis = 0) 
plt.bar(X.columns, feature_importance_normalized) 
plt.xlabel('Feature Labels') 
plt.ylabel('Feature Importances') 
plt.title('Comparison of different Feature Importances') 
plt.show()
X_new = data.iloc[:, [2]].values
y_new = data.iloc[:, 4].values
from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor(random_state = 0)
reg.fit(X_new, y_new)
y_pred = reg.predict(X_new)
y_pred
y_test = reg.predict([[28]])
y_test