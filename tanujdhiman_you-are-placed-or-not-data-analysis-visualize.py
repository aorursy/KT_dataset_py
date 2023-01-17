# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import matplotlib.pyplot as plt # visualizations
data = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data.head(5)
data.info()
data.gender.value_counts()
data.ssc_b.value_counts()
data.hsc_b.value_counts()
data.hsc_s.value_counts()
data.degree_t.value_counts()
data.workex.value_counts()
data.specialisation.value_counts()
data.status.value_counts()
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
data["gender"] = label.fit_transform(data["gender"])
data["ssc_b"] = label.fit_transform(data["ssc_b"])
data["hsc_b"] = label.fit_transform(data["hsc_b"])
data["hsc_s"] = label.fit_transform(data["hsc_s"])
data["degree_t"] = label.fit_transform(data["degree_t"])
data["workex"] = label.fit_transform(data["workex"])
data["specialisation"] = label.fit_transform(data["specialisation"])
data["status"] = label.fit_transform(data["status"])
data.head()
data.hist(figsize = (20, 20))
plt.show()
import seaborn as sns
sns.heatmap(data.corr())
plt.show()
import plotly.express as px
data_original = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
fig = px.scatter(data_original, x="salary", 
                 color="degree_p",
                 size='degree_p', 
                 hover_data=['gender', 'ssc_p', 'hsc_p', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'degree_p', 
                            'specialisation', 'mba_p', 'status', 'etest_p'], 
                 title = "Salary Plot")
fig.show()
fig = px.scatter(data_original, x="ssc_p", 
                 color="degree_p",
                 size='degree_p', 
                 hover_data=['gender', 'hsc_p', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'degree_p', 
                            'specialisation', 'mba_p', 'status', 'etest_p'], 
                 title = "ssc_p Plot")
fig.show()
data.info()
X = data.iloc[:, 0:13].values
y = data.iloc[:, 13].values
y
X
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X , y)
print(model.feature_importances_) 
feat_importances = pd.Series(model.feature_importances_)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
print("Maximum important feature index is : ", model.feature_importances_.argmax()) 
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X, y)
y_pred = clf.predict(X)
y_pred
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)
cm