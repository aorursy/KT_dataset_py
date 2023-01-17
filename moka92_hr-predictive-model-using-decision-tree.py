import numpy as np

import pandas as pd 

from sklearn import tree

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split 

from sklearn.metrics import precision_recall_fscore_support

hr_dataset = pd.read_csv("../input/HR_comma_sep.csv")

hr_dataset.head()
target = ["left"] 

features = list(set(hr_dataset.columns.values) - set(target))



hr_data_features = hr_dataset[features]

hr_data_target = hr_dataset[target]



hr_data_target.head()

hr_data_features.head()
dept_encoded = LabelEncoder().fit_transform(hr_data_features['sales'])

salary_encoded = LabelEncoder().fit_transform(hr_data_features['salary'])



hr_data_features = hr_data_features.drop(['sales', 'salary'], axis=1)



hr_data_features['dept'] = pd.Series(dept_encoded)

hr_data_features['salary'] = pd.Series(salary_encoded)
x_train, x_test, y_train, y_test =train_test_split(hr_data_features,hr_data_target , test_size=0.2)

clf = tree.DecisionTreeClassifier()

clf.fit(x_train, y_train)
pred_vals = clf.predict(x_test)

model_eval = precision_recall_fscore_support(y_test, pred_vals, average='weighted')

model_eval
import graphviz 

dot_data = tree.export_graphviz(clf, out_file=None) 

graph = graphviz.Source(dot_data) 

graph.render("hr_analysis") 

graph 