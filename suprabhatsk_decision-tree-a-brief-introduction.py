import numpy as np
import pandas as pd
label = ['P', 'NP', 'P','NP', 'NP', 'NP', 'P','NP', 'NP', 'NP', 'NP','NP', 'P', 'NP', 'P','NP', 'P', 'NP', 'P','NP', 'P', 'NP', 'P','NP','P', 'NP', 'NP','NP', 'P', 'NP', 'P', 'NP', 'P','NP', 'P', 'NP', 'NP','NP', 'P', 'NP', 'P','NP',  'P', 'NP', 'NP','P', 'P', 'P', 'P','P', 'P', 'NP', 'P','NP','P', 'NP', 'P','NP', 'P', 'NP']

sex = ['M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'F', 'F', 'F', 'F','M', 'F', 'F', 'F', 'F', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'F', 'F', 'F', 'F']

cls = ['IX', 'IX', 'IX', 'IX', 'IX', 'IX', 'IX', 'IX', 'IX', 'IX', 'IX', 'IX', 'X', 'IX', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'IX', 'IX', 'IX', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'IX', 'IX', 'IX', 'X', 'IX', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'IX']

height = [5, 6, 5, 6, 5, 6, 5, 6,5, 6, 5, 6,6,6, 5, 5, 6, 5, 6, 5, 6, 5, 6,5, 6, 5, 6,6,6, 5,5, 6, 5, 6, 5, 6, 5, 6,5, 6, 5, 6,6,6, 5,5, 6, 5, 6, 5, 6, 5, 6,5, 6, 5, 6,6,6, 5]

weight = [50, 58, 50, 58, 50, 58, 50, 58, 50, 58, 50, 58, 50, 58, 50, 58, 50, 58, 50, 58,50, 58, 50, 58, 50, 58, 50, 58, 50, 58,50, 58, 50, 58, 50, 58, 50, 58, 50, 58,50, 58, 50, 58, 50, 58, 50, 58, 50, 58,50, 58, 50, 58, 50, 58, 50, 58, 50, 58]

# Data frame created using the above list
df = pd.DataFrame({'Weight': weight, 'Height': height, 'Class': cls, 'Sex': sex, 'label': label})
df.info()
df.head() 
# We have converted the categorical variable to numerical variable
code = {'P': 1, 'NP': 0}
df['new_label'] = df['label'].map(code) 
new_s = pd.get_dummies(df.Sex)
new_c = pd.get_dummies(df.Class)
df[new_s.columns] = new_s
df[new_c.columns] = new_c
df.head()
new_df = df[['F', 'M', 'IX', 'X', 'Weight', 'Height','new_label']]
new_df.head()
from sklearn import tree
model = tree.DecisionTreeClassifier(random_state = 23)

feature = new_df.drop(['new_label'], axis = 1)
label = new_df.new_label
model.fit(feature, label) 
import graphviz
graph = tree.export_graphviz(model, out_file=None, filled=True)
graphviz.Source(graph) 
model.fit(feature, label) 
from sklearn import tree
model1 = tree.DecisionTreeClassifier(min_samples_leaf = 5, random_state = 23)
model1.fit(feature, label) 
import graphviz
graph1 = tree.export_graphviz(model1, out_file=None, filled=True)
graphviz.Source(graph1) 
# Did you see the change in the tree structure. Now, lets try some other parameters and see how does it affect the tree structure.
from sklearn import tree
model2 = tree.DecisionTreeClassifier(min_samples_split = 15, random_state = 23)
model2.fit(feature, label) 
import graphviz
graph2 = tree.export_graphviz(model2, out_file=None, filled=True)
graphviz.Source(graph2) 
# Did you see the change in the tree structure. 
# Now, lets try to find out the best parameter by using the grid search CV
from sklearn.model_selection import GridSearchCV

param1 = {'min_samples_leaf': [2,3,4],
         'min_samples_split': [2,3,5,10,12,14],
         'max_depth': [2,3,4,56],
         'criterion': ['gini', 'entropy'],
         'max_features':[2,3,4]}

CV = GridSearchCV(model, param1)
CV.fit(feature, label)
best = CV.best_estimator_
best
graphCV = tree.export_graphviz(best, filled=True, out_file=None)
graphviz.Source(graphCV)
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

accuracy_score(best.predict(feature), label)
cross_val_score(best, feature, label, cv=5)
# The accuracy score is 85% which is pretty good and cross validation score seems to be pretty consistant. 
# Hopefully this kenrel might help you to understand decsion tree better. Let me know if you have any questions.