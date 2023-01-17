import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
concrete_df = pd.read_csv('../input/concrete_data.csv')

concrete_df.head()
X = concrete_df.drop('csMPa', axis=1)

Y = concrete_df['csMPa']
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
from sklearn.tree import DecisionTreeRegressor
# First weak learner 

tree_reg1 = DecisionTreeRegressor(max_depth=3)

tree_reg1.fit(X_train, y_train)
# Residual Error from the first decision tree

y2 = y_train - tree_reg1.predict(X_train)

y2[:10]
# Second weak learner 

tree_reg2 = DecisionTreeRegressor(max_depth=4)



'''

y2: the residial error of the previous Decision Tree

'''

tree_reg2.fit(X_train, y2)
# Residual error of the second weak learner

y3 = y2 - tree_reg2.predict(X_train)

y3[:10]
# Third weak learner

tree_reg3 = DecisionTreeRegressor(max_depth=4)

tree_reg3.fit(X_train, y3)
y4 = y3 - tree_reg3.predict(X_train)

y4[:10]
y_pred = sum(tree.predict(X_test) for tree in (tree_reg1, 

                                               tree_reg2, 

                                               tree_reg3))
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
#Additional learner

tree_reg4 = DecisionTreeRegressor(max_depth=4)

tree_reg4.fit(X_train, y4) 
y_pred = sum(tree.predict(X_test) for tree in (tree_reg1, 

                                               tree_reg2, 

                                               tree_reg3,

                                               tree_reg4))



r2_score(y_test, y_pred)
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(max_depth=3, 

                                n_estimators=3,

                                learning_rate=1.0)



gbr.fit(X_train, y_train)
y_pred = gbr.predict(X_test)

r2_score(y_test, y_pred)
gbr = GradientBoostingRegressor(max_depth=3, 

                                n_estimators=3,

                                learning_rate=0.1)



gbr.fit(X_train, y_train)

y_pred = gbr.predict(X_test)

r2_score(y_test, y_pred)
gbr = GradientBoostingRegressor(max_depth=3, 

                                n_estimators=40,

                                learning_rate=0.1)



gbr.fit(X_train, y_train)

y_pred = gbr.predict(X_test)

r2_score(y_test, y_pred)