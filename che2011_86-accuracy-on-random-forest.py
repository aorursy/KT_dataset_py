import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
# Open file.
df = pd.read_csv('../input/ks-projects-201801.csv')
df.head()
# Create features.

df.loc[:,'goal_reached'] = df['pledged'] / df['goal'] # Pledged amount as a percentage of goal.

df.loc[df['backers'] == 0, 'backers'] = 1 # Convert zero backers to 1 to prevent undefined division.
df.loc[:,'pledge_per_backer'] = df['pledged'] / df['backers'] # Pledged amount per backer.

# Create the subset.
df_sub = df[['category', 'main_category', 'goal_reached', 'pledge_per_backer', 'state', 'country']]
# Check shape and data types.
print(df_sub.shape)
print('\n')
print(df_sub.dtypes)
# Check for nulls.
df_sub.isnull().sum()
# Code the goal_reached and pledge_per_backer features as 0 or 1.
df_sub['goal_reached_cat'] = np.where(df_sub['goal_reached']>0.5, 1, 0)
df_sub['pledge_per_backer_cat'] = np.where(df_sub['pledge_per_backer']>df_sub['pledge_per_backer'].mean(), 1, 0)

# Create the final subset of features that will be used to predict if a kickstarter project will be successfully funded.
df_final = df_sub[['category', 'main_category', 'goal_reached_cat', 'pledge_per_backer_cat', 'state', 'country']]
# Set variables.
X = df_final.drop(['state'], axis=1)
X = pd.get_dummies(X)
y = df_final['state']

# Set the start time for execution speed.
import time
start_time = time.clock() 

# Split into train and test sets.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Fit the model.
from sklearn import tree
dtc = tree.DecisionTreeClassifier(criterion='entropy', max_features=3, max_depth=4, random_state = 1337)
dtc.fit(X_train, y_train)

# Run predictions.
y_predict = dtc.predict(X_test)

# Return accuracy score.
from sklearn.metrics import accuracy_score
print('Score:', accuracy_score(y_test, y_predict))
print('Runtime: '+'%s seconds'% (time.clock() - start_time)) # End time for execution speed.
from IPython.display import Image
import pydotplus
import graphviz

dot_data = tree.export_graphviz(dtc, out_file=None, filled=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
from sklearn import ensemble
from sklearn.model_selection import cross_val_score

start_time = time.clock() # Start time.
rfc = ensemble.RandomForestClassifier()
print('Scores: ', cross_val_score(rfc, X, y, cv=10)) # Return cross validation scores.
print('Runtime: '+'%s seconds'% (time.clock() - start_time)) # End time.
