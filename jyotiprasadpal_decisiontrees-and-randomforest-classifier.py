import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.ensemble import RandomForestClassifier



from subprocess import call
data = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

data.sample(3)
data.info()
data.Outcome.value_counts()
X=data.loc[:, 'Pregnancies':'Age']

y=data.loc[:, 'Outcome']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42)
y_train.value_counts()
#create a gini based tree

dt = DecisionTreeClassifier(criterion='gini', min_samples_leaf=50)

dt.fit(X_train, y_train)
def display_tree(estimator, out_file_name, feature_names, fig_size=(16, 18)):

    dot_out_file = out_file_name + '.dot'

    png_out_file = out_file_name + '.png'

    export_graphviz(decision_tree=estimator, out_file=dot_out_file, feature_names=feature_names, class_names=True, rounded = True, proportion = False, precision = 2, filled = True)

    # Convert to png

    call(['dot', '-Tpng', dot_out_file, '-o', png_out_file, '-Gdpi=600'])



    # Display in python

    fig, ax = plt.subplots(figsize=fig_size)

    ax.imshow(plt.imread(png_out_file))

    ax.axis('off') #don't show the x and y axis



display_tree(dt,'tree', X.columns)
rfc = RandomForestClassifier(max_depth = 3, n_estimators=10, oob_score=True)

rfc.fit(X_train, y_train)
#print oob error

print(f'Out of bag score: ', rfc.oob_score_, end='\n\n')



#find feature importances

feature_importances = pd.Series(rfc.feature_importances_, index=X_train.columns).sort_values(ascending=False)

print(feature_importances)



fig, ax = plt.subplots(figsize=(12, 5))

sns.barplot(x=X_train.columns, y=rfc.feature_importances_)
# Extract single tree and visualize the tree

estimator = rfc.estimators_[5]

display_tree(estimator,'tree_limited', X.columns)