from sklearn import neighbors, datasets

#load the Iris dataset
iris = datasets.load_iris()

# iris
# pandas is a lib for data analysis and modelling
import pandas as pd

# it uses "dataframes" - so prep a dataframe
iris_df = pd.DataFrame(data=iris.data[:,:], columns=iris.feature_names[:])

# add the classes to the dataframe as the last column
iris_df['target'] = pd.Series(iris.target)

# iris_df
# seaborn is a statistical data visualisation package
import seaborn as sns

# it uses the venerable matplotlib
import matplotlib
import matplotlib.pyplot as plt

# set some visual stuff
sns.set(style="white", color_codes=True)

# plot the classes for two features at a time...
sns.FacetGrid(iris_df, hue="target", size=5) \
   .map(plt.scatter, "sepal length (cm)", "sepal width (cm)") \
   .add_legend()
# representing the spread of classes for an individual feature
sns.boxplot(x="target", y="sepal length (cm)", data=iris_df)
# visualise all each feature vs. every other feature, as well as each feature spreads at once...
# there's a couple of error messages produced here but the charts should plot just fine.
sns.pairplot(iris_df, hue="target", size=2, diag_kind="kde")
# set a value for k (no. of nearest neighbours)
k = 5

# create a new classifier instance
clf = neighbors.KNeighborsClassifier(k, weights='distance')

# see what defaults we got
# clf
# visualise the areas that will "capture" a data point

# slice two features out of the data
feature_x_col = 'sepal length (cm)'
feature_y_col = 'petal length (cm)'

feature_x = iris_df.loc[:, feature_x_col]
feature_y = iris_df.loc[:, feature_y_col]

# and grab the target
target = iris_df.loc[:, 'target']

# feature_x
# feature_y
# target

# fit the model based on all the data for two features
clf.fit(iris_df.loc[:, [feature_x_col, feature_y_col]], target)
# make a grid so we can run the classifier over each point and colour it in with the predicted class
step = 0.02 # distance between points

# min and max values for the features
x_min, x_max = feature_x.min() - 1, feature_x.max() + 1
y_min, y_max = feature_y.min() - 1, feature_y.max() + 1

# generate a grid, using numpy
import numpy as np
xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                     np.arange(y_min, y_max, step))

# use the model to predict a class for all the points on the grid - flatten the 2d structure first
predictions = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# predictions

# now reshape the flattened structure back into the 2-d structure we need
predictions_xy = predictions.reshape(xx.shape)

# predictions_xy
# plot the grid, predictions and training data

from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFCCCC', '#CCFFCC','#CCCCFF'])
cmap_bold = ListedColormap(['#990000', '#009900','#000099'])

plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
plt.pcolormesh(xx, yy, predictions_xy, cmap=cmap_light)
 
# Plot also the training points
plt.scatter(feature_x, feature_y, c=target, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("3-Class classification (k = %i)" % (k))
plt.show()
# now try all combinations of features, see which really works best!

from sklearn.model_selection import cross_val_score

from itertools import chain, combinations
i = set([0,1,2,3])
results = []
for feature_subset in chain.from_iterable(combinations(i, r) for r in range(len(i) + 1)):
    if feature_subset:
        med_score = np.median(cross_val_score(clf, iris.data[:, feature_subset], iris.target, cv=3));
        print('{:1.2f} {}'.format(med_score, feature_subset))