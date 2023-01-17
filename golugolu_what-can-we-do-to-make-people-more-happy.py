# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
year2015 = pd.read_csv('../input/2015.csv')
year2015.head()
countries = year2015['Country']

region = year2015['Region']
X2015 = year2015.iloc[:,5:]

y2015 = year2015['Happiness Score']
X2015.head()
# for 2015

from sklearn.ensemble import ExtraTreesRegressor

import matplotlib.pyplot as plt

forest = ExtraTreesRegressor(n_estimators=1000,

                              random_state=42)

forest.fit(X2015, y2015)

importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(X2015.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the feature importances of the forest

plt.figure(figsize=(10,8))

plt.title("Feature importances")

plt.bar(range(X2015.shape[1]), importances[indices],

       color="b", yerr=std[indices], align="center")

plt.xticks(range(X2015.shape[1]), indices)

plt.xlim([-1, X2015.shape[1]])

plt.show()
# as we know now features like health economy and dystpia residue affect the most 

# so let's do indivisual test with happiness and the 3 features 



plt.scatter(X2015['Economy (GDP per Capita)'], y2015)

plt.scatter(X2015['Health (Life Expectancy)'], y2015)

plt.scatter(X2015['Dystopia Residual'], y2015)

plt.figure(figsize=(12,10))

plt.show()
# the above graph we can see Economy and health are pretty compact and linear and hence a linear model

# will give a good prediction

# whearas the Dystopia residua is scattered which can be seen in the above relation graph

# the importance decreases significantly

from sklearn.ensemble.partial_dependence import plot_partial_dependence

from sklearn.ensemble.partial_dependence import partial_dependence

from sklearn.ensemble import GradientBoostingRegressor

from mpl_toolkits.mplot3d import Axes3D

names = X2015.columns

clf = GradientBoostingRegressor(n_estimators=1000, max_depth=10,

                                    learning_rate=0.1, loss='huber',

                                    random_state=42)

clf.fit(X2015, y2015)

features = [0, 5, 1, 2, (0, 2)]

fig, axs = plot_partial_dependence(clf, X2015, features,

                                       feature_names=names,

                                       n_jobs=3 ,grid_resolution=100, figsize = (12,10))







fig = plt.figure(figsize=(8,6))

target_feature = (0, 2)

pdp, axes = partial_dependence(clf, target_feature,

                                  X=X2015, grid_resolution=1000)

XX, YY = np.meshgrid(axes[0], axes[1])

Z = pdp[0].reshape(list(map(np.size, axes))).T

ax = Axes3D(fig)

surf = ax.plot_surface(XX, YY, Z, rstride=1, cstride=1, cmap = plt.get_cmap('viridis') )

ax.set_xlabel(names[target_feature[0]])

ax.set_ylabel(names[target_feature[1]])

ax.set_zlabel('Happiness')

ax.view_init(elev=22, azim=122)

plt.colorbar(surf)

plt.suptitle('Partial dependence')

plt.subplots_adjust(top=0.9)

plt.show()

# let's find the influence of government

trust = X2015['Trust (Government Corruption)']

plt.plot(trust)

plt.show()
# lets create a scatter plot of happiness and trust 



plt.scatter(trust , y2015)

plt.xlabel('trust')

plt.ylabel('happiness')
# well we can clearly see there is exact realtion, so we might start finding interanl relation 

# and let's see how well that works out 

from scipy import stats

X = [trust, y2015]

from sklearn.ensemble import IsolationForest

classifier = IsolationForest(max_samples=158)

classifier.fit(X)

#scores_pred = clf.decision_function(X)

#threshold = stats.scoreatpercentile(scores_pred,

#                                           100 * outliers_fraction)

y_pred = clf.predict(X)

n_errors = (y_pred != ground_truth).sum()