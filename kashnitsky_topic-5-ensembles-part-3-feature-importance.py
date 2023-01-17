import numpy as np

import pandas as pd



from sklearn.datasets import load_iris

import matplotlib.pyplot as plt

import seaborn as sns



iris = load_iris()

data = iris['data']

target = iris['target']
data = pd.DataFrame(data, columns=iris['feature_names'])

data.head()
target = pd.Series(target).map({0: 0, 1: 0, 2: 1})
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=3, max_depth=3, random_state=17)

rfc.fit(data, target);
tree_list = rfc.estimators_
from sklearn import tree

plt.figure(figsize=(16,12))

tree.plot_tree(tree_list[0], filled=True, feature_names=iris['feature_names'], 

               class_names=['Y', 'N'], node_ids=True);
plt.figure(figsize=(16,12))

tree.plot_tree(tree_list[1], filled=True, feature_names=iris['feature_names'], 

               class_names=['Y', 'N'], node_ids=True);
plt.figure(figsize=(6,4))

tree.plot_tree(tree_list[2], filled=True, feature_names=iris['feature_names'], 

               class_names=['Y', 'N'], node_ids=True);
print(iris['feature_names'])

print(rfc.feature_importances_)
from sklearn.ensemble.forest import RandomForestRegressor
hostel_data = pd.read_csv("../input/hostel_factors.csv")

features = {"f1":u"Staff",

"f2":u"Hostel booking",

"f3":u"Check-in and check-out",

"f4":u"Room condition",

"f5":u"Shared kitchen condition",

"f6":u"Shared space condition",

"f7":u"Extra services",

"f8":u"General conditions & conveniences",

"f9":u"Value for money",

"f10":u"Customer Co-creation"}



forest = RandomForestRegressor(n_estimators=1000, max_features=10,

                                random_state=0)



forest.fit(hostel_data.drop(['hostel', 'rating'], axis=1), 

           hostel_data['rating'])

importances = forest.feature_importances_



indices = np.argsort(importances)[::-1]

# Plot the feature importancies of the forest

num_to_plot = 10

feature_indices = [ind+1 for ind in indices[:num_to_plot]]



# Print the feature ranking

print("Feature ranking:")

  

for f in range(num_to_plot):

    print("%d. %s %f " % (f + 1, 

            features["f"+str(feature_indices[f])], 

            importances[indices[f]]))

plt.figure(figsize=(15,5))

plt.title(u"Feature Importance")

bars = plt.bar(range(num_to_plot), 

               importances[indices[:num_to_plot]],

       color=([str(i/float(num_to_plot+1)) 

               for i in range(num_to_plot)]),

               align="center")

ticks = plt.xticks(range(num_to_plot), 

                   feature_indices)

plt.xlim([-1, num_to_plot])

plt.legend(bars, [u''.join(features["f"+str(i)]) 

                  for i in feature_indices]);