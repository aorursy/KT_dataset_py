import numpy as np

import pandas as pd

from sklearn.cluster import k_means
data = pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")
y = k_means(data, 2)[1]
from sklearn import tree
classifier = tree.DecisionTreeClassifier(

    min_samples_leaf = 20

)

classifier.fit(data, y)
print(classifier.score(data, y))
tree.plot_tree(classifier)