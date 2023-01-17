import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_excel('../input/covid19/dataset.xlsx')

data.columns = map(lambda x: x.lower().strip().replace('\xa0', '_').replace(' ', '_'), data.columns)

data.head()
import missingno as msno

msno.matrix(data, labels=True, fontsize=9)
useful_data = data[['sars-cov-2_exam_result', 'hematocrit', 'hemoglobin', 'platelets', 'mean_platelet_volume', 'red_blood_cells', 'lymphocytes', 'mean_corpuscular_hemoglobin_concentration_(mchc)', 'leukocytes', 'basophils', 'mean_corpuscular_hemoglobin_(mch)', 'eosinophils', 'mean_corpuscular_volume_(mcv)', 'monocytes', 'red_blood_cell_distribution_width_(rdw)']].dropna()



data_length = len(data)

useful_data_length = len(useful_data)

print('Number of samples: {}'.format(data_length))

print('Number of useful samples: {}'.format(useful_data_length))

print('Useful portion of the dataset: {:.2f}%'.format((useful_data_length / data_length) * 100.0))
sns.countplot(x='sars-cov-2_exam_result', data=useful_data)
# Here we separate the data into a matrix X and a vector y (data and target)

X = useful_data.loc[:, useful_data.columns != 'sars-cov-2_exam_result']

y = useful_data[['sars-cov-2_exam_result']]



from imblearn.under_sampling import RandomUnderSampler

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rus = RandomUnderSampler(random_state=42, sampling_strategy='majority')

X_test, y_test = rus.fit_resample(X_test, y_test) # Here, we undersample the majority class of the test set
from sklearn import tree

from sklearn.model_selection import cross_val_score



decision_tree_classifier = tree.DecisionTreeClassifier(max_depth=3, class_weight='balanced')

decision_tree_scores = cross_val_score(decision_tree_classifier, X, y, cv=5)

print("Scores: {}".format(decision_tree_scores))

print("Mean: %0.2f" % decision_tree_scores.mean())

print("Standard deviation: %0.2f" % decision_tree_scores.std())
from sklearn.metrics import plot_confusion_matrix



decision_tree_classifier.fit(X_train, y_train)

plot_confusion_matrix(decision_tree_classifier, X_test, y_test, normalize='true')
import graphviz

decision_tree_classifier.fit(X, y)

dot_data = tree.export_graphviz(decision_tree_classifier, out_file=None, 

                     rotate=True,

                     feature_names=X.columns,

                     class_names=decision_tree_classifier.classes_,

                     filled=True, rounded=True,  

                     special_characters=True)

graph = graphviz.Source(dot_data)

graph
from sklearn.ensemble import RandomForestClassifier



random_forest_classifier = RandomForestClassifier(n_estimators=30, max_depth=3, class_weight='balanced_subsample')

random_forest_scores = cross_val_score(random_forest_classifier, X, y.values.ravel(), cv=5)

print("Scores: {}".format(random_forest_scores))

print("Mean: %0.2f" % random_forest_scores.mean())

print("Standard deviation: %0.2f" % random_forest_scores.std())
from sklearn.metrics import plot_confusion_matrix



random_forest_classifier.fit(X_train, y_train)

plot_confusion_matrix(random_forest_classifier, X_test, y_test, normalize='true')
from imblearn.ensemble import BalancedRandomForestClassifier



balanced_random_forest_classifier = BalancedRandomForestClassifier(n_estimators=30, max_depth=3, sampling_strategy='majority')

balanced_random_forest_scores = cross_val_score(balanced_random_forest_classifier, X, y.values.ravel(), cv=5)

print("Scores: {}".format(balanced_random_forest_scores))

print("Mean: %0.2f" % balanced_random_forest_scores.mean())

print("Standard deviation: %0.2f" % balanced_random_forest_scores.std())
from sklearn.metrics import plot_confusion_matrix



balanced_random_forest_classifier.fit(X_train, y_train)

plot_confusion_matrix(balanced_random_forest_classifier, X_test, y_test, normalize='true')