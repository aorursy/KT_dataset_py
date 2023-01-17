import numpy as np



X = [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
split_index = 9

X_1 = X[0:split_index]

X_2 = X[split_index:]

assert X_1 + X_2 == X
print('X <= ' + str(split_index - 1) + ': ' + str(X_1))

print('X >  ' + str(split_index - 1) + ': ' + str(X_2))
import math

import scipy.stats as stats



def probabilities(X):

    p_1 = np.sum(X) / len(X)

    p_0 = 1 - p_1

    return [p_0, p_1]



assert np.sum(probabilities(X)) == 1

    

def entropy(X):

    return stats.entropy(probabilities(X), base=2)



assert entropy([0, 0]) == entropy([1, 1]) == 0

assert entropy([1, 1, 0, 0]) == entropy([0, 1, 0, 1]) == 1
S0 = entropy(X)

S1 = entropy(X_1)

S2 = entropy(X_2)
print('S0       : ' + str(S0))

print('S(X <= 8): ' + str(S1))

print('S(X >  8): ' + str(S2))
N = len(X)

N1 = len(X_1)

N2 = len(X_2)



assert N1 + N2 == N
information_gain = S0 - (N1/N)*S1 - (N2/N)*S2

print('Information Gain: ' + ('%0.4f' % information_gain))
num_samples = 2000

multiplier = 10



x_1 = np.random.random_sample(num_samples) * multiplier

x_2 = np.random.random_sample(num_samples) * multiplier

x_3 = np.random.random_sample(num_samples) * multiplier

x_4 = np.random.random_sample(num_samples) * multiplier

x_5 = np.random.random_sample(num_samples) * multiplier

x_6 = np.random.random_sample(num_samples) * multiplier

x_7 = np.random.random_sample(num_samples) * multiplier

x_8 = np.random.random_sample(num_samples) * multiplier

x_9 = np.random.random_sample(num_samples) * multiplier

X = np.array([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9]).T

y = (np.power(x_1, 2) / 4 + np.power(x_2, 2) / 9) <= 16
from sklearn.model_selection import train_test_split



X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.3,random_state=17)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



tree = DecisionTreeClassifier(max_features=9, criterion='gini', min_samples_leaf=2, max_depth=100)

tree.fit(X_train, y_train)

tree_pred = tree.predict(X_holdout)

accuracy_score(y_holdout, tree_pred)
from sklearn.datasets import load_iris 

iris_data = load_iris()
X = iris_data['data']

y = iris_data['target']

assert len(X) == len(y)
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=4, random_state=17)

tree.fit(X, y)
!pip install pydotplus

import pydotplus 

from sklearn.tree import export_graphviz



def tree_graph_to_png(tree, feature_names, png_file_to_save):

    tree_str = export_graphviz(tree, feature_names=feature_names, 

                                     filled=True, out_file=None)

    graph = pydotplus.graph_from_dot_data(tree_str)

    graph.write_png(png_file_to_save)
tree_graph_to_png(tree=tree, feature_names=iris_data['feature_names'], png_file_to_save='tree.png')

from IPython.display import Image

Image(filename = 'tree.png')
from scipy.special import comb



def probability_of_correct_answer(N, p):

    results = []

    m = math.floor(N / 2) + 1

    i_s = np.arange(m, N + 1)

    combs = [comb(N, i) for i in i_s]

    p_s = np.ones(len(i_s)) * p

    

    return np.sum(combs * np.power(p_s, i_s) * np.power(1 - p_s, N - i_s))
result = probability_of_correct_answer(7, 0.8)

print("Answer: " + ("%0.3f" % result))