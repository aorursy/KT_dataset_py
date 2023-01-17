import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets
from IPython.display import display, HTML
from sklearn.model_selection import learning_curve, StratifiedKFold, train_test_split
import seaborn as sns
import itertools
import graphviz 
%matplotlib inline
# Calculate Entropy

def compute_entropy(y):
    """
    :param y: The data samples of a discrete distribution
    """
    if len(y) < 2: #  a trivial case
        return 0
    freq = np.array( y.value_counts(normalize=True) )
    return -(freq * np.log2(freq + 1e-6)).sum() # the small eps for 
    # safe numerical computation 

# Calculate Info Gain

def compute_info_gain(samples, attr, target):
    values = samples[attr].value_counts(normalize=True)
    split_ent = 0
    for v, fr in values.iteritems():
        index = samples[attr]==v
        sub_ent = compute_entropy(target[index])
        split_ent += fr * sub_ent
    ent = compute_entropy(target)
    return ent - split_ent

class TreeNode:
    """
    A recursively defined data structure to store a tree.
    Each node can contain other nodes as its children
    """
    def __init__(self, node_name="", min_sample_num=10, default_decision=None):
        self.children = {} # Sub nodes --
        # recursive, those elements of the same type (TreeNode)
        self.decision = None # Undecided
        self.split_feat_name = None # Splitting feature
        self.name = node_name
        self.default_decision = default_decision
        self.min_sample_num = min_sample_num

    def pretty_print(self, prefix=''):
        if self.split_feat_name is not None:
            for k, v in self.children.items():
                v.pretty_print(f"{prefix}:When {self.split_feat_name} is {k}")
                #v.pretty_print(f"{prefix}:{k}:")
        else:
            print(f"{prefix}:{self.decision}")

    def predict(self, sample):
        if self.decision is not None:
            # uncomment to get log information of code execution
            print("Decision:", self.decision)
            return self.decision
        elif self.split_feat_name is None:
            # uncomment to get log information of code execution
            print("Decision:", self.decision)
            return self.decision
        else: 
            # this node is an internal one, further queries about an attribute 
            # of the data is needed.
            # print(sample)
            print("MY FEATURE" + self.split_feat_name)
            attr_val = sample[self.split_feat_name]
            # print(self.children)
            child = self.children[attr_val]
            # uncomment to get log information of code execution
            print("Testing ", self.split_feat_name, "->", attr_val)
            return child.predict(sample)

    def fit(self, X, y):
        """
        The function accepts a training dataset, from which it builds the tree 
        structure to make decisions or to make children nodes (tree branches) 
        to do further inquiries
        :param X: [n * p] n observed data samples of p attributes
        :param y: [n] target values
        """
        if self.default_decision is None:
            self.default_decision = y.mode()[0]
            
            
        print(self.name, "received", len(X), "samples")
        if len(X) < self.min_sample_num:
            # If the data is empty when this node is arrived, 
            # we just make an arbitrary decision
            if len(X) == 0:
                self.decision = self.default_decision
                print("DECISION", self.decision)
            else:
                self.decision = y.mode()[0]
                print("DECISION", self.decision)
            return
        else: 
            unique_values = y.unique()
            if len(unique_values) == 1:
                self.decision = unique_values[0]
                print("DECISION", self.decision)
                return
            else:
                info_gain_max = 0
                for a in X.keys(): # Examine each attribute
                    aig = compute_info_gain(X, a, y)
                    if aig > info_gain_max:
                        info_gain_max = aig
                        self.split_feat_name = a
                if self.split_feat_name == None:
                  return
                print(f"Split by {self.split_feat_name}, IG: {info_gain_max:.2f}")
                self.children = {}
                for v in X[self.split_feat_name].unique():
                    index = X[self.split_feat_name] == v
                    self.children[v] = TreeNode(
                        node_name=self.name + ":" + self.split_feat_name + "==" + str(v),
                        min_sample_num=self.min_sample_num,
                        default_decision=self.default_decision)
                    self.children[v].fit(X[index], y[index])
# Testing out using Data From Kaggle - using the breast cancer wisconsin dataset 
# Reference: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

import pandas as pd
df = pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

#finding out the rows and columns
print(df)

#Setting up train test split for the first time
X = df.drop(columns=["id","diagnosis"])
y = df["diagnosis"]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
 
#Healthy cells typically have nuclei with a standard size and shape while cancer cells often have nuclei that are large and mishapen. 
#As such, the size and shape of the nucleus should be a good predictor for whether or not a sample is cancerous.

#Doing feature selection - viewing which features are correlated
print("\nFeature Correlation:\n")
g = sns.heatmap(X_train.corr(),cmap="BrBG",annot=False)

#Removing all features which are correlated
X2 = X.drop(columns=['perimeter_mean', 'radius_mean',
       'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean',  'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst', 'Unnamed: 32'])
y2 = y

# Binning the numerical data so that we can use it for the ID3 decision tree algorithm
X2['texture_mean'] = pd.qcut(X2['texture_mean'], q=3, labels=[f'texture_mean_{i}' for i in range(3)])
X2['area_mean'] = pd.qcut(X2['area_mean'], q=3, labels=[f'area_mean_{i}' for i in range(3)])
X2['symmetry_mean'] = pd.qcut(X2['symmetry_mean'], q=3, labels=[f'symmetry_mean_{i}' for i in range(3)])

# Setting up the new train test split
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, y2, test_size=0.2)

print(X_train2)
print(X_test2)

t = TreeNode() 
t.fit(X_train2, Y_train2)
t.pretty_print()
#Setting up the variables for calculation and correct modelling
corr = 0
err_fp = 0
err_fn = 0

# Test tree working
# print(X_test2)
# print(type(X_test2))
# print(Y_test2)
# print(X_test2.iloc[0])

for (i, data), tgt in zip(X_test2.iterrows(), Y_test2):
    # print(i)
    print(tgt)
    print(data)
    a = t.predict(data)
    if a and not tgt:
        err_fp += 1
    elif not a and tgt:
        err_fn += 1
    else:
        corr += 1

precision = corr/(corr+err_fp)
recall = corr/(corr+err_fn)
f1_score = 2*((precision*recall)/(precision+recall))

corr, err_fp, err_fn, precision, recall, f1_score