import numpy as np
def entropy(y):
    cnt = np.bincount(y)
    # creates an array cnt, cnt[i] = count of entries of number i.
    
    probabilities = cnt / len(y)
    # array of probabilities for each state
        
    res = -np.sum([p * np.log(p) for p in probabilities if p > 0])
    # calculating entropy
    return res


class ClassifyNode:
    
    # ClassidyNode class contains child nodes, best threshold and best feature for splitting
    
    def __init__(self, feature=None, threshold=None, l_node=None, r_node=None, mark=None):
        
        self.feature = feature     
        # idx of feature for splitting
        
        self.threshold = threshold
        # threshold value for splitting
        
        self.l_node = l_node       
        # left ClassifyNode object 
        
        self.r_node = r_node       
        # right ClassifyNode object 
        
        self.mark = mark           
        # if it is a leaf node, it gets a class mark 
        
    def is_leaf_node(self):
        return self.mark is not None
        
class MyDecisionTreeClassifier:
    
    def __init__(self, min_samples_split=2, max_depth="inf", n_feats=None):
        self.min_samples_split = min_samples_split 
        # min number of samples in node to allow splitting
        
        self.max_depth = max_depth
        # max depth of tree
        
        self.n_feats = n_feats
        # if n_feats < X.shape[1] then choose features randomly
        
        self.root = None
        # root ClassifyNode object
        
    def fit(self, X, y):
        
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        # avoid situation when n_feats > real number of features 
        
        self.root = self._build_node(X, y)
        # The first node for splitting
        
    
        
        
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
        # see _traverse_tree function
    
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
        # if current node is terminal then return class with biggest occurrence level
        
            return node.mark
        
        
        if x[node.feature] <= node.threshold:  
        # chek feature in this node and compare node threshold with feature value in x 
            
            return self._traverse_tree(x, node.l_node)
            # send x to child node until it gets the leaf node
            
        return self._traverse_tree(x, node.r_node)
    
        
    
    
    
    def _build_node(self, X, y, depth=0):
        
        n_samples, n_features = X.shape 
        # in arrays and Dataframes shapes are in format (n_samples, n_features)
        
        n_labels = len(np.unique(y))
        # array of unique classes
        
        # check stopping criteria (max_depth & min_samples_split)
        
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_mark = self._most_encountered_mark(y)
            return ClassifyNode(mark=leaf_mark)
        
        
        feature_idxs = np.random.choice(n_features, self.n_feats, replace=False)
        # if we decided to choose a subset of features
        
        best_feature, best_thresh = self._best_split(X, y, feature_idxs)
        # see _best_split()
        
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        # see _split()
        
        left = self._build_node(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._build_node(X[right_idxs, :], y[right_idxs], depth+1)
        # after we find best split, we give left data and right data to child nodes
        
        return ClassifyNode(best_feature, best_thresh, left, right)
        
        
    def _best_split(self, X, y, feature_idxs):
        
        best_information_gain = -1
        
        split_idx, split_thresh = None, None
        
        for idx in feature_idxs:
            # for each feature
            
            X_column=X[:, idx]
            # take all values in this features

            thresholds = np.unique(X_column)
            # drop similar values
            
            for threshold in thresholds:
                # for each unique value
                
                gain = self._information_gain(y, X_column, threshold)
                # see _information_gain()
                
                if gain > best_information_gain:
                    best_information_gain = gain
                    split_idx = idx
                    split_thresh = threshold
                # save information gain in case it better than previous
                    
        return split_idx, split_thresh
    
    
    def _information_gain(self, y, X_column, split_thresh):
        
        parent_entropy = entropy(y)
        # see entropy()
        
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        # see _split()
        
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        # check if we splitted nothing
        
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        # number of samples in left and right nodes
        
        entropy_l, entropy_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        
        child_entropy = (n_l/n)*entropy_l + (n_r/n)*entropy_r
        # weighted sum of left and right child
        
        info_gain = parent_entropy - child_entropy
        # finally calculated information gain 
        
        return info_gain
    
    
    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        
        return left_idxs, right_idxs
        
    
    def _most_encountered_mark(self, y):
        most_encountered = np.argmax(np.bincount(y))
        return most_encountered

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred)/len(y_true)
    return accuracy


data = datasets.load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("Train shape:", X_train.shape)

sk_clf = DecisionTreeClassifier(max_depth=100, criterion='entropy', random_state=0)
my_clf = MyDecisionTreeClassifier(max_depth=100)

my_clf.fit(X_train, y_train)
sk_clf.fit(X_train, y_train)

my_y_pred = my_clf.predict(X_train)
sk_y_pred = sk_clf.predict(X_train)

my_acc = accuracy(y_train, my_y_pred)
sk_acc = accuracy(y_train, sk_y_pred)

print("Working test on train data", my_acc)
print("Working test on train data sklearn", sk_acc)

my_y_pred = my_clf.predict(X_test)
sk_y_pred = sk_clf.predict(X_test)

my_acc = accuracy(y_test, my_y_pred)
sk_acc = accuracy(y_test, sk_y_pred)

print("Working test on test data", my_acc)
print("Working test on test data sklearn", sk_acc)
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred)/len(y_true)
    return accuracy


data = datasets.load_breast_cancer()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("Train shape:", X_train.shape)

sk_clf = DecisionTreeClassifier(max_depth=100, criterion='entropy', random_state=0)
my_clf = MyDecisionTreeClassifier(max_depth=100)

%time my_clf.fit(X_train, y_train)
%time sk_clf.fit(X_train, y_train)

