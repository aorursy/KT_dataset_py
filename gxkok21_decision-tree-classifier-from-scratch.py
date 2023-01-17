import math



import numpy as np



from sklearn.datasets import load_iris

from sklearn.metrics import accuracy_score
def compute_gini_impurity(labels):

    """

    Compute the gini impurity of a set

    """

    

#     x, y = data, labels

    y = labels

    

#     x_len = len(x)

#     assert x_len == len(y), "Data and labels have different length"

    

    categories = np.unique(y)

    

    p2_sum = 0

    for cat in categories:

#         p_cat = (len(x[(y == cat).reshape(-1,)]) / x_len)

        p_cat = (y == cat).sum() / y.size

        p2_sum += p_cat ** 2

    

    return 1 - p2_sum
def compute_entropy(labels):

    """

    Compute the entropy of a set

    """

    

#     x, y = data, labels

    y = labels

    

#     x_len = len(x)

#     assert x_len == len(y), "Data and labels have different length"

    

    categories = np.unique(y)

    

    entropy = 0

    for cat in categories:

#         p_cat = (len(x[(y == cat).reshape(-1,)]) / x_len)

        p_cat = (y == cat).sum() / y.size

        entropy += -p_cat * math.log2(p_cat)

        

    return entropy
def find_best_split_value(col, label, criterion="gini", categorical_col=False):

    """

    Find the best split value given a feature col and labels

    """

    

    assert criterion == "gini" or criterion == "entropy", "Invalid criterion: {}".format(criterion)

    

    len_col = len(col)

    

#     col_min, col_max = np.min(col), np.max(col)

#     print("col_min:", col_min)

#     print("col_max:", col_max)



    split_vals = np.unique(col)

    split_ginis = np.ones_like(split_vals)

    

    if categorical_col:

        # Theres only one way to split a categorical variable

        print("split_vals:", split_vals)

        

        split_selections = [

            col == split_vals[i] for i in range(len(split_vals))

        ]

        print(split_selections)

        

        num_splits = len(split_selections)

        print("num_splits:", num_splits)

        

        splits = []

        for j in range(num_splits):

            split = (

                col[split_selections[j]], 

                label[split_selections[j]]

            )

            splits.append(split)

        

        gini = 0

        for j in range(num_splits):

            if criterion == "gini":

#                 gini += (len(splits[j][0]) / len_col) * compute_gini_impurity(splits[j][0], splits[j][1])

                print("here1")

                gini += (len(splits[j][0]) / len_col) * compute_gini_impurity(splits[j][1])

            else:

#                 gini += (len(splits[j][0]) / len_col) * compute_entropy(splits[j][0], splits[j][1])

                print("here2")

                gini += (len(splits[j][0]) / len_col) * compute_entropy(splits[j][1])

                

        return None, gini

    else:

        # For continuous variable, where should we split on?

        

        # Iterate through the unique values of the col to find the best split val

        for i in range(len(split_vals)):

            split_selections = [

                col > split_vals[i],

                col <= split_vals[i]

            ]



            num_splits = len(split_selections)



            splits = []

            for j in range(num_splits):

                split = (

                    col[split_selections[j]], 

                    label[split_selections[j]]

                )

                splits.append(split)



            gini = 0

            for j in range(num_splits):

                if criterion == "gini":

#                     gini += (len(splits[j][0]) / len_col) * compute_gini_impurity(splits[j][0], splits[j][1])

                    gini += (len(splits[j][0]) / len_col) * compute_gini_impurity(splits[j][1])

                else:

#                     gini += (len(splits[j][0]) / len_col) * compute_entropy(splits[j][0], splits[j][1])

                    gini += (len(splits[j][0]) / len_col) * compute_entropy(splits[j][1])



            split_ginis[i] = gini



        return split_vals[np.argmin(split_ginis)], split_ginis[np.argmin(split_ginis)]
def find_best_split(data, label, criterion="gini", categorical_cols=[]):

    """

    Find the feature that gives the best split

    """

    

    if len(data.shape) == 1:

        data = data.reshape(-1, 1)

    

    split_vals = []

    split_ginis = []

    

    for i in range(data.shape[1]):

        categorical_col = i in categorical_cols

        split_val, split_gini = find_best_split_value(data[:, i], label, 

                                                      criterion=criterion, 

                                                      categorical_col=categorical_col)



        split_vals.append(split_val)

        split_ginis.append(split_gini)

        

    return (

        np.argmin(split_ginis), # The index of the feature column to split on

        split_vals[np.argmin(split_ginis)], # The split value

        split_ginis[np.argmin(split_ginis)], # The gini impurity of the split

    )
class DecisionTreeNode(object):

    def __init__(self, parent=None):

        self.parent = parent

        if self.parent:

            self.parent.add_child(self)

            

        self.children = []

        self.is_leaf = True

            

    def add_child(self, child):

        assert isinstance(child, DecisionTreeNode), "Provided child node is not an instance of DecisionTreeNode"

    

        self.children.append(child)



    def get_children(self):

        return self.children

    

    def get_parent(self):

        return self.parent

        

    def set_split_node(self, feature_index, split_val, gini, is_categorical_feature=False):

        """

        Set this node as a splitting node

        """

        self.feature_index = feature_index

        self.split_val = split_val

        self.gini = gini

        self.is_categorical_feature = is_categorical_feature

        self.is_leaf = False

    

    def get_splits(self, X, y):

        assert not self.is_leaf, "Can't split on leaf node"

        

        splits = []

        

        if len(X.shape) == 1:

            X = X.reshape(-1, 1)

        

        if self.is_categorical_feature:

            splits.append(

                (X[X[:,self.feature_index] == self.split_val], y[X[:,self.feature_index] == self.split_val])

            )

            splits.append(

                (X[X[:,self.feature_index] != self.split_val], y[X[:,self.feature_index] != self.split_val])

            )

        else:

            splits.append(

                (X[X[:,self.feature_index] > self.split_val], y[X[:,self.feature_index] > self.split_val])

            )

            splits.append(

                (X[X[:,self.feature_index] <= self.split_val], y[X[:,self.feature_index] <= self.split_val])

            )

            

        return splits

    

    def fit(self, X, y):

        self.categories = np.unique(y)

        self.cat_probs = []

        

        for cat in self.categories:

            cat_prob = len(X[(y == cat).reshape(-1,)]) / len(X)

            self.cat_probs.append(cat_prob)

            

        self.fitted = True

    

    def predict(self, X):

        assert self.fitted, "Not fitted"

        

        if len(X.shape) == 1:

            X = X.reshape(-1, 1)

        

        if self.is_leaf:

#             print("")

#             print("self.parent.split_value:", self.parent.split_val)

#             print("self.parent.feature_index:", self.parent.feature_index)

            split_group = 0 if self.parent.children[0] == self else 1

#             print("split_group:", split_group)

#             print("X:", X)

            if len(X):

                print("self.categories:", self.categories)

                print("self.cat_probs:", self.cat_probs)

#             print("self.X", self.X)

#             print("self.y", self.y)

            

#             assert np.all(self.X == X)



            if split_group == 0:

                assert np.all(X[:,self.parent.feature_index] > self.parent.split_val), "Error"

            else:

                assert np.all(X[:,self.parent.feature_index] <= self.parent.split_val), "Error"

            

            res = np.array([ self.categories[np.argmax(self.cat_probs)] for i in range(len(X)) ])

#             print("res:", res)

            return res

        else:

            pred_1 = self.children[0].predict(X[np.where(X[:,self.feature_index] > self.split_val)])

#             print("pred_1.shape:", pred_1.shape)

            pred_2 = self.children[1].predict(X[np.where(X[:,self.feature_index] <= self.split_val)])

#             print("pred_2.shape:", pred_2.shape)

            

            pred = np.concatenate((pred_1, pred_2))

#             print("pred:", pred)

#             print("pred.shape:", pred.shape)

            

            index = np.concatenate((

                np.where(X[:,self.feature_index] > self.split_val),

                np.where(X[:,self.feature_index] <= self.split_val)

            ), axis=1).reshape((-1,))

#             print("index:", index)

#             print("index.shape:", index.shape)

            

            argsort_index = np.argsort(index)

#             print("argsort_index:", argsort_index)

            

            return pred[argsort_index]

#             return pred
def build_tree(X, y, 

               parent=None,

               depth=0, max_depth=2, 

               min_split_size=5,

               criterion="gini",

               categorical_cols=[]):

    node = DecisionTreeNode(parent=parent)

    node.fit(X, y)

    size = len(X)

    

    print(" ")

    print("depth: {}, max_depth: {}".format(depth, max_depth))

    print("size: {}, min_split_size: {}".format(size, min_split_size))

    

    if depth >= max_depth:

        print("max depth reached, not splitting")

    if size < min_split_size:

        print("group size smaller than min split size, not splitting")

    

    if depth < max_depth and size >= min_split_size:

        print("split...")

        feature_index, split_val, gini = find_best_split(X, y, criterion=criterion, categorical_cols=categorical_cols)

        print("feature_index:", feature_index)

        print("split_val:", split_val)

        print("gini:", gini)

        node.set_split_node(feature_index, split_val, gini)

        

        splits = node.get_splits(X, y)

        print("num_splits:", len(splits))

        for splitted_X, splitted_y in splits:

                build_tree(splitted_X, splitted_y,

                           parent=node,

                           depth=depth+1, max_depth=max_depth,

                           min_split_size=min_split_size,

                           criterion=criterion)

        

    return node
class DecisionTreeClassifier(object):

    def __init__(self, max_depth=2, min_split_size=5, criterion="gini"):

        self.max_depth = max_depth

        self.min_split_size = min_split_size

        self.tree = None

        self.criterion = criterion

        

    def fit(self, X, y):

        self.tree = build_tree(X, y, 

                               max_depth=self.max_depth, 

                               min_split_size=self.min_split_size,

                               criterion=self.criterion)

        return self

        

    def predict(self, X):

        assert self.tree, "Not fit"

        return self.tree.predict(X)
data = load_iris()
find_best_split_value(data.data[:,1], data.target)
data2 = np.zeros((10,))
data2[[2, 3, 4, 5]] = 1
data2
targets = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
targets
data2[[6, 8]] = 2
data2
find_best_split_value(data2, targets, categorical_col=True)
find_best_split(data.data, data.target)
model = DecisionTreeClassifier(criterion="entropy")
model = model.fit(data.data, data.target)
accuracy_score(

    y_true=data.target,

    y_pred=model.predict(data.data)

)