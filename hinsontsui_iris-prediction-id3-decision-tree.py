import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris #using the iris dataset from sklearn
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load data
iris = load_iris()

df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])
print(f"Totally, there are {len(df)} records")
df
def compute_entropy(y):
    """
    :param y: The data samples of a discrete distribution
    """
    if len(y) < 2: #  a trivial case
        return 0
    freq = np.array( y.value_counts(normalize=True) )
    return -(freq * np.log2(freq + 1e-6)).sum() # the small eps for 
    # safe numerical computation 
    
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
        else: 
            # this node is an internal one, further queries about an attribute 
            # of the data is needed.
            attr_val = sample[self.split_feat_name]
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
                print(f"Split by {self.split_feat_name}, IG: {info_gain_max:.2f}")
                self.children = {}
                for v in X[self.split_feat_name].unique():
                    index = X[self.split_feat_name] == v
                    self.children[v] = TreeNode(
                        node_name=self.name + ":" + self.split_feat_name + "==" + str(v),
                        min_sample_num=self.min_sample_num,
                        default_decision=self.default_decision)
                    self.children[v].fit(X[index], y[index])
X = df.drop( "target", axis = 1)
y = df["target"]

t = TreeNode(min_sample_num=50)
t.fit(X, y)
attributes = ["sepal length (cm)",
              "sepal width (cm)",
              "petal length (cm)",
              "petal width (cm)"]
new_attributes = []
for a in attributes:
    new_a = "Quant4." + a
    df[new_a] = pd.qcut(df[a], q=4, labels=["q1", "q2", "q3", "q4"])
    new_attributes.append(new_a)
X = df[new_attributes]
y = df["target"]

t = TreeNode(min_sample_num=50)
t.fit(X, y)
corr = 0
err_fp = 0
err_fn = 0
for (i, ct), tgt in zip(X.iterrows(), y):
    a = t.predict(ct)
    if a and not tgt:
        err_fp += 1
    elif not a and tgt:
        err_fn += 1
    else:
        corr += 1
        
corr, err_fp, err_fn
# Parameters
n_classes = 3
plot_colors = "ryb"
plot_step = 0.02

plt.figure(figsize = (20,10))
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3],
                                [1, 2], [1, 3], [2, 3]]):
    # We only take the two corresponding features
    X = iris.data[:, pair]
    y = iris.target

    # Train
    clf = DecisionTreeClassifier().fit(X, y)

    # Plot the decision boundary
    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=3.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.suptitle("Decision surface of a decision tree using paired features", size = 30)
plt.legend(loc='lower right', borderpad=0.1, handletextpad=0.1, fontsize = 20)
plt.axis("tight")
show_parms = DecisionTreeClassifier().fit(X, y)
plt.figure(figsize = (20,10))
plot_tree(show_parms, filled=True)
plt.show()
show_parms_max_depth_3 = DecisionTreeClassifier(max_depth=3).fit(X, y)
plt.figure(figsize = (10,5))
plot_tree(show_parms_max_depth_3, filled=True)
plt.show()
show_parms_max_depth_3 = DecisionTreeClassifier(min_impurity_decrease=0.01).fit(X, y)
plt.figure(figsize = (10,5))
plot_tree(show_parms_max_depth_3, filled=True)
plt.show()