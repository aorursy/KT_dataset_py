import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")





print("Columns: {0}".format(df.columns.tolist()))

print("Row Count: {0}, Column Count: {1}".format(len(df), len(df.columns)))

print(df.head())

print("Number of NaN entries per column")

print(df.isnull().sum().div(len(df)))
# While no significant pre-processing will be performed, the two categorical variables will be convereted from int to string. This will 

# make the decision tree simpler to develop.

df['cp'] = df['cp'].astype(str)

df['thal'] = df['thal'].astype(str)



print(df["target"].value_counts())
import seaborn as sn

import matplotlib.pyplot as plt

import scipy.stats as stats



sn.heatmap(pd.crosstab(df["sex"], df["target"]), annot=True, cmap="YlGnBu")

plt.plot()

coef, p_value = stats.pearsonr(df["sex"], df["target"])

print("Matthew's Correlation Coefficient: {0}, p-value: {1}".format(coef, p_value))
from collections import deque



class BDTNode:

    def __init__(self, df, target_col, level):

        """An individual node in a binary decision tree. It's either a leaf node or has two children, where true_child is the child of the node matching the BDTNode's condition, 

        and false_child is the child not matching the given condition.

         

        Keyword Arguments:

        df - The filtered DataFrame of this specific node, where the DataFrame will be a subset of the parent node's DataFrame after conditional filtering.

        target_col - The target variable to be used for training. As this tree is only intended for use with binary classification tasks, there can only be one target variable.

        level - The level of this node in the BinaryDecisionTree, with 0 denoting the root node.

        """

        self.df = df

        self.target = target_col

        self.level = level

        self.condition = None

        self.true_child = None

        self.false_child = None

        self.ig = 0

        

    def entropy(self):

        """Returns the entropy of this specific BDTNode.

        """

        return 0 if len(self.df[self.target].value_counts()) != 2 else (-self.df[self.target].value_counts()[0]/len(self.df) * np.log2(self.df[self.target].value_counts()[0]/len(self.df))) - (self.df[self.target].value_counts()[1]/len(self.df) * np.log2(self.df[self.target].value_counts()[1]/len(self.df)))

    

    def get_information_gain(self):

        """Returns the information gained by this specific node (not including grandchildren), or 0 if this node is a leaf node.

        """

        return 0 if not self.true_child else self.entropy() - ((self.true_child.entropy() * (len(self.true_child.df)/len(self.df))) + (self.false_child.entropy() * (len(self.false_child.df)/len(self.df))))

    

    def eval_conditional(self, lval, rval):

        if not self.condition:

            return False

        elif "==" in self.condition:

            return lval == rval

        else:

            return lval >= rval

        



class BinaryDecisionTree:

    def __init__(self, df, height, target_col):

        """A binary decision tree. The qualification of binary denotes that the decision tree is for binary classification tasks, and not categorical classification tasks, due to 

        how entropy is calculated in the BDTNode.

        

        Keyword Argumments:

        df - The DataFrame to use for training the BinaryDecisionTree. Note that the BinaryDecisionTree will not split the data up into testing and training sets - that must be done beforehand.

        height - The maximum height of the BinaryDecisionTree. This helps prevent overfitting in lieu of pruning.

        target_col - The target variable to be used for training. As this tree is only intended for use with binary classification tasks, there can only be one target variable.

        """

        self.height = height

        self.target = target_col

        self.root = BDTNode(df, self.target, 0)

        

    def train(self):

        """Trains the BinaryDecisionTree on the DataFrame specified by self.df. Builds branches using the self.branch(node) method.

        """

        q = deque([self.root])

        while q:

            curr_node = q.pop()

            if self._branch(curr_node):

                q.extendleft([curr_node.true_child, curr_node.false_child])

             

    def _branch(self, node):

        """A utility function which comprises the core logic of the BinaryDecisionTree. Branches a single BDTNode into two child nodes by greedily partitioning the DataFrame into

        two subsets based on the condtional split which results in the most information gain.

        

        Returns:

        True - if self._branch(node) paritioned node's data into two subsets

        False if self._branch(node) resulted in node becoming a leaf node, either due to the maximum height of the tree being reached or a failure to find a partition leading to

        information gain.

        """

        if node.level == self.height:

            return False

        df = node.df

        input_cols = df.columns.tolist()

        input_cols.remove(self.target)

        for col in input_cols:

            for unique_value in df[col].unique():

                subsets = [("{0} == {1}".format(col, unique_value), df.loc[df[col] == unique_value], df.loc[df[col] != unique_value]), 

                           ("{0} >= {1}".format(col, unique_value), df.loc[df[col] >= unique_value], df.loc[df[col] < unique_value])] if node.df[col].dtype in ["int64", 

                           "float64"] else [("{0} == {1}".format(col, unique_value), df.loc[df[col] == unique_value], df.loc[df[col] != unique_value])]

                for condition, c1, c2 in subsets:

                    old_true_child, old_false_child, old_condition = node.true_child, node.false_child, node.condition

                    node.condition = condition

                    node.true_child = BDTNode(c1, node.target, node.level + 1)

                    node.false_child = BDTNode(c2, node.target, node.level + 1)

                    ig = node.get_information_gain()

                    if ig > node.ig:

                        node.ig = ig

                    else:

                        node.condition = old_condition

                        node.true_child, node.false_child = old_true_child, old_false_child

        return node.ig != 0

    

    def predict(self, data):

        """Predicts the outcome for a specific data point.

        """

        curr_node = self.root

        while curr_node.true_child:

            col, eq, val = curr_node.condition.split()

            if curr_node.df[col].dtype in ["int64", "float64"]:

                curr_node = curr_node.true_child if curr_node.eval_conditional(data[col], float(val)) else curr_node.false_child

            else:

                curr_node = curr_node.true_child if curr_node.eval_conditional(data[col], val) else curr_node.false_child

        return ("The outcome for ({0}) is: {1}".format(str(data), curr_node.df[self.target].value_counts().idxmax()), curr_node.df[self.target].value_counts().idxmax())

    

    def predict_df(self, test_df, labels):

        """Makes a prediction for an entire testing set DataFrame and returns a confusion matrix of the predictions.

        

        Keyword Arguments:

        test_df - The testing set DataFrame

        labels - The ordering of labels for the confusion matrix to be returned

        """

        df = test_df.copy()

        df["prediction"] = df.apply(lambda row: self.predict(row.to_dict())[1], axis=1)

        return confusion_matrix(df[self.target], df["prediction"], labels=labels)

    

    

class Metrics:

    def __init__(self, cm):

        """A convenience class which helps display model performance metrics given a confusion matrix. 

        

        Keyword Arguments:

        cm - A 2D Array constituting a Confusion Matrix

        """

        self.cm = cm

        

    def get_accuracy(self):

        return sum([self.cm[i][i] for i in range(len(self.cm))]) / np.sum(self.cm)



    def get_recalls(self):

        return [self.cm[i][i]/np.sum(self.cm, axis=1)[i] for i in range(len(self.cm))]



    def get_precisions(self):

        return [self.cm[i][i]/np.sum(self.cm, axis=0)[i] for i in range(len(self.cm))]



    def get_weighted_f1(self):

        class_freq = np.sum(self.cm, axis=0)

        recalls, precisions = self.get_recalls(), self.get_precisions()

        return sum([(2 * frequency * precision * recall) / (precision + recall) for frequency, precision, recall in zip(class_freq, precisions, recalls)]) / sum(class_freq)

    

    def print(self):

        print("Confusion Matrix:\n{0}".format(self.cm))

        print("Accuracy: {0}".format(self.get_accuracy()))

        print("Recalls: {0}".format(self.get_recalls()))

        print("Precisions: {0}".format(self.get_precisions()))

        print("Weighted F1: {0}".format(self.get_weighted_f1()))

        
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



df_train, df_test = train_test_split(df, test_size=0.25, random_state=100)

        

dt = BinaryDecisionTree(df_train, 5, "target")

dt.train()



dt_metrics = Metrics(dt.predict_df(df_test, labels=[0, 1]))

dt_metrics.print()
from sklearn.model_selection import KFold



def k_fold_eval(df, height, splits, random_state=100, verbose=False):

    """A convenience function which obtains the K-fold Cross-Validation metrics for a BinaryDecisionTree model.

    

    Keyword Arguments:

    df - The total DataFrame, which will be split into train/test subsets.

    height - The maximum height of the BinaryDecisionTree

    splits - The number of folds to use for the K-fold Cross-Validation

    random_state - The specific K-fold random state to use

    verbose - Whether or not to log additional details to console

    """

    kf_cm = [[0, 0], [0, 0]]

    k_fold = KFold(n_splits=splits, random_state=random_state, shuffle=True)

    for train_set, test_set in k_fold.split(df):

        df_train, df_test = df.iloc[train_set], df.iloc[test_set]

        dt = BinaryDecisionTree(df_train, height, "target")

        dt.train()

        if verbose:

            print("Root node condition: {0}".format(dt.root.condition))

            print("Root true_child node condition: {0}".format(dt.root.true_child.condition))

            print("Root false_child node condition: {0}".format(dt.root.false_child.condition))

        dt_cm = dt.predict_df(df_test, labels=[0, 1])

        if verbose:

            Metrics(dt_cm).print()

        kf_cm = np.add(kf_cm, dt_cm)

    print("Combined Metrics:")

    Metrics(kf_cm).print()

k_fold_eval(df, 5, 4, verbose=True)
from scipy import stats



class BinaryRandomForest:

    def __init__(self, df, num_trees, tree_dimensionality, tree_height, target_col):

        """Makes use of the BinaryDecisionTree class to allow for the construction of a Random Forest model.

        

        Keyword Arguments:

        df - The training set DataFrame

        num_trees - The number of decision trees in the Random Forest

        tree_dimensionality - The subset of variables that an individual tree should be trained on. For example, tree_dimensionality == 2 would mean each tree would be trained on a random

        subset of the input variables of size two, chosen at random

        tree_height - The maximum height of any individual tree

        target_col - The target column to model

        """

        self.df = df

        self.n = num_trees

        self.k = tree_dimensionality

        self.h = tree_height

        self.target = target_col

        self.forest = []

    

    def train(self):

        """Trains the model by initializing and training a random forest of size self.n

        """

        for i in range(self.n):

            subset = self.df.sample(frac=1, replace=True, random_state=i)

            df_subset = pd.concat([subset.drop(self.target, axis=1).sample(n=self.k, axis=1, random_state=i),subset[[self.target]]],axis=1)

            dt = BinaryDecisionTree(df_subset, self.h, self.target)

            dt.train()

            self.forest.append(dt)

            

    def predict(self, data):

        """Predicts the outcome for a specific data point.

        """

        predictions = []

        for dt in self.forest:

            predictions.append(dt.predict(data)[1])

        return ("The outcome for ({0}) is: {1}".format(str(data), stats.mode(predictions)[0][0]), stats.mode(predictions)[0][0])

    

    def predict_df(self, test_df, labels):

        """Makes a prediction for an entire testing set DataFrame and returns a confusion matrix of the predictions.

        

        Keyword Arguments:

        test_df - The testing set DataFrame

        labels - The ordering of labels for the confusion matrix to be returned

        """

        df = test_df.copy()

        df["prediction"] = df.apply(lambda row: self.predict(row.to_dict())[1], axis=1)

        return confusion_matrix(df[self.target], df["prediction"], labels=labels)

        

        

def k_fold_eval_rf(df, n_trees, tree_dim, height, splits, random_state=100, verbose=False):

    """A convenience function which obtains the K-fold Cross-Validation metrics for a BinaryRandomForest model.

    

    Keyword Arguments:

    df - The total DataFrame, which will be split into train/test subsets.

    n_trees - The number of trees in the BinaryRandomForest

    tree_dim - The number of variables a single tree is trained on.

    height - The maximum height of a BinaryDecisionTree in the BinaryRandomForest

    splits - The number of folds to use for the K-fold Cross-Validation

    random_state - The specific K-fold random state to use

    verbose - Whether or not to log additional details to console

    """

    kf_cm = [[0, 0], [0, 0]]

    k_fold = KFold(n_splits=splits, random_state=random_state, shuffle=True)

    for train_set, test_set in k_fold.split(df):

        df_train, df_test = df.iloc[train_set], df.iloc[test_set]

        rf = BinaryRandomForest(df_train, n_trees, tree_dim, height, "target")

        rf.train()

        if verbose:

            for dt in rf.forest:

                print("Root node condition: {0}".format(dt.root.condition))

        rf_cm = rf.predict_df(df_test, labels=[0, 1])

        #Metrics(rf_cm).print()

        kf_cm = np.add(kf_cm, rf_cm)

    print("Combined Metrics:")

    Metrics(kf_cm).print()

#k_fold_eval_rf(df, 10, 5, 3, 4, verbose=True)
#k_fold_eval_rf(df, 25, 5, 3, 4)

#k_fold_eval_rf(df, 50, 5, 3, 4)
#k_fold_eval_rf(df, 75, 5, 3, 4)

#k_fold_eval_rf(df, 100, 5, 3, 4)
# Increase the max height of a tree from 3 to 5.

#k_fold_eval_rf(df, 25, 5, 5, 4)



# Decrease the max height of a tree from 3 to 2.

#k_fold_eval_rf(df, 25, 5, 2, 4)



# Increase the dimensions of trees from 5 to 8.

#k_fold_eval_rf(df, 25, 8, 3, 4)



# Decrease the dimensions of trees from 5 to 3.

#k_fold_eval_rf(df, 25, 3, 3, 4)
import matplotlib.pyplot as plt



# This takes several days to complete. The insight gathered from it is included below.

#

#for height in range(1, 4):

#    for dim in range(1, 14):

#        print((height, dim))

#        k_fold_eval_rf(df, 25, dim, height, 4)



f1_scores_25_1 = [.7218, .7652, .8223, .8100, .8159, .8026, .7958, .8026, .7928, .7930, .7833, .7825, .7729]

f1_scores_25_2 = [.7115, .7834, .8188, .8276, .8146, .7959, .8002, .8188, .8242, .8150, .8120, .8184, .8211]

f1_scores_25_3 = [.7276, .7769, .8273, .8130, .8069, .8273, .8307, .8174, .8200, .8273, .8339, .8140, .8177]

f1_scores_10_1 = [.6554, .7958, .7851, .7851, .7950, .7785, .7819, .8151, .8150, .7984, .7984, .8083, .7719]

f1_scores_10_2 = [.6548, .7725, .8023, .7985, .7986, .7852, .7964, .8123, .8088, .8096, .8100, .8119, .7851]

f1_scores_10_3 = [.6600, .7593, .7892, .7924, .7917, .8152, .8185, .8251, .8485, .8352, .8387, .8325, .8261]

f1_scores_50_1 = [.7575, .7955, .8320, .8092, .8259, .8156, .8193, .8329, .8195, .8092, .7889, .7727, .7723]

f1_scores_50_2 = [.7399, .7726, .8215, .8215, .8245, .8153, .7959, .8218, .8127, .8032, .8184, .8150, .8041]

f1_scores_50_3 = [.7498, .7795, .8234, .8064, .8106, .8237, .8428, .8399, .8297, .8365, .8464, .8234, .8271]



fig, ax = plt.subplots(2, figsize=(12, 12))



ax[0].plot([i + 1 for i in range(len(f1_scores_10_3))], f1_scores_10_3, color="red", linestyle='dotted', label="Trees == 10, Height == 3")

ax[0].plot([i + 1 for i in range(len(f1_scores_10_2))], f1_scores_10_2, color="blue", linestyle='dotted', label="Trees == 10, Height == 2")

ax[0].plot([i + 1 for i in range(len(f1_scores_10_1))], f1_scores_10_1, color="black", linestyle='dotted', label="Trees == 10, Height == 1")

ax[0].plot([i + 1 for i in range(len(f1_scores_25_3))], f1_scores_25_3, color="red", linestyle='dashed', label="Trees == 25, Height == 3")

ax[0].plot([i + 1 for i in range(len(f1_scores_25_2))], f1_scores_25_2, color="blue", linestyle='dashed', label="Trees == 25, Height == 2")

ax[0].plot([i + 1 for i in range(len(f1_scores_25_1))], f1_scores_25_1, color="black", linestyle='dashed', label="Trees == 25, Height == 1")

ax[0].plot([i + 1 for i in range(len(f1_scores_50_3))], f1_scores_50_3, color="red", label="Trees == 50, Height == 3")

ax[0].plot([i + 1 for i in range(len(f1_scores_50_2))], f1_scores_50_2, color="blue", label="Trees == 50, Height == 2")

ax[0].plot([i + 1 for i in range(len(f1_scores_50_1))], f1_scores_50_1, color="black", label="Trees == 50, Height == 1")

ax[0].set_title("Random Forest Performance")

ax[0].grid()

ax[0].legend()

ax[0].set_ylabel("Weighted F1 Scores")



ax[1].plot([i + 1 for i in range(13)], [(f1_scores_10_1[i] + f1_scores_25_1[i] + f1_scores_50_1[i])/3 for i in range(13)],

         color="red", label="Average of Height == 1")

ax[1].plot([i + 1 for i in range(13)], [(f1_scores_10_2[i] + f1_scores_25_2[i] + f1_scores_50_2[i])/3 for i in range(13)],

         color="blue", label="Average of Height == 2")

ax[1].plot([i + 1 for i in range(13)], [(f1_scores_10_3[i] + f1_scores_25_3[i] + f1_scores_50_3[i])/3 for i in range(13)],

         color="green", label="Average of Height == 3")

ax[1].plot([i + 1 for i in range(13)], [(f1_scores_10_1[i] + f1_scores_10_2[i] + f1_scores_10_3[i])/3 for i in range(13)],

         color="red", linestyle='dashed', label="Average of Trees == 10")

ax[1].plot([i + 1 for i in range(13)], [(f1_scores_25_1[i] + f1_scores_25_2[i] + f1_scores_25_3[i])/3 for i in range(13)],

         color="blue", linestyle='dashed', label="Average of Trees == 25")

ax[1].plot([i + 1 for i in range(13)], [(f1_scores_50_1[i] + f1_scores_50_2[i] + f1_scores_50_3[i])/3 for i in range(13)],

         color="green", linestyle='dashed', label="Average of Trees == 50")



ax[1].set_xlabel("Number of Dimensions")

ax[1].set_ylabel("Weighted F1 Scores")

ax[1].legend()

ax[1].grid()

plt.show()
# This takes days to run. The output from running it is shown below.

#for height in range(1, 11):

#    k_fold_eval_rf(df, 50, 10, height, 4)



f1s = [.8092, .8032, .8365, .8239, .8365, .8203, .8234, .8234, .8265, .8231]



plt.plot([i+1 for i in range(10)], f1s)

plt.xlabel("Random Forest Tree Heights")

plt.ylabel("Weighted F1 Scores")

plt.title("Performance of Random Forests with 100 10-dim trees")

plt.grid()

plt.show()
'''

for dims in [5, 7, 10]:

    for height in range(1, 11):

        k_fold_eval_rf(df, 100, 5, height, 4)

'''



f1s_10dim = [.8092, .8032, .8365, .8239, .8365, .8203, .8234, .8234, .8265, .8231]

f1s_7dim = [.8261, .8188, .8399, .8295, .8299, .8297, .8329, .8327, .8227, .8227]

f1s_5dim = [.8268, .8119, .8229, .8234, .8200, .8198, .8134, .8166, .8164, .8164]

plt.plot([i+1 for i in range(10)], f1s_10dim, label="10-dim Trees", color="red")

plt.plot([i+1 for i in range(10)], f1s_7dim, label="7-dim Trees", color="blue")

plt.plot([i+1 for i in range(10)], f1s_5dim, label="5-dim Trees", color="green")

plt.xlabel("Random Forest Tree Heights")

plt.ylabel("Weighted F1 Scores")

plt.title("Performance of Random Forests with 100 trees")

plt.grid()

plt.legend()

plt.show()
class MixedBinaryRandomForest(BinaryRandomForest):

    def __init__(self, df, num_trees, tree_dimensionality_range, tree_height_range, target_col, seed=100):

        """Makes use of the BinaryDecisionTree class to allow for the construction of a Random Forest model.

        

        Keyword Arguments:

        df - The training set DataFrame

        num_trees - The number of decision trees in the Random Forest

        tree_dimensionality_range - A range of subsets of variables that an individual tree should be trained on. For example, tree_dimensionality_range == (2, 4) would mean each tree would be trained on a random

        subset of the input variables of size between two and four, chosen at random

        tree_height_range - A range of maximum heights of any individual tree. For example, tree_height_range == (2, 4) would mean each tree would have a height between two and four, randomly chosen.

        target_col - The target column to model

        seed - The random seed to use for deterministic purposes of the notebook

        """

        self.df = df

        self.n = num_trees

        self.k_range = tree_dimensionality_range

        self.h_range = tree_height_range

        self.target = target_col

        self.forest = []

        np.random.seed(seed)

    

    def train(self):

        """Trains the model by initializing and training a random forest of size self.n

        """

        for i in range(self.n):

            subset = self.df.sample(frac=1, replace=True, random_state=i)

            df_subset = pd.concat([subset.drop(self.target, axis=1).sample(n=np.random.randint(self.k_range[0], self.k_range[1] + 1), axis=1, random_state=i),subset[[self.target]]],axis=1)

            dt = BinaryDecisionTree(df_subset, np.random.randint(self.h_range[0], self.h_range[1] + 1), self.target)

            dt.train()

            self.forest.append(dt)

            

def k_fold_eval_mrf(df, n_trees, tree_dim_range, height_range, splits, random_state=100, verbose=False):

    """A convenience function which obtains the K-fold Cross-Validation metrics for a MixedBinaryRandomForest model.

    

    Keyword Arguments:

    df - The total DataFrame, which will be split into train/test subsets

    n_trees - The number of trees in the BinaryRandomForest

    tree_dim_range - The range of the number of variables a single tree is trained on, chosen at random

    height - The range of maximum heights of a BinaryDecisionTree in the MixedBinaryRandomForest

    splits - The number of folds to use for the K-fold Cross-Validation

    random_state - The specific K-fold random state to use

    verbose - Whether or not to log additional details to console

    """

    kf_cm = [[0, 0], [0, 0]]

    k_fold = KFold(n_splits=splits, random_state=random_state, shuffle=True)

    for train_set, test_set in k_fold.split(df):

        df_train, df_test = df.iloc[train_set], df.iloc[test_set]

        rf = MixedBinaryRandomForest(df_train, n_trees, tree_dim_range, height_range, "target")

        rf.train()

        if verbose:

            for dt in rf.forest:

                print("Root node condition: {0}".format(dt.root.condition))

        rf_cm = rf.predict_df(df_test, labels=[0, 1])

        #Metrics(rf_cm).print()

        kf_cm = np.add(kf_cm, rf_cm)

    print("Combined Metrics:")

    Metrics(kf_cm).print()
# Build the model. The results of building the model are provided below.

#

#k_fold_eval_mrf(df, 100, (5, 10), (3, 5), 4)
class BDTShuffledNode(BDTNode):

    def __init__(self, df, cols, target_col, level):

        """An individual node in a binary decision tree. It's either a leaf node or has two children, where true_child is the child of the node matching the BDTShuffledNode's condition, 

        and false_child is the child not matching the given condition.

         

        Keyword Arguments:

        df - The filtered DataFrame of this specific node, where the DataFrame will be a subset of the parent node's DataFrame after conditional filtering.

        cols - The input columns which the specific node uses to make a split decision.

        target_col - The target variable to be used for training. As this tree is only intended for use with binary classification tasks, there can only be one target variable.

        level - The level of this node in the BinaryDecisionTree, with 0 denoting the root node.

        """

        self.df = df

        self.input_cols = cols

        self.target = target_col

        self.level = level

        self.condition = None

        self.true_child = None

        self.false_child = None

        self.ig = 0

        



class ShuffledBinaryDecisionTree(BinaryDecisionTree):

    def __init__(self, df, dim, height, target_col):

        """A binary decision tree. The qualification of binary denotes that the decision tree is for binary classification tasks, and not categorical classification tasks, due to 

        how entropy is calculated in the BDTShuffledNode.

        

        Keyword Argumments:

        df - The DataFrame to use for training the ShuffledBinaryDecisionTree. Note that the ShuffledBinaryDecisionTree will not split the data up into testing and training sets - that must be done beforehand.

        dim - The dimensionality of the ShuffledBinaryDecisonTree. Determines how many values each individual node will use as criteria for splitting.

        height - The maximum height of the ShuffledBinaryDecisionTree. This helps prevent overfitting in lieu of pruning.

        target_col - The target variable to be used for training. As this tree is only intended for use with binary classification tasks, there can only be one target variable.

        """

        self.df = df

        self.height = height

        self.dim = dim

        self.target = target_col

        self.input_cols = list(df.columns)

        self.input_cols.remove(target_col)

        self.root = BDTShuffledNode(df, np.random.choice(self.input_cols, dim), self.target, 0)

             

    def _branch(self, node):

        """A utility function which comprises the core logic of the BinaryDecisionTree. Branches a single BDTNode into two child nodes by greedily partitioning the DataFrame into

        two subsets based on the condtional split which results in the most information gain.

        

        Returns:

        True - if self._branch(node) paritioned node's data into two subsets

        False if self._branch(node) resulted in node becoming a leaf node, either due to the maximum height of the tree being reached or a failure to find a partition leading to

        information gain.

        """

        if node.level == self.height:

            return False

        df = node.df

        input_cols = node.input_cols

        for col in input_cols:

            for unique_value in df[col].unique():

                subsets = [("{0} == {1}".format(col, unique_value), df.loc[df[col] == unique_value], df.loc[df[col] != unique_value]), 

                           ("{0} >= {1}".format(col, unique_value), df.loc[df[col] >= unique_value], df.loc[df[col] < unique_value])] if node.df[col].dtype in ["int64", 

                           "float64"] else [("{0} == {1}".format(col, unique_value), df.loc[df[col] == unique_value], df.loc[df[col] != unique_value])]

                for condition, c1, c2 in subsets:

                    old_true_child, old_false_child, old_condition = node.true_child, node.false_child, node.condition

                    node.condition = condition

                    node.true_child = BDTShuffledNode(c1, np.random.choice(self.input_cols, self.dim), node.target, node.level + 1)

                    node.false_child = BDTShuffledNode(c2,  np.random.choice(self.input_cols, self.dim), node.target, node.level + 1)

                    ig = node.get_information_gain()

                    if ig > node.ig:

                        node.ig = ig

                    else:

                        node.condition = old_condition

                        node.true_child, node.false_child = old_true_child, old_false_child

        return node.ig != 0

    

    

class ShuffledBinaryRandomForest(MixedBinaryRandomForest):

    def __init__(self, df, num_trees, tree_dimensionality_range, tree_height_range, target_col, r_seed=100):

        super().__init__(df, num_trees, tree_dimensionality_range, tree_height_range, target_col, seed=r_seed)

    

    def train(self):

        """Trains the model by initializing and training a random forest of size self.n

        Same behavior as the parent class MixedBinaryRandomForest, except that trees in the forest are ShuffledBinaryDecisionTrees, and not plain BinaryDecisionTrees

        """

        for i in range(self.n):

            subset = self.df.sample(frac=1, replace=True, random_state=i)

            dt = ShuffledBinaryDecisionTree(df, np.random.randint(self.k_range[0], self.k_range[1] + 1), np.random.randint(self.h_range[0], self.h_range[1] + 1), self.target)

            dt.train()

            self.forest.append(dt)
def k_fold_eval_srf(df, n_trees, tree_dim_range, height_range, splits, random_state=100, verbose=False):

    """A convenience function which obtains the K-fold Cross-Validation metrics for a ShuffledBinaryRandomForest model.

    

    Keyword Arguments:

    df - The total DataFrame, which will be split into train/test subsets

    n_trees - The number of trees in the BinaryRandomForest

    tree_dim_range - The range of the number of variables a single tree is trained on, chosen at random

    height - The range of maximum heights of a BinaryDecisionTree in the MixedBinaryRandomForest

    splits - The number of folds to use for the K-fold Cross-Validation

    random_state - The specific K-fold random state to use

    verbose - Whether or not to log additional details to console

    """

    kf_cm = [[0, 0], [0, 0]]

    k_fold = KFold(n_splits=splits, random_state=random_state, shuffle=True)

    for train_set, test_set in k_fold.split(df):

        df_train, df_test = df.iloc[train_set], df.iloc[test_set]

        rf = ShuffledBinaryRandomForest(df_train, n_trees, tree_dim_range, height_range, "target")

        rf.train()

        if verbose:

            for dt in rf.forest:

                print("Root node condition: {0}".format(dt.root.condition))

        rf_cm = rf.predict_df(df_test, labels=[0, 1])

        #Metrics(rf_cm).print()

        kf_cm = np.add(kf_cm, rf_cm)

    print("Combined Metrics:")

    Metrics(kf_cm).print()

    

#k_fold_eval_srf(df, 100, (5, 10), (3, 5), 4)
def k_fold_eval_sbt(df, height, dim, splits, random_state=100, verbose=False):

    """A convenience function which obtains the K-fold Cross-Validation metrics for a ShuffledBinaryDecisionTree model.

    

    Keyword Arguments:

    df - The total DataFrame, which will be split into train/test subsets.

    height - The maximum height of the BinaryDecisionTree

    dim - The dimensionality of the ShuffledBinaryDecisionTree

    splits - The number of folds to use for the K-fold Cross-Validation

    random_state - The specific K-fold random state to use

    verbose - Whether or not to log additional details to console

    """

    kf_cm = [[0, 0], [0, 0]]

    k_fold = KFold(n_splits=splits, random_state=random_state, shuffle=True)

    for train_set, test_set in k_fold.split(df):

        df_train, df_test = df.iloc[train_set], df.iloc[test_set]

        dt = ShuffledBinaryDecisionTree(df_train, dim, height, "target")

        dt.train()

        if verbose:

            print("Root node condition: {0}".format(dt.root.condition))

            print("Root true_child node condition: {0}".format(dt.root.true_child.condition))

            print("Root false_child node condition: {0}".format(dt.root.false_child.condition))

        dt_cm = dt.predict_df(df_test, labels=[0, 1])

        if verbose:

            Metrics(dt_cm).print()

        kf_cm = np.add(kf_cm, dt_cm)

    print("Combined Metrics:")

    Metrics(kf_cm).print()



''' 

df_subset = pd.concat([df.drop("target", axis=1).sample(3, axis=1, random_state=100), df[["target"]]],axis=1)



for height in range(1, 11):

    print("BinaryDecisionTree:")

    k_fold_eval(df_subset, height, 4)

    print("ShuffledBinaryDecisionTree:")

    k_fold_eval_sbt(df, height, 3, 4)

'''



bdt_f1s = [.6928, .7176, .6916, .6875, .7000, .7000, .7000, .7000, .7000, .7000]

sbdt_f1s = [.7453, .7468, .7326, .7559, .7773, .7508, .7556, .7588, .7302, .7526]

plt.plot([i+1 for i in range(len(bdt_f1s))], bdt_f1s, label="BinaryDecisionTree", color="red")

plt.plot([i+1 for i in range(len(sbdt_f1s))], sbdt_f1s, label="ShuffledBinaryDecisionTree", color="blue")

plt.legend()

plt.grid()

plt.xlabel("Tree Height")

plt.ylabel("Weighted F1 Score")

plt.title("Three-Dimensional Decision Tree Performance")

plt.show()
#k_fold_eval_srf(df, 200, (5, 10), (3, 5), 4)

#k_fold_eval_srf(df, 200, (3, 5), (3, 5), 4)

#k_fold_eval_srf(df, 200, (3, 5), (5, 7), 4)