!pip install pydotplus
import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import pandas as pd



from matplotlib.colors import ListedColormap

from pydotplus import graph_from_dot_data

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz

from IPython.display import Image
def mean_squared_error(y, y_preds):

    return np.square(y - y_preds).mean()



def mean_absolute_error(y, y_preds):

    return np.abs(y - y_preds).mean()



def standard_deviation(n, summed_vals, summed_vals_squared):

    """ Standard deviation using summed vals and sum of squares """

    return np.sqrt((summed_vals_squared / n) - 

                   np.square(summed_vals / n))
def gini_index(p):

    """ Gini impurity for binary classification """

    return p*(1-p) + ((1-p)*(1 - (1-p)))



def entropy(p):

    """ Entropy for binary classification """

    return - (p*np.log2(p) + ((1-p)*np.log2(1-p)))



def classification_error(p):

    """ Classification error for binary classification """

    return 1 - np.max([p, 1 - p])
example_data = np.random.rand(1000,2)*10



class_a = example_data[np.where((example_data[:, 0] > 3) & (example_data[:, 1] < 8))]

class_b = example_data[np.where((example_data[:, 0] <= 3) | (example_data[:, 1] >= 8))]



X = np.concatenate([class_a, class_b])

y = np.concatenate([np.zeros(class_a.shape[0]), np.ones(class_b.shape[0])])



plt.figure(figsize=(5,5))

plt.scatter(class_a[:, 0], class_a[:, 1], label='Class A', 

            marker='o', color='red', edgecolor='black')

plt.scatter(class_b[:, 0], class_b[:, 1], label='Class B', 

            marker='x', color='blue', edgecolor='black')

plt.xlabel("x")

plt.ylabel('y')

plt.title("Example Data", weight='bold')

plt.show()





decision_tree = DecisionTreeClassifier()

logreg_model = LogisticRegression(solver='lbfgs')



decision_tree.fit(X, y)

logreg_model.fit(X, y)
def plot_boundaries(X, y, clf, resolution=0.02, figsize=(5,5), title='Decision Boundary'):

    """ Plot basic decision boundaries for given classifier and 

        X and y data """

    

    plt.figure(figsize=figsize)

    

    markers = ('o', 'x', 's', '^', 'v')

    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    cmap = ListedColormap(colors[:len(np.unique(y))])

    

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),

                           np.arange(x2_min, x2_max, resolution))

    

    Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    Z = Z.reshape(xx1.shape)

    

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)

    plt.xlim(xx1.min(), xx1.max())

    plt.xlim(xx2.min(), xx2.max())

    

    plt.title(title, weight='bold')

    

    for idx, class_example in enumerate(np.unique(y)):

        plt.scatter(x=X[y == class_example, 0], y=X[y == class_example, 1],

                   alpha=0.8, c=colors[idx], marker=markers[idx],

                   label=class_example, edgecolor='black')

        

plot_boundaries(X, y, decision_tree, title="Decision Tree")

plot_boundaries(X, y, logreg_model, title="Logistic Regression")
def show_tree_graph(tree_model, feature_names):

    """ Output a decision tree to notebook """

    draw_data = export_graphviz(tree_model, filled=True, 

                                rounded=True, feature_names=feature_names, 

                                out_file=None, rotate=True)

    graph = graph_from_dot_data(draw_data)



    return Image(graph.create_png())



cols = ['Class A', 'Class B']

show_tree_graph(decision_tree, feature_names=cols)
np.random.seed(12)
fish_df = pd.read_csv('/kaggle/input/fish-market/Fish.csv')

fish_df.head(5)
y = fish_df['Weight'].values

X = fish_df.drop('Weight', axis=1)
X['Species'].value_counts().plot.bar()

plt.show()
encoder = LabelEncoder()

encoder.fit(X['Species'])

encoder.classes_
X['Species'] = encoder.transform(X['Species'])

X['Species'].value_counts()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 

                                                    stratify=X['Species'], random_state=7)



print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
rf_regr = RandomForestRegressor(n_estimators=2, max_depth=3)

rf_regr.fit(X_train, y_train)
def feature_importances(rf_model, dataframe):

    return pd.DataFrame({'columns' : dataframe.columns, 

                         'importance' : rf_model.feature_importances_}

                       ).sort_values('importance', ascending=False)
importances = feature_importances(rf_regr, X_train)



plt.figure(figsize=(7,5))

sns.barplot(x="columns", y="importance", data=importances)

plt.ylabel("Feature Importances", weight='bold')

plt.xlabel("Features", weight='bold')

plt.title("Random Forest Feature Importances", weight='bold')

plt.show()

print(importances)
cmap = sns.cubehelix_palette(dark=.1, light=.7, as_cmap=True)

plt.figure(figsize=(10,5))

sns.scatterplot(x=X_train['Width'], y=y_train, data=X_train, 

                hue='Length3', s=80, palette="Set1", style='Species')

plt.xlabel("Width", weight='bold')

plt.ylabel("Weight", weight='bold')

plt.show()
plt.figure(figsize=(14,5))

plt.subplot(1, 2, 1)

sns.scatterplot(x=X_train['Width'], y=y_train, data=X_train, s=80)

plt.xlabel("Width", weight='bold')

plt.ylabel("Weight", weight='bold')



plt.subplot(1, 2, 2)

sns.scatterplot(x=X_train['Length3'], y=y_train, data=X_train, s=80, style='Species')

plt.xlabel("Length3", weight='bold')

plt.ylabel("Weight", weight='bold')

plt.show()
plt.figure(figsize=(14,5))

plt.subplot(1, 2, 1)

sns.scatterplot(x=X_train['Width']*X_train['Length3'], y=y_train, data=X_train, s=80)

plt.xlabel(f"Width x Length3", weight='bold')

plt.ylabel("Weight", weight='bold')



plt.subplot(1, 2, 2)

sns.scatterplot(x=X_train['Width'], y=np.log1p(y_train), data=X_train, s=80)

plt.xlabel("Width", weight='bold')

plt.ylabel("Log(Weight)", weight='bold')

plt.show()
def plot_tree_regression(X, y, x_axis, model, figsize=(6,4)):

    """ Helper function to plot data and regression line """

    X_ = X[x_axis] # indexes of sorted X vals

    sort_idx = X_.values.flatten().argsort()

    

    plt.figure(figsize=figsize)

    plt.scatter(x=X[x_axis], y=y, color='tab:blue', edgecolor='white', s=70)

    

    plt.step(X_.iloc[sort_idx], model.predict(X.iloc[sort_idx, :]), 

             color='tab:red', lw=2, alpha=0.8)

    return None
plot_tree_regression(X_train, y_train, x_axis='Width', model=rf_regr, figsize=(8,5))
X_subset = X['Width'].values.reshape(-1, 1)



rf_model = RandomForestRegressor(max_depth=2, n_estimators=10).fit(X_subset, y)



plt.figure(figsize=(15, 8))

sort_idx = X_subset.flatten().argsort()



for i in range(0, 6):

    

    decision_tree = rf_model.estimators_[i]

    preds = decision_tree.predict(X_subset[sort_idx])

    r2 = r2_score(y[sort_idx], preds)

    mse = mean_squared_error(y[sort_idx], preds)

    

    ax = plt.subplot(2, 3, i + 1)

    ax.set_title(f"Decision Tree {i + 1} \n$R^{2} = {r2:.2f}$", weight='bold')

    ax.scatter(x=X_subset, y=y, color='tab:blue', edgecolor='white', s=70)

    ax.step(X_subset[sort_idx], preds, color='tab:red', lw=5, alpha=0.7)

    

    if i == 0 or i == 3:

        ax.set_ylabel("Weight", weight='bold')

    

    if i > 2:

        ax.set_xlabel("Width", weight='bold')

    

plt.tight_layout()

plt.show()
preds = rf_model.predict(X_subset[sort_idx])

r2 = r2_score(y[sort_idx], preds)

mse = mean_squared_error(y[sort_idx], preds)



plt.figure(figsize=(8,5))

plt.scatter(x=X_subset, y=y, color='tab:blue', edgecolor='white', s=70)

plt.step(X_subset[sort_idx], preds, color='tab:red', lw=5, alpha=0.7)

plt.title(f"Random Forest (Average of all trees) \n$R^{2} = {r2:.2f}$", weight='bold')

plt.ylabel("Weight", weight='bold')

plt.xlabel("Width", weight='bold')

plt.show()
X_subset = X['Width'].values.reshape(-1, 1)

sort_idx = X_subset.flatten().argsort()



plt.figure(figsize=(11, 6))



for i in range(1, 5):

    

    tree_model = DecisionTreeRegressor(max_depth=i).fit(X_subset, y)

    preds = tree_model.predict(X_subset[sort_idx])

    r2 = r2_score(y[sort_idx], preds)

    

    plt.subplot(2, 2, i)

    plt.scatter(x=X_subset, y=y, color='tab:blue', edgecolor='white', s=70)

    plt.step(X_subset[sort_idx], preds, 

             color='tab:red', lw=4, alpha=0.8)

    plt.title(f"Max Depth = {i} \n$R^{2} = {r2:.2f}$", weight='bold')

    plt.ylabel("Weight", weight='bold')

    

    if i > 2:

        plt.xlabel("Width", weight='bold')

        

plt.tight_layout()

plt.show()
class RandomForest():

    """ Python implementation of a random forest regressor """

    def __init__(self, x, y, num_trees, sample_size, feature_proportion=1.0, 

                 min_leaf=5, bootstrap=False, random_seed=12):

        np.random.seed(random_seed)

        self.x = x

        self.y = y

        self.num_trees = num_trees

        self.sample_size = sample_size

        self.feature_proportion = feature_proportion

        self.min_leaf = min_leaf

        self.bootstrap = bootstrap

        self.trees = [self.create_tree(bootstrap) for i in range(num_trees)]

    

    

    def create_tree(self, bootstrap=False):

        """ Form individual decision tree """

        

        # obtain a random sample of indices and identify oob samples

        idxs = np.random.permutation(self.y.shape[0])[:self.sample_size]

        

        oob_idxs = None

        

        # if bootstrap chosen get bootstrap sample and oob indexes

        if bootstrap:

            idxs, oob_idxs = self.bootstrap_samples(idxs)

        

        return DecisionTree(self.x.iloc[idxs], self.y[idxs], 

                            feat_proportion=self.feature_proportion, 

                            idxs=np.array(range(self.sample_size)), 

                            oob_idxs=oob_idxs, 

                            min_leaf=self.min_leaf)

    

    

    def predict(self, x):

        """ Return the mean of predictions across trees """

        # call predict function from each Tree class

        return np.mean([t.predict(x) for t in self.trees], axis=0)

    

    

    def oob_score(self):

        """ Calculate and return each tree OOB R2 score and the average 

            OOB score across all decision trees """

        

        tree_oob_scores = []

        

        # find oob score for each tree and append to results

        for tree in self.trees:

            

            # find current tree oob predictions and labels

            tree_oob_labels = self.y[tree.oob_idxs]

            tree_oob_preds = tree.predict(self.x.iloc[tree.oob_idxs].values)

            

            # calculate R2 score for predictions on current tree

            tree_oob_r2 = r2_score(tree_oob_labels, tree_oob_preds)

            

            # add R2 score for oob predictions from this tree

            tree_oob_scores.append(tree_oob_r2)

        

        tree_oob_scores = np.array(tree_oob_scores)

        

        # find average oob scores across all trees

        avg_oob_score = np.mean(tree_oob_scores)

    

        return tree_oob_scores, avg_oob_score

    

    

    def bootstrap_samples(self, idxs):

        """ Return bootstrapped sample indices based on y and sample size """

        

        # take sample (with replacement) of idxs and set as bootstrap sample

        sample_idxs = np.random.randint(0, len(idxs), size=self.sample_size)

        bootstrap_idxs = idxs[sample_idxs]

        

        # find out-of-bag (OOB) samples from the passed idxs array

        i = np.arange(self.sample_size)

        oob_i = np.array([ind for ind in i if ind not in sample_idxs])

        oob_idxs = idxs[oob_i]

        

        return bootstrap_idxs, oob_idxs

    

    

    def feature_importances(self):

        """ Find the feature importances by shuffling each feature

            and finding the drop in score relative to baseline. """

        

        # find baseline r2 score - all features will compare against this

        baseline_score = r2_score(self.y, self.predict(self.x.values))

        

        # dictionary to store feature importances

        feat_importances = {}

        columns = self.x.columns

        

        # iterate through each column, shuffle and get new score

        for feat_column in columns:

            

            # shuffle only current column

            temp_df = self.x.copy()

            feat_vals = temp_df[feat_column].values

            np.random.shuffle(feat_vals)

            

            # find new R2 score with shuffled feature

            shuffled_score = r2_score(self.y, self.predict(temp_df.values))

            

            # calculate how much score has changed - this represents importance

            feat_score = (baseline_score - shuffled_score) / baseline_score

            

            # add to importance dict

            feat_importances[feat_column] = feat_score

        

        importance_df = pd.DataFrame.from_dict(feat_importances, 

                                               orient='index',

                                               columns=['Importance'])

        

        return importance_df.sort_values('Importance', ascending=False)
class DecisionTree():

    """ Form a basic decision tree """

    def __init__(self, x, y, idxs=None, oob_idxs=None, 

                 min_leaf=5, feat_proportion=1.0):

        if idxs is None:

            idxs = np.arange(len(y))

        self.x = x

        self.y = y

        self.idxs = idxs

        self.oob_idxs = oob_idxs

        self.min_leaf = min_leaf

        self.feat_proportion = feat_proportion

        self.rows = len(idxs)

        self.cols = self.x.shape[1]

        self.val = np.mean(y[idxs])

        self.score = float('inf')

        self.binary_split()

    

    

    def __repr__(self):

        """ String reputation of our decision tree """

        text = f'n: {self.rows}, val: {self.val}'

        if not self.is_leaf:

            text += f', score: {self.score}, split: {self.split}, var: {self.split_name}'

        return text

    

    

    def binary_split(self):

        """ find best feature and level to split at to produce greatest

            reduction in variance """

        

        # randomly select sub-sample of features

        num_feat = int(np.ceil(self.cols*self.feat_proportion))

        col_idxs = range(self.cols)

        feature_subset = np.random.permutation(col_idxs)[:num_feat]

        

        # iteratively split each col and find best

        for i in feature_subset:

            self.best_binary_split(i)

        # if leaf node stop

        if self.score == float('inf'):

            return

        

        # get split col and idxs for lhs and rhs splits

        x = self.split_col_values

        lhs = np.nonzero(x <= self.split)[0]

        rhs = np.nonzero(x > self.split)[0]

        

        # create new decision trees for each split

        self.left_split = DecisionTree(self.x, self.y, self.idxs[lhs])

        self.right_split = DecisionTree(self.x, self.y, self.idxs[rhs])

            

            

    def best_binary_split(self, feat_idx):

        """ Find best binary split for the given feature """

        x = self.x.values[self.idxs, feat_idx]

        y = self.y[self.idxs]

        

        # sort our data

        sorted_ind = np.argsort(x)

        sorted_x = x[sorted_ind]

        sorted_y = y[sorted_ind]

        

        # get count, sum and square sum of lhs and rhs

        lhs_count = 0

        rhs_count = self.rows

        lhs_sum = 0.0

        rhs_sum = sorted_y.sum()

        lhs_sum2 = 0.0

        rhs_sum2 = np.square(sorted_y).sum()

        

        # iterate through all values of selected feature - eval score

        for i in range(0, self.rows - self.min_leaf):

            x_i = sorted_x[i]

            y_i = sorted_y[i]

            

            # update count and sums

            lhs_count += 1

            rhs_count -= 1   

            lhs_sum += y_i

            rhs_sum -= y_i

            lhs_sum2 += y_i**2

            rhs_sum2 -= y_i**2

            

            # if less than min leaf or dup value - skip

            if i < self.min_leaf - 1 or x_i == sorted_x[i + 1]:

                continue

            

            # find standard deviations of left and right sides

            lhs_std = self.standard_deviation(lhs_count, lhs_sum, lhs_sum2)

            rhs_std = self.standard_deviation(rhs_count, rhs_sum, rhs_sum2)

            

            # find weighted score

            current_score = (lhs_count*lhs_std) + (rhs_count*rhs_std)

            

            # if score lower (better) than previous, update

            if current_score < self.score:

                self.feat_idx = feat_idx

                self.score = current_score

                self.split = x_i

    

    

    def standard_deviation(self, n, summed_vals, summed_vals_squared):

        """ Standard deviation using summed vals, sum of squares, and data size """

        return np.sqrt((summed_vals_squared / n) - np.square(summed_vals / n))

    

    

    def predict(self, x):

        """ Find and return predictions for all the samples in x """

        return np.array([self.predict_sample(x_i) for x_i in x])

    

    

    def predict_sample(self, x_i):

        """ Take a sample x_i and return the predicted value using recursion """

        # if leaf node - return mean value

        if self.is_leaf:

            return self.val

        

        # if value less than tree split value lhs, else rhs

        elif x_i[self.feat_idx] <= self.split:

            tree = self.left_split

        else:

            tree = self.right_split

            

        # recursively continue through the tree with x_i until leaf node

        return tree.predict_sample(x_i)

    

    

    @property

    def split_name(self):

        """ return name of column we are splitting on """

        return self.x.columns[self.feat_idx]

    

    

    @property

    def split_col_values(self):

        """ return values of column we have split on """

        return self.x.values[self.idxs, self.feat_idx]

    

    

    @property

    def is_leaf(self):

        """ If leaf node, score will be infinity """

        return self.score == float('inf')
random_forest = RandomForest(X_train, y_train, 10, X_train.shape[0], feature_proportion=1.0, bootstrap=True)

tree = random_forest.trees[0]

x_data, y_data = tree.x, tree.y

x_data.columns
tree
tree_oob_scores, avg_oob_score = random_forest.oob_score()

print(f"OOB R2 Score: {avg_oob_score}")
tree_oob_scores
np.mean(tree_oob_scores)
feature_importances = random_forest.feature_importances()

feature_importances
feature_importances.plot.bar()

plt.show()
def show_tree_graph(tree_model, feature_names):

    """ Output a decision tree to notebook """

    draw_data = export_graphviz(tree_model, filled=True, 

                                rounded=True, feature_names=feature_names, 

                                out_file=None, rotate=True)

    graph = graph_from_dot_data(draw_data)



    return Image(graph.create_png())
m = RandomForestRegressor(n_estimators=1, max_depth=1, min_samples_leaf=3, bootstrap=False)

m.fit(X_train, y_train)



cols = X_train.columns.values



show_tree_graph(m.estimators_[0], feature_names=cols)
m = RandomForestRegressor(n_estimators=10, max_depth=2, bootstrap=True, oob_score=True)

m.fit(X_train, y_train)



show_tree_graph(m.estimators_[0], feature_names=cols)
m.oob_score_
randf = RandomForest(X_train, y_train, 10, X_train.shape[0], feature_proportion=0.75, bootstrap=True)

tree = randf.trees[0]
tree
for individual_tree in randf.trees:

    print(individual_tree)
# obtain predictions from custom model and scikit learn

preds = randf.predict(X_test.values)

sklearn_preds = m.predict(X_test)



plt.figure(figsize=(14, 4))

plt.subplot(1, 2, 1)

plt.title("Custom Random Forest", weight='bold')

plt.xlabel("Predictions", weight='bold')

plt.ylabel("Weight", weight='bold')

plt.scatter(preds, y_test, alpha=0.7, color='tab:blue')



plt.subplot(1, 2, 2)

sklearn_preds = m.predict(X_test)

plt.scatter(sklearn_preds, y_test, alpha=0.7, color='tab:red')

plt.title("Scikit-learn Random Forest", weight='bold')

plt.xlabel("Predictions", weight='bold')

plt.ylabel("Weight", weight='bold')

plt.show()
print(f"R^2 Score for custom model on the test set: {r2_score(preds, y_test)}")

print(f"R^2 Score for Scikit-Learn model on the test set: {r2_score(sklearn_preds, y_test)}")