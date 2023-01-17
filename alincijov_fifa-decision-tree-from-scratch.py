from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from collections import Counter
from random import seed
from math import sqrt
from random import randrange

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Distribution graphs (histogram/bar graph) of column data
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()

# Correlation matrix
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()

# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()

df15 = pd.read_csv('/kaggle/input/players_15.csv', delimiter=',', encoding='utf8')
df15.dataframeName = 'players_15.csv'

df16 = pd.read_csv('/kaggle/input/players_16.csv', delimiter=',', encoding='utf8')
df16.dataframeName = 'players_16.csv'

df17 = pd.read_csv('/kaggle/input/players_17.csv', delimiter=',', encoding='utf8')
df17.dataframeName = 'players_17.csv'

df18 = pd.read_csv('/kaggle/input/players_18.csv', delimiter=',', encoding='utf8')
df18.dataframeName = 'players_18.csv'

df19 = pd.read_csv('/kaggle/input/players_19.csv', delimiter=',', encoding='utf8')
df19.dataframeName = 'players_19.csv'

df20 = pd.read_csv('/kaggle/input/players_20.csv', delimiter=',', encoding='utf8')
df20.dataframeName = 'players_20.csv'
df = pd.concat([df15, df16, df17, df18, df19, df20])
# remove duplicates and take the mean
df = df.groupby('short_name').mean().reset_index()
df.dataframeName = 'fifa_players'
df.head(5)
skills_df = df[['pace', 'shooting', 'passing', 'dribbling', 'overall']]
skills_df.dataframeName = 'fifa_players'
skills_df.dropna(inplace=True)
plotCorrelationMatrix(skills_df, 5)
x_df = df[['shooting', 'passing', 'dribbling', 'overall']]
x_df = x_df.dropna()
high_iloc = x_df[(x_df['overall'] >= 80) & (x_df['overall'] < 100)].index.values
medium_iloc = x_df[(x_df['overall'] >= 50) & (x_df['overall'] < 80)].index.values
low_iloc = x_df[x_df['overall'] < 50].index.values
x_df.loc[high_iloc, 'overall'] = 2.0
x_df.loc[medium_iloc, 'overall'] = 1.0
x_df.loc[low_iloc, 'overall'] = 0.0
shooting_mean = x_df['shooting'].mean()
shooting_std = x_df['shooting'].std()

passing_mean = x_df['passing'].mean()
passing_std = x_df['passing'].std()

dribbling_mean = x_df['dribbling'].mean()
dribbling_std = x_df['dribbling'].std()
x_df['shooting'] = (x_df['shooting'] - shooting_mean) / shooting_std
x_df['passing'] = (x_df['passing'] - passing_mean) / passing_std
x_df['dribbling'] = (x_df['dribbling'] - dribbling_mean) / dribbling_std
def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])
class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None
class DecisionTree:

    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (depth >= self.max_depth
                or n_labels == 1
                or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedily select the best split according to information gain
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        
        # grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        # parent loss
        parent_entropy = entropy(y)

        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # compute the weighted avg. of the loss for the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)
        if(len(most_common) > 0):
            most_common = most_common[0][0]
        else:
            most_common = 0.0
        return most_common   
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy
dataset = x_df.to_numpy()

X = np.array(dataset[:,:-1], dtype=np.float64)
y = np.array(dataset[:,-1], dtype=np.int64)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
clf = DecisionTree(max_depth=10)
clf.fit(X_train, y_train)
    
y_pred = clf.predict(X_test)
acc = accuracy(y_test, y_pred)

print ("Accuracy:", acc)
def player_stats_normalized(row):
    pace = (row[0] - pace_mean) / pace_std
    shooting = (row[1] - shooting_mean) / shooting_std
    passing = (row[2] - passing_mean) / passing_std
    dribbling = (row[3] - dribbling_mean) / dribbling_std
    
    return [pace, shooting, passing, dribbling]
def expected_group(overall):
    if(overall >= 80):
        return 2.0
    if(overall >=50):
        return 1.0
    else:
        return 0.0
def player_stats_by_name(name):
    player_stats = df[['shooting', 'passing', 'dribbling', 'overall']].iloc[df[df['short_name'] == name].index[0]].to_numpy()
    player_stats_norm = player_stats_normalized(player_stats)
    player_overall = np.array([expected_group(player_stats[-1])])
    return np.hstack((player_stats_norm, player_overall))
def predict_by_stats(stats):
    predicted = clf.predict(stats[:-1].reshape(1,4))[0]
    expected = stats[-1]
    return float(predicted), expected
# use the decision tree to predict Messi group
messi_stats = player_stats_by_name('L. Messi')
predicted, expected = predict_by_stats(messi_stats)
print("Predicted ", predicted, " Expected ", expected)
players = np.array(df['short_name'].unique())
np.random.shuffle(players)
players_overall = [player_stats_by_name(p) for p in players[:25]]
def remove_nan(players_name_overall):
    indexes = []
    result = None
    for i, (name, stats) in enumerate(players_name_overall):
        if(np.isnan(stats).any()):
            indexes.append(i)
    result = np.delete(players_name_overall, indexes, axis=0)
    return result
players_name_overall = list(map(list, zip(players, players_overall)))
players_name_overall = remove_nan(players_name_overall)
players = players_name_overall[:,0]
players_stats = players_name_overall[:,1]
predicted = []
expected = []
for stats in players_stats:
    pred, expec = predict_by_stats(stats)
    predicted.append(pred)
    expected.append(expec)
players_df_stats = df[df['short_name'].isin(players)][['short_name', 'shooting', 'passing', 'dribbling', 'overall']]
players_df_predicted = pd.DataFrame(list(map(list, zip(players, predicted, expected))), columns=["short_name", "predicted", "expected"])
df_final = pd.merge(players_df_stats, players_df_predicted, how='right', on='short_name')
pd.set_option('display.max_rows', df_final.shape[0]+1)
df_final