import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from plotly.tools import FigureFactory as ff
from time import time
from IPython.display import display

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/train.csv')
data.head()
data = data.drop(columns=['Name','Ticket','Cabin'])
len(data)
data = data.dropna()
len(data)
def entropy(data, y_col):
    # Get all the values for the Y column
    y_val = data[y_col].value_counts().index.values
    # Get the vector with the number of element for each Y class
    tmp = np.array([len(data[data[y_col] == y_val[i]]) for i in range(0, len(y_val))])
    return sum(-tmp / len(data) * np.log2(tmp / len(data)))
def gini_impurity(data, y_col):
    # Get all the values for the Y column
    y_val = data[y_col].value_counts().index.values
    # Get the vector with the number of element for each Y class
    tmp = np.array([len(data[data[y_col] == y_val[i]]) for i in range(0, len(y_val))])
    return 1 - sum((tmp / len(data))**2)
test = pd.DataFrame(data=[['A',1],['A',1],['A',1],['B',1],['B',0],['B',0]]
                    , columns=['letter','bit'])
test
display(test[test['letter'] == 'A'])
display(test[test['letter'] == 'B'])
def info_gain(data, feature_col, y_col, criterion='entropy'):
    # Get all the values for this feature
    feature_val = data[feature_col].value_counts().index.values
    # Get the vector of the number of element for each class
    len_feat = np.array([len(data[data[feature_col] == feature_val[i]]) for i in range(0, len(feature_val))])
    # Get the vector of the criterion for each class
    if criterion == 'entropy':
        crit_feat = np.array([entropy(data[data[feature_col] == feature_val[i]], y_col) for i in range(0, len(feature_val))])
        gain = entropy(data, y_col) - sum((len_feat / len(data)) * crit_feat)
    elif criterion == 'gini':
        crit_feat = np.array([gini_impurity(data[data[feature_col] == feature_val[i]], y_col) for i in range(0, len(feature_val))])
        gain = gini_impurity(data, y_col) - sum((len_feat / len(data)) * crit_feat)
    return gain
class DecisionTree:
    def __init__(self, data, y_col, cat_cols=[], cont_cols=[], criterion='entropy', max_depth=5):
        self.data = data
        self.y_col = y_col
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.criterion = criterion
        self.leaves = list()
        self.max_depth = max_depth if len(cat_cols) > max_depth or len(cont_cols) > 0 else len(cat_cols)
        
    def get_best_split_continuous(self, feature_col, data):
        # Init best gain and best split
        best_gain, best_split = -1, -1
        # Get all the values for this feature
        feat_val = data[feature_col].drop_duplicates().sort_values().reset_index(drop=True).dropna()
        # Get the information gain for each feature and keep the best
        for i in range(1, len(feat_val)):
            split = (feat_val[i - 1] + feat_val[i]) / 2
            data[feature_col + '_tmp'] = data[feature_col] <= split
            gain = info_gain(data, feature_col + '_tmp', self.y_col, criterion=self.criterion)
            best_gain, best_split = (gain, split) if best_gain < gain else (best_gain, best_split)
        return best_split, best_gain
    
    def get_best_feat_leaf(self, data, leaf=None):
        cat_cols = [c for c in self.cat_cols if c not in leaf.get_feat_parent()] if leaf is not None else self.cat_cols
        all_gains = [info_gain(data, c, self.y_col, criterion=self.criterion) for c in cat_cols]
        continuous = [(c, self.get_best_split_continuous(c, data)) for c in self.cont_cols]
        cont_gains = [c[1][1] for c in continuous]

        all_gains = all_gains + cont_gains if len(continuous) > 0 and len(all_gains) > 0 else all_gains if len(
            all_gains) > 0 else cont_gains
        all_cols = cat_cols + self.cont_cols if len(cat_cols) > 0 and len(self.cont_cols) > 0 else cat_cols if len(
            cat_cols) > 0 else cont_cols
        
        best_feat = pd.Series(data=all_gains, index=all_cols).idxmax()
        
        return best_feat if best_feat not in cont_cols else [c for c in continuous if c[0] == best_feat][0]
        
    def learn(self):
        t0 = time()
        print('----- START LEARNING -----')
        # Get the first feature where to split
        feat = self.get_best_feat_leaf(self.data)
        split = None
        
        # If the type is not a string then it's a continuous feature 
        # and we get the best value to split
        if (type(feat) != type(str())):
            split = feat[1][0]
            feat = feat[0]    
        
        # Add it to the Tree
        self.leaves.append(Leaf(None
                                , None
                                , self.data
                                , feat
                                , self.data[self.y_col]
                                , split))
        
        for i in range (1, self.max_depth):
            print('----- BEGIN DEPTH '+str(i)+' at %0.4f s -----' % (time() - t0))
            # Get all the leaves that are in the upper depth
            leaves_parent = [l for l in self.leaves if l.depth == i-1]
            
            # If there is 0 parent we can stop the learning algorithm
            if(len(leaves_parent) == 0):
                break
            else:
                for leaf in leaves_parent:
                    # If there is only one value that means it's useless to split
                    # because we already have our prediction
                    if(len(leaf.values) == 1):
                        continue
                    # Get all values for the current feature
                    feature_val = leaf.data[leaf.feature] <= leaf.split if leaf.split is not None else leaf.data[leaf.feature]
                    feature_val = feature_val.value_counts().index.values
                    
                    # Add all possibilities to the Tree
                    for k in range(0, len(feature_val)): 
                        if leaf.split is None:
                            data = leaf.data[leaf.data[leaf.feature] == feature_val[k]]
                        else:
                            split_cond = leaf.data[leaf.feature] <= leaf.split
                            data = leaf.data[split_cond == feature_val[k]]
                        
                        if len(data) > 0:
                            # Get the best feature for the split
                            next_feat = self.get_best_feat_leaf(data, leaf)

                            split = None

                            # If the type is not a string then it's a continuous feature 
                            # and we get the best value to split
                            if (type(next_feat) != type(str())):
                                split = next_feat[1][0]
                                next_feat = next_feat[0]
                            
                            self.leaves.append(Leaf(prev_leaf=leaf
                                                , condition=feature_val[k]
                                                , data=data
                                                , feature=next_feat
                                                , values=data[self.y_col]
                                                , split=split))
        print('Number of leaves : '+str(len(self.leaves)))
        print('----- END LEARNING : %0.4f s-----' % (time() - t0))
        print()
        
    def display_final_leaves(self):
        leaves = [l for l in self.leaves if len(l.values) == 1 or l.depth == self.max_depth]
        for l in leaves:
            l.display()
                        
    def predict(self, data):
        pred = list()
        for i in range(0, len(data)):
            row = data.iloc[i,:]
            leaf = self.leaves[0]
            while(len(leaf.values) > 1 and leaf.depth < self.max_depth):
                if leaf.split is None:
                    tmp_leaf = [l for l in self.leaves if (l.prev_leaf == leaf and l.condition == row[leaf.feature])]
                else:
                    tmp_leaf = [l for l in self.leaves if (l.prev_leaf == leaf and l.condition == (row[leaf.feature] <= leaf.split))]
                if (len(tmp_leaf) > 0):
                    leaf = tmp_leaf[0]                    
                else:
                    break
            pred.append(leaf.pred_class)
        return pred
    
class Leaf:
    def __init__(self, prev_leaf, condition, data, feature, values, split=None):
        self.prev_leaf = prev_leaf
        self.depth = 0 if prev_leaf is None else prev_leaf.depth+1
        self.condition = condition
        self.data = data
        self.feature = feature
        self.values = values.value_counts(sort=False)
        self.pred_class = self.set_predict_class()
        self.split = split
        
    def set_predict_class(self):
        return self.values.idxmax()
    
    def get_feat_parent(self):
        cols = [self.feature]
        leaf = self
        while(leaf.prev_leaf is not None):
            cols.append(leaf.prev_leaf.feature)
            leaf = leaf.prev_leaf
        return cols

    def display(self):
        cond = ''
        leaf = self
        while(leaf.prev_leaf is not None):
            if leaf.prev_leaf.split is None:
                cond = str(leaf.prev_leaf.feature)+' : '+str(leaf.condition)+' --> '+cond
            else:
                cond = str(leaf.prev_leaf.feature)+' <= '+str(round(leaf.prev_leaf.split,2))+' : '+str(leaf.condition)+' --> '+cond
            leaf = leaf.prev_leaf
        print(cond+' prediction : '+str(self.pred_class))
cat_cols = ['Sex', 'Embarked', 'SibSp', 'Parch']
cont_cols = ['Age', 'Fare']

tree_gini = DecisionTree(data, 'Survived', cat_cols=cat_cols, cont_cols=cont_cols, criterion='gini', max_depth=5)
tree_entr = DecisionTree(data, 'Survived', cat_cols=cat_cols, cont_cols=cont_cols, criterion='entropy', max_depth=5)
tree_gini.learn()
tree_entr.learn()
print('----- GINI TREE -----')
tree_gini.display_final_leaves()
print()
print('----- ENTROPY TREE -----')
tree_entr.display_final_leaves()
def plot_confusion_matrix(y_true, y_pred, name):
    trace = go.Heatmap(z=confusion_matrix(y_true, y_pred),
                       x=['Died', 'Survived'],
                       y=['Died', 'Survived'],
                       colorscale='Reds')
    
    layout = go.Layout(title='Confusion Matrix '+name,
                            xaxis=dict(
                                title='Prediction'
                            ),
                            yaxis=dict(
                                title='Real'
                            )
                        )
    fig = go.Figure(data=[trace], layout=layout)
    
    py.iplot(fig)
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
pred_gini = tree_gini.predict(data)
pred_entr = tree_entr.predict(data)

print('----- GINI TREE -----')
print('F1 Score : '+str(f1_score(data['Survived'], pred_gini)))
print('Accuracy : '+str(accuracy_score(data['Survived'], pred_gini)))
print('----- ENTROPY TREE -----')
print('F1 Score : '+str(f1_score(data['Survived'], pred_entr)))
print('Accuracy : '+str(accuracy_score(data['Survived'], pred_entr)))

plot_confusion_matrix(data['Survived'], pred_gini, 'Gini impurity')
plot_confusion_matrix(data['Survived'], pred_entr, 'Entropy')