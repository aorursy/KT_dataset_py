# Basic

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import math

import statistics 



# Tools

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.manifold import TSNE

from imblearn.over_sampling import SMOTE

from scipy import stats



# Model

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score, cross_val_predict

from sklearn.pipeline import make_pipeline

from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline

from imblearn.under_sampling import NearMiss

from sklearn.decomposition import PCA

from sklearn.decomposition import FastICA

from sklearn.decomposition import TruncatedSVD

from sklearn.random_projection import GaussianRandomProjection

from sklearn.random_projection import SparseRandomProjection



# Algorithms

from sklearn import ensemble, tree, svm, naive_bayes, neighbors, linear_model, gaussian_process, neural_network

import xgboost as xgb

from xgboost.sklearn import XGBClassifier



# Evaluation

from sklearn.metrics import f1_score, accuracy_score, roc_curve, roc_auc_score

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix, recall_score, precision_score, precision_recall_curve

# System

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

df = pd.read_csv("../input/creditcard.csv")

print(df.shape)

print(df.columns)

display(df.sample(5))
rs = RobustScaler()

df['Time'] = rs.fit_transform(df['Time'].values.reshape(-1,1))

rs = RobustScaler()

df['Amount'] = rs.fit_transform(df['Amount'].values.reshape(-1,1))
df['Class'].value_counts()
#Creating The Balanced Data Set

frauds = df.loc[df['Class'] == 1]

nonfrauds = df.loc[df['Class']  == 0][:4920]

undersample = pd.concat([frauds , nonfrauds])
undersample.shape
%%time

X = undersample.drop(['Class'] , axis =1)

Y = undersample['Class']
tsne = TSNE(n_components = 2 , random_state = 32).fit_transform(X.values)

f , (ax1) = plt.subplots(1,1, figsize = (16,8))

ax1.scatter(tsne[:,0] , tsne[:,1] , c = (Y==0) , cmap = 'coolwarm' , label = 'No Fraud')

ax1.scatter(tsne[:,0] , tsne[:,1] , c = (Y==1) , cmap = 'coolwarm' , label = 'Fraud')

ax1.set_title('T-SNE (Fraud vs No Fraud)' , fontsize = 18)

ax1.legend()

ax1.grid(True)
def get_additional_features(train , test ):

    col = list(test.columns)

    n_comp = 2

    #TSVD

    tsvd = TruncatedSVD(n_components = n_comp  , random_state = 98)

    tsvd_result_train = tsvd.fit_transform(train[col])

    tsvd_result_test = tsvd.transform(test[col])

    #PCA

    pca = PCA(n_components = n_comp , random_state = 98)

    pca_result_train = pca.fit_transform(train[col])

    pca_result_test = pca.transform(test[col])

    #FICA

    ica = FastICA(n_components =n_comp , random_state = 98)

    ica_result_train = ica.fit_transform(train[col])

    ica_result_test = ica.transform(test[col])

    #GRP

    grp = GaussianRandomProjection(n_components = n_comp , random_state = 98)

    grp_result_train = grp.fit_transform(train[col])

    grp_result_test = grp.transform(test[col])

    #SRP

    srp = SparseRandomProjection(n_components = n_comp , random_state = 98 , dense_output =True )

    srp_result_train = srp.fit_transform(train[col])

    srp_result_test = srp.transform(test[col])

    for i in range(1,n_comp+1):

        train['tsvd_' + str(i)] = tsvd_result_train[:, i - 1]

        test['tsvd_' + str(i)] = tsvd_result_test[:, i - 1]

        train['pca_' + str(i)] = pca_result_train[:, i - 1]

        test['pca_' + str(i)] = pca_result_test[:, i - 1]

        train['ica_' + str(i)] = ica_result_train[:, i - 1]

        test['ica_' + str(i)] = ica_result_test[:, i - 1]

        train['grp_' + str(i)] = grp_result_train[:, i - 1]

        test['grp_' + str(i)] = grp_result_test[:, i - 1]

        train['srp_' + str(i)] = srp_result_train[:, i - 1]

        test['srp_' + str(i)] = srp_result_test[:, i - 1]

    return train ,test
x_train , x_test , y_train , y_test = train_test_split(X , Y , test_size = 0.2 , random_state = 32)

x_train , x_test , y_train , y_test = x_train.values , x_test.values , y_train.values ,y_test.values
x=[]

y=[]

z=[]

t=[]

u=[]

v=[]

w=[]

for i in range(1,1000,1):

    no = i/1000

    x.append(no)

    y.append(-no*math.log2(no))

    z.append(1 - no**2)

    t.append((-no**2 * math.log2(no)))

    u.append(1-((-no * math.log2(no)) + (no**2)))

    v.append(-math.log10(no))

#     w.append((no * math.log2(no)) + (1-no**2))

    

f , (ax1) = plt.subplots(1,1, figsize = (16,8))

ax1.scatter(x, z,cmap = 'coolwarm' , label='Gini Index',marker = '.')

ax1.scatter(x, y,cmap = 'coolwarm' , label='Entropy Index',marker = '.')

ax1.scatter(x, u,cmap = 'coolwarm' , label='Gini Index and Entropy Index',marker = '.')

ax1.scatter(x[50:], v[50:],cmap = 'coolwarm' , label='Suprisal Index',marker = '.')

ax1.scatter(x, t,cmap = 'coolwarm' , label='Custom Index',marker = '.')

ax1.set_title('Attribute Selection' , fontsize = 14)

ax1.legend()

ax1.grid(True)
def compute_gini_impurity(labels):

    y = labels

    categories = np.unique(y)

    p2_sum = 0

    for cat in categories:

        p_cat = (y == cat).sum() / y.size

        p2_sum += p_cat ** 2

    

    return 1 - p2_sum



def compute_entropy(labels):

    y = labels

    categories = np.unique(y)

    entropy = 0

    for cat in categories:

        p_cat = (y == cat).sum() / y.size

        entropy += -p_cat * math.log2(p_cat)

        

    return entropy



def compute_custom(labels):

    y = labels

    categories = np.unique(y)

    entropy = 0

    for cat in categories:

        p_cat = (y == cat).sum() / y.size

        entropy += (1-((-p_cat * math.log2(p_cat)) + (p_cat**2)))

#         entropy += -log10(p_cat)

    return entropy   
def find_best_split_value(col, label, criterion="gini", categorical_col=False):

    assert criterion == "gini" or criterion == "entropy" or criterion == 'custom', "Invalid criterion: {}".format(criterion)

    len_col = len(col)

    split_vals = np.unique(col)

    split_ginis = np.ones_like(split_vals)

    if categorical_col:

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

                print("here1")

                gini += (len(splits[j][0]) / len_col) * compute_gini_impurity(splits[j][1])

            elif criterion =='entropy':

                print("here2")

                gini += (len(splits[j][0]) / len_col) * compute_entropy(splits[j][1])

            else:

                print("here3")

                gini += (len(splits[j][0]) / len_col) * compute_entropy(splits[j][1])

                

        return None, gini

    else:

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

                    gini += (len(splits[j][0]) / len_col) * compute_gini_impurity(splits[j][1])

                else:

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

               depth=0, max_depth=5, 

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

    def __init__(self, max_depth=3, min_split_size=5, criterion="gini"):

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
model = DecisionTreeClassifier(criterion="custom")

model = model.fit(x_train , y_train)

conf_matrix = confusion_matrix(y_true=y_test, y_pred=model.predict(x_test))

fig, ax = plt.subplots(figsize=(7.5, 7.5))

ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)

for i in range(conf_matrix.shape[0]):

    for j in range(conf_matrix.shape[1]):

        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

        

plt.xlabel('Predictions', fontsize=18)

plt.ylabel('Actuals', fontsize=18)

plt.title('Confusion Matrix', fontsize=18)

plt.show()

model = DecisionTreeClassifier(criterion="gini")

model = model.fit(x_train , y_train)

conf_matrix = confusion_matrix(y_true=y_test, y_pred=model.predict(x_test))

fig, ax = plt.subplots(figsize=(7.5, 7.5))

ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)

for i in range(conf_matrix.shape[0]):

    for j in range(conf_matrix.shape[1]):

        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

        

plt.xlabel('Predictions', fontsize=18)

plt.ylabel('Actuals', fontsize=18)

plt.title('Confusion Matrix', fontsize=18)

plt.show()

model = DecisionTreeClassifier(criterion="entropy")

model = model.fit(x_train , y_train)

conf_matrix = confusion_matrix(y_true=y_test, y_pred=model.predict(x_test))

fig, ax = plt.subplots(figsize=(7.5, 7.5))

ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)

for i in range(conf_matrix.shape[0]):

    for j in range(conf_matrix.shape[1]):

        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

        

plt.xlabel('Predictions', fontsize=18)

plt.ylabel('Actuals', fontsize=18)

plt.title('Confusion Matrix', fontsize=18)

plt.show()

accuracy_score(

    y_true=y_test,

    y_pred=model.predict(x_test)

)
MLA = [

    ensemble.AdaBoostClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),

    gaussian_process.GaussianProcessClassifier(),

    linear_model.LogisticRegressionCV(),

    linear_model.RidgeClassifierCV(),

    linear_model.Perceptron(),

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    neighbors.KNeighborsClassifier(),

    svm.SVC(probability=True),

    svm.NuSVC(probability=True),

    svm.LinearSVC(),

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    xgb.XGBClassifier()

    ]
MLA = [

    tree.DecisionTreeClassifier(criterion = 'gini'),

    tree.DecisionTreeClassifier(criterion = 'entropy'),

    ]
%%time

col = []

algorithms = pd.DataFrame(columns = col)

ind= 0

for a in MLA:

    a.fit(x_train , y_train)

    pred = a.predict(x_test)

    acc = accuracy_score(y_test , pred)

    f1 = f1_score(y_test , pred)

    cross_score = cross_val_score(a, x_train , y_train).mean()

    Alg = a.__class__.__name__

    

    algorithms.loc[ind , 'Algorithm'] = Alg

    algorithms.loc[ind , 'Accuracy'] =round( acc * 100 , 2)

    algorithms.loc[ind , 'F1_score'] = round( f1 * 100 , 2)

    algorithms.loc[ind , 'Cross_val_score'] = round( cross_score * 100 , 2)

    ind+=1
algorithms.sort_values(by=['Cross_val_score'] ,  ascending = False , inplace = True)
algorithms.head()
g = sns.barplot('Cross_val_score' , 'Algorithm' , data = algorithms)

g.set_xlabel('CV_score')

g.set_title('Algo Score')
