# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/credit-card-approval'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

numerical_labels = ['ID', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
                    'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL',
                    'CNT_FAM_MEMBERS', 'MONTHS_BALANCE', 'STATUS']
categorical_labels = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                        'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                        'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']

all_labels = ['CODE_GENDER',
 'FLAG_OWN_CAR',
 'FLAG_OWN_REALTY',
 'CNT_CHILDREN',
 'AMT_INCOME_TOTAL',
 'NAME_INCOME_TYPE',
 'NAME_EDUCATION_TYPE',
 'NAME_FAMILY_STATUS',
 'NAME_HOUSING_TYPE',
 'FLAG_MOBIL',
 'FLAG_WORK_PHONE',
 'FLAG_PHONE',
 'FLAG_EMAIL',
 'OCCUPATION_TYPE',
 'CNT_FAM_MEMBERS',
 'MONTHS_BALANCE']


def read_csv(input_file):
    # Read data from CSV input
    df = pd.read_csv(input_file,header=0, encoding='utf-8')
    # Remove all data entries with missing parameter values
    for all_cat in list(df):
        df[all_cat].replace('', np.nan, inplace=True)
        df.dropna(subset=[all_cat], inplace=True)

    # Categorize string labels into numerical data
    encoder = LabelEncoder()
    for cat in categorical_labels:
        labels = encoder.fit_transform(df[cat])
        df[cat] = labels
    
    # input variables
    x = df.values[:, 1:17]
    # output variables
    y = df.values[:, 17:18]
    
    x_balanced, y_balanced = SMOTE().fit_sample(x, y)

    y_balanced = y_balanced.astype(int)

    # 80/20 split between training and validation set
    x_train, x_test, y_train, y_test = train_test_split(x_balanced, y_balanced, stratify=y_balanced, test_size=0.2)

    # sklearn DT classifiers wants a column vector as input to the output values for multiclass classifiers
    y_trans = np.array(y_train).reshape(-1,1)
    y_train = np.reshape(y_train, len(y_train))

#     # 80/20 split between training and validation set. Can try 90/10 split since there is little data
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    return x_train, x_test, y_train, y_test

# data = read_csv('./data/final_status_preserve2.csv')
data = read_csv('/kaggle/input/credit-card-approval/final_status_preserve2.csv')

def read_fit_predict(data, model_type, depth, leaf, impurity=0.0, samples=1, plot_dimensions=(10, 10, 10)):
    """
    Preprocess data from input file, 
    fit a decision tree, 
    print the test accuracy (80-20 split), 
    print confusion matrix, 
    plot decision tree
    
    Parameters
    ----------
    input_file : str
        csv file to read the data from 
        
    model_type : str
        {'decision_tree' or 'random_forest'}
        
    depth : int
        max_depth of decision tree
        
    leaf : int
        max leaf nodes of decision tree
        
    impurity : int (default = 0.0)
        min_impurity_decrease - A node will be split if this split 
        induces a decrease of the impurity greater than or equal to this value.
        
        Node impurity is a measure of the homogeneity of the labels at the node. 
        For classification problems, can be measured with Gini or entropy. Gini
        is used in this case.
        
        You can say a node is pure when all of its records belong to the same class, 
        such nodes known as the leaf node
        
    samples : int (default = 1)
        min_samples_leaf -  minimum number of samples required to be at a leaf node. 
        A split point at any depth will only be considered if it leaves at least 
        min_samples_leaf training samples in each of the left and right branches.
        
    plot_dimensions : (int, int, int)
        Define the (height, width, fontsize) of decision tree plot
        
    """
    x_train, x_test, y_train, y_test = data
    

    
    if model_type != 'decision_tree' and model_type != 'random_forest':
        model_type = 'decision_tree'
    
    if model_type == 'decision_tree':
        # Fit  Decision Tree
        clf = DecisionTreeClassifier(
            max_depth=depth,
            max_leaf_nodes=leaf,
            min_impurity_decrease=impurity,
            min_samples_leaf=samples
        )
        
    elif model_type == 'random_forest':
        clf = RandomForestClassifier(
            max_depth=depth,
            max_leaf_nodes=leaf,
            min_impurity_decrease=impurity,
            min_samples_leaf=samples
        )
    else:
        clf = DecisionTreeClassifier(
            max_depth=depth,
            max_leaf_nodes=leaf,
            min_impurity_decrease=impurity,
            min_samples_leaf=samples
        )
        
        
    clf.fit(x_train, y_train)

    # Predict test set and print accuracy
    y_predict = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    
    if model_type == 'decision_tree':
        title = "Model: {}\nLeaf: {} | Impurity: {}\nSamples: {} | Depth: {}\nNum Leaves: {}\nTree depth: {}\nAccuracy: {}".format(
            model_type, leaf, impurity, samples, depth, clf.get_n_leaves(), clf.get_depth(), accuracy)
        height, width, fontsize = plot_dimensions
        plt.figure(figsize=(height,width))
        plot_tree(clf, feature_names = all_labels, class_names=[str(x) for x in clf.classes_], filled = True)
    
    else:
        height, width, fontsize = plot_dimensions
        plt.figure(figsize=(height,width))
        plot_tree(clf.estimators_[0], feature_names = all_labels, class_names=[str(x) for x in clf.classes_], filled = True)
        title = "Model: {}\nLeaf: {} | Impurity: {}\nSamples: {} | Depth: {}\nAccuracy: {}".format(
            model_type, leaf, impurity, samples, depth, accuracy)
#     plt.savefig('./plots/{}_{}_{}_{}_{}_{}.png'.format(model_type, leaf, impurity, samples, depth, accuracy))

    disp = plot_confusion_matrix(clf, x_test, y_test,
                                     cmap=plt.cm.Blues
                                )
    disp.ax_.set_title(title, fontdict={'fontsize': 8})
#     disp.figure_.savefig('./plots/{}_{}_{}_{}_{}_{}_params.png'.format(model_type, leaf, impurity, samples, depth, accuracy))

    
    

    return clf
n=1
# for depth in range(250, 501, 50):
#     for leaf in range(250, 501, 50):
depth = None
# leaf = 500
for model in ['decision_tree', 'random_forest']:
    leaf = 500
    clf = read_fit_predict(data=data,
                     model_type = model,
                     depth = depth,
                     leaf = leaf,
                     impurity = 0.00005,
                     samples = 1,
                     plot_dimensions=(30,8,10)
                    )
x_train, x_test, y_train, y_test = data
y_pred = clf.predict(x_test)
confusion_matrix(y_test, y_pred)
n=1
# for depth in range(250, 501, 50):
#     for leaf in range(250, 501, 50):
depth = None
leaf = 1000
models = []
for model in ['decision_tree', 'random_forest']:
    clf = read_fit_predict(data=data,
                     model_type = model,
                     depth = depth,
                     leaf = leaf,
                     impurity = 0.00005,
                     samples = 1,
                     plot_dimensions=(30,8,10)
                    )
    models += [clf]
len(models[1].estimators_)
n=1
# for depth in range(250, 501, 50):
#     for leaf in range(250, 501, 50):
depth = None
leaf = None
models2 = []
for model in ['decision_tree', 'random_forest']:
    clf = read_fit_predict(data=data,
                     model_type = model,
                     depth = depth,
                     leaf = leaf,
                     impurity = 0.00005,
                     samples = 1,
                     plot_dimensions=(30,8,10)
                    )
    models2 += [clf]
n_leaves = [d_t.get_n_leaves() for d_t in models2[1].estimators_]
plt.plot(n_leaves)
