# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
games = pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')
games.head()
pd.isnull(games).sum() > 0
len(games)

values = {'Critic_Score': 0, 'Critic_Count': 0, 'User_Score': 0, 'User_Count': 0}
games['Critic_Score'].fillna(0, inplace=True)
games['Critic_Count'].fillna(0, inplace=True)
games['User_Score'].fillna(0, inplace=True)
games['User_Count'].fillna(0, inplace=True)
games.loc[games['Publisher']=='Nintendo', ['Developer']] = 'Nintendo'
games.loc[games['Publisher']=='Nintendo', ['Rating']] = 'E'
games.loc[games['Publisher']=='Activision', ['Rating']] = 'M'
games.loc[games['Publisher']=='Activision', ['Developer']] = 'Treyarch'
# the user score to 0 where user score is tbd
games.loc[games['User_Score']=='tbd', ['User_Score']] = 0
#convert user score to a np.number
games['User_Score']=games['User_Score'].astype(np.number)
del games['Year_of_Release']
games = games.dropna(how='any',axis=0)
pd.isnull(games).sum() > 0
len(games)
games.head()
# m_or_e = ['E', 'M']
# ratingsEM = games.loc[games['Rating'].isin(m_or_e)]
# len(ratingsEM)
ratingsEM = games.copy()
ratingsEM.head()
del ratingsEM['NA_Sales']
del ratingsEM['EU_Sales']
del ratingsEM['JP_Sales']
del ratingsEM['Other_Sales']
ratingsEM.head()
def create_label_encoder_dict(df):
    from sklearn.preprocessing import LabelEncoder

    label_encoder_dict = {}
    for column in df.columns:
        # Only create encoder for categorical data types
        if not np.issubdtype(df[column].dtype, np.number) and column != 'Age':
            label_encoder_dict[column]= LabelEncoder().fit(df[column].astype(str))
    return label_encoder_dict
label_encoders = create_label_encoder_dict(ratingsEM)
print("Encoded Values for each Label")
print("="*32)
for column in label_encoders:
    print("="*32)
    print('Encoder(%s) = %s' % (column, label_encoders[column].classes_ ))
    print(pd.DataFrame([range(0,len(label_encoders[column].classes_))], columns=label_encoders[column].classes_, index=['Encoded Values'] ).T)
ratingsEM_data = ratingsEM.copy() # create copy of initial data set
for column in ratingsEM_data.columns:
    if column in label_encoders:
        ratingsEM_data[column] = label_encoders[column].transform(ratingsEM_data[column])
print("Transformed data set")
print("="*32)
ratingsEM_data.head()
X_data = ratingsEM_data.iloc[:, :-1]
Y_data = ratingsEM_data['Rating']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30)
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
# Create the classifier with a maximum depth of 2 using entropy as the criterion for choosing most significant nodes
# to build the tree
clf = DecisionTreeClassifier(criterion='entropy',min_samples_split=2)
# Hint : Change the max_depth to 10 or another number to see how this affects the tree
clf.fit(X_train, y_train)
pd.DataFrame([ "%.2f%%" % perc for perc in (clf.feature_importances_ * 100)], \
             index = X_data.columns, columns = ['Feature Significance in Decision Tree'])
import graphviz
dot_data = tree.export_graphviz(clf,out_file=None,
feature_names=X_data.columns,
class_names=label_encoders[Y_data.name].classes_,
filled=True, rounded=True, proportion=True,
node_ids=True, #impurity=False,
special_characters=True)
graph = graphviz.Source(dot_data)
graph
def tree_to_code(tree, feature_names, label_encoders={}):
    from sklearn.tree import _tree
    '''
     Outputs a decision tree model as a Python function
     Parameters:
     -----------
     tree: decision tree model
     The decision tree to represent as a function
     feature_names: list
     The feature names of the dataset used for building the decision tree
     '''
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def decision_tree({}):".format(", ".join(feature_names)))
    def recurse(node, depth):
        indent = " " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else: # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            #print(node)

            name = tree_.feature[node]
            if name in label_encoders:
                if isinstance(label_encoders[name] , LabelEncoder) or True:
                    print ("{}-return {}".format(indent, label_encoders[name].inverse_transform(tree_.value[node])))
                    return
            print("{}return {} # Distribution of samples in node".format(indent, tree_.value[node]))
    recurse(0, 1)
print("Decision Tree Rules")
print("="*32)
tree_to_code(clf, X_data.columns, label_encoders)
label_encoders = create_label_encoder_dict(ratingsEM)
print("Encoded Values for each Label")
print("="*32)
for column in label_encoders:
    print("="*32)
    print('Encoder(%s) = %s' % (column, label_encoders[column].classes_ ))
    print(pd.DataFrame([range(0,len(label_encoders[column].classes_))], columns=label_encoders[column].classes_, index=['Encoded Values'] ).T)
k=(clf.predict(X_test) == y_test) # Determine how many were predicted correctly
k.value_counts()
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, clf.predict(X_test), labels=y_test.unique())
cm
def plot_confusion_matrix(cm, classes,
    normalize=False,
    title='Confusion matrix',
    cmap=plt.cm.Blues):
    import itertools
    """
     This function prints and plots the confusion matrix.
     Normalization can be applied by setting `normalize=True`.
     """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
plot_confusion_matrix(cm,ratingsEM['Rating'].unique())
