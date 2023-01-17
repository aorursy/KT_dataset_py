%matplotlib inline

import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt





from sklearn.model_selection import cross_val_score, train_test_split

from sklearn import metrics





#data = pd.read_csv(os.path.join("data", "loan_sub.csv"), sep=',')

data = pd.read_csv(os.path.join("../input", "loan_sub.csv"), sep=',')
data.columns
# safe_loans =  1 => safe

# safe_loans = -1 => risky

data = pd.read_csv(os.path.join("../input", "loan_sub.csv"), sep=',')

data['safe_loans'] = data['bad_loans'].apply(lambda x: 1 if x == 0 else -1)

data.drop('bad_loans', axis = 1, inplace = True)

data['safe_loans'].value_counts(normalize=True)
cols = ['grade', 'term','home_ownership', 'emp_length']

target = 'safe_loans'



data = data[cols + [target]]

data.head()
data['safe_loans'].value_counts()



# use the percentage of bad and good loans to undersample the safe loans.

bad_ones = data[data['safe_loans'] == -1]

safe_ones = data[data['safe_loans'] == 1]

percentage = bad_ones.shape[0] / safe_ones.shape[0]



risky_loans = bad_ones

safe_loans = safe_ones.sample(frac = percentage, random_state= 3)



# combine two kinds of loans

data_set = pd.concat([risky_loans, safe_loans], axis = 0)
data_set[target].value_counts(normalize=True)
def label_encode(data, columns=['pclass','name_title','embarked', 'sex']):

    for col in columns:

        data[col] = data[col].apply(lambda x: str(x))

        new_cols = [col + '_' + i for i in data[col].unique()]

        data = pd.concat([data, pd.get_dummies(data[col], prefix=col)[new_cols]], axis=1)

        del data[col]

    return data
#grade, home_ownership, target

cols = ['grade', 'term','home_ownership', 'emp_length']

data_set = label_encode(data_set, cols)

data_set.head()
train_data, test_data = train_test_split(data_set, test_size = 0.2, random_state = 3)

trainX, trainY = train_data[train_data.columns[1:]], pd.DataFrame(train_data[target])

testX, testY = test_data[test_data.columns[1:]], pd.DataFrame(test_data[target])

def count_errors(labels_in_node):

    """

    Input: (Numpy Array/Pandas series)labels in node, eg: [-1,-1,1,-1,1]

    Output: (Int) if we do the major class voting, how many errors we make?

    """

    if len(labels_in_node) == 0:

        return 0

    

    positive_ones = labels_in_node[labels_in_node == 1].shape[0]

    negative_ones = labels_in_node[labels_in_node == -1].shape[0]

    

    return negative_ones / len(labels_in_node) if positive_ones > negative_ones else positive_ones / len(labels_in_node)





def best_split(data, features, target):

    """

    We want to select out the best feature such that it splits the data best based on your measurement(IG/accuracy)

    Input: (Pandas DataFrame)data

           (List of String) features  candidates we can choose feature from

           (String) target  the target name we shoot for. eg: 'safe_loan' 

           

    Output: (String) the best feature

    """

    # return the best feature

    best_feature = None

    best_error = 2.0 

    num_data_points = float(len(data))  



    for feature in features:

        

        # 左分支对应当前特征为0的数据点

        left_split = data[data[feature] == 0]

        

        # 右分支对应当前特征为1的数据点

        right_split = data[data[feature] == 1]

        

        # 计算左边分支里犯了多少错

        left_misses = count_errors(left_split[target])



        # 计算右边分支里犯了多少错

        right_misses = count_errors(right_split[target])

            

        # 计算当前划分之后的分类犯错率

        error = left_misses * len(left_split) / num_data_points + right_misses * len(right_split) / num_data_points



        # 更新应选特征和错误率，注意错误越低说明该特征越好

        if error < best_error:

            best_error = error

            best_feature = feature

    return best_feature
def entropy(labels_in_node):

    # 二分类问题: 0 or 1

    n = len(labels_in_node)

    s1 = (labels_in_node==1).sum()

    if s1 == 0 or s1 == n: # indicates the labels are the same~

        return 0

    

    p1 = float(s1) / n

    p0 = 1 - p1

    return -p0 * np.log2(p0) - p1 * np.log2(p1)





def best_split_entropy(data, features, target):

    """

    We want to select out the best feature such that it splits the data best based on your measurement(IG/accuracy)

    Input: (Pandas DataFrame)data

           (List of String) features  candidates we can choose feature from

           (String) target  the target name we shoot for. eg: 'safe_loan' 

           

    Output: (String) the best feature

    """

    

    best_feature = None

    best_info_gain = float('-inf') 

    num_data_points = float(len(data))

    # 计算划分之前数据集的整体熵值

    entropy_original = entropy(data[target])



    for feature in features:

        

        # 左分支对应当前特征为0的数据点

        left_split = data[data[feature] == 0]

        

        # 右分支对应当前特征为1的数据点

        right_split = data[data[feature] == 1]

        

        # 计算左边分支的熵值

        left_entropy = entropy(left_split[target])



        # 计算右边分支的熵值

        right_entropy = entropy(right_split[target])

            

        # 计算左边分支与右分支熵值的加权和（数据集划分后的熵值）

        entropy_split = left_entropy * len(left_split) / num_data_points + right_entropy * len(right_split) / num_data_points

        

        # 计算划分前与划分后的熵值差得到信息增益

        info_gain = entropy_original - entropy_split



        # 更新最佳特征和对应的信息增益的值

        if info_gain > best_info_gain:

            best_info_gain = info_gain

            best_feature = feature

    return best_feature

    
class TreeNode:

    def __init__(self, is_leaf, prediction, split_feature):

    # TODO

        self.is_leaf = is_leaf

        self.prediction = prediction

        self.split_feature = split_feature

        self.left, self.right = None, None

        
from sklearn.base import BaseEstimator

from sklearn.metrics import accuracy_score

class MyDecisionTree(BaseEstimator):

    

    def __init__(self, max_depth, min_error, split_way = 1, debug = False):

        self.max_depth = max_depth

        self.min_error = min_error

        self.split_way = split_way

        self.debug = debug

    

    def fit(self, X, Y, data_weights = None):

        

        data_set = pd.concat([X, Y], axis=1)

        features = X.columns

        target = Y.columns[0]

        self.root_node = self.create_tree(data_set, features, target, 0, self.max_depth, self.min_error)

        

        

    def predict(self, X):

        """

        Input:  (Pandas DataFrame/Numpy array, size: m * n) a matrix and each row indicates a data point

        Output: (Pandas DataFrame/Numpy array, size: m * 1) array of the predicted result

        

        Tips: each row is predicted by the function predict_single_data()

        """

        prediction = X.apply(lambda row: self.predict_single_data(self.root_node, row), axis=1)

        return prediction

        

        

    def score(self, testX, testY):

        """

        Tips: use defined predict function to get predicted result and compare it with testY

        """

        target = testY.columns[0]

        result = self.predict(testX)

        return accuracy_score(testY[target], result)

    

    

    def create_tree(self, data, features, target, current_depth = 0, max_depth = 10, min_error=0):

        """

        Input

            data: (pandas data frame) the input data

            features: (pandas series/dataframe/numpy array) available features

            target: (pandas series/dataframe/numpy array)  the target to predict

            current_depth: (Int)  current depth of the tree

            max_depth: (Int)  the maximum depth of the tree

            min_error: (Float) the minimum error reduction  

            

        Output:

            (TreeNode)  root

        """        

        

        

        

        """

        探索三种不同的终止划分数据集的条件  

  

        termination 1, 当错误率降到min_error以下, 终止划分并返回叶子节点  

        termination 2, 当特征都用完了, 终止划分并返回叶子节点  

        termination 3, 当树的深度等于最大max_depth时, 终止划分并返回叶子节点

        """

        

    

        # 拷贝以下可用特征

        remaining_features = features[:]



        target_values = data[target]



        # termination 1   bonus task

        if count_errors(target_values) <= min_error:

            if self.debug:

                print("Termination 1 reached.")     

            return self.create_leaf(target_values)



        # termination 2

        if len(remaining_features) == 0:

            if self.debug:

                print("Termination 2 reached.")    

            return self.create_leaf(target_values)



        # termination 3

        if current_depth >= max_depth: 

            if self.debug:

                print("Termination 3 reached.")

            return self.create_leaf(target_values)







        # 选出最佳当前划分特征

        # split way == 1: 根據entropy劃分

        # split way == 2: 根據error劃分

        if self.split_way == 1:

            split_feature = best_split_entropy(data, features, target)# TODO  # 根据信息增益来划分

        else:

            split_feature = best_split(data, features, target)



        # 选出最佳特征后，该特征为0的数据分到左边，该特征为1的数据分到右边

        left_split = data[data[split_feature] == 0]

        right_split = data[data[split_feature] == 1]



        # 剔除已经用过的特征

        remaining_features = remaining_features.drop(split_feature)

        if self.debug:

            print("Split on feature %s. (%s, %s)" % (split_feature, str(len(left_split)), str(len(right_split))))



        # 如果当前数据全部划分到了一边，直接创建叶子节点返回即可

        if len(left_split) == len(data):

            if self.debug:

                print("Perfect split!")

            return self.create_leaf(left_split[target])

        if len(right_split) == len(data):

            if self.debug:

                print("Perfect split!")

            return self.create_leaf(right_split[target])



        # 递归上面的步骤

        left_tree = self.create_tree(left_split, remaining_features, target, current_depth + 1, max_depth, min_error)  

        right_tree = self.create_tree(right_split, remaining_features, target, current_depth + 1, max_depth, min_error) 



        #生成当前的树节点

        result_node = TreeNode(False, None, split_feature)

        result_node.left = left_tree

        result_node.right = right_tree

        return result_node    

    

    

    

    def create_leaf(self, target_values):

        """

        Input: (Pandas DataFrame/Numpy Array)  target_values  eg: [-1,1,-1,-1,1]

        Output: (TreeNode) node   Note that you should fill in the correct information for each attribute of the result

        """

        # 用于创建叶子的函数



        # 初始化一个树节点

        leaf = TreeNode(True, None, None)



        # 统计当前数据集里标签为+1和-1的个数，较大的那个即为当前节点的预测结果

        num_positive_ones = target_values[target_values == 1].shape[0]

        num_negative_ones = target_values[target_values == -1].shape[0]



        if num_positive_ones > num_negative_ones:

            leaf.prediction = 1

        else:

            leaf.prediction = -1



        # 返回叶子        

        return leaf 

    

    

    

    def predict_single_data(self, tree, x, annotate = False): 

        """

        Input:  (TreeNode)  tree

                (Pandas DataFrame) x  it's a single array or one row from a pandas dataframe (one data point)

                (Bool)  annotate  if intermediate result is displayed

        Output:  (Int)  -1 or 1 in our case

        """

        

        

        # 如果已经是叶子节点直接返回叶子节点的预测结果

        if tree.is_leaf:

            if annotate: 

                print("leaf node, predicting %s" % tree.prediction)

            return tree.prediction

        else:

            # 查询当前节点用来划分数据集的特征

            split_feature_value = x[tree.split_feature]



            if annotate: 

                print("Split on %s = %s" % (tree.split_feature, split_feature_value))

            if split_feature_value == 0:

                #如果数据在该特征上的值为0，交给左子树来预测

                return self.predict_single_data(tree.left, x, annotate)

            else:

                #如果数据在该特征上的值为0，交给右子树来预测

                return self.predict_single_data(tree.right, x, annotate)  

    

    def count_leaves(self):

        return self.count_leaves_helper(self.root_node)

    

    def count_leaves_helper(self, tree):

        return 1 if tree.is_leaf else self.count_leaves_helper(tree.left) + self.count_leaves_helper(tree.right)

        

    
model_entropy = MyDecisionTree(max_depth = 10, min_error = 1e-15, split_way = 1, debug = True)

model_error = MyDecisionTree(max_depth = 10, min_error = 1e-15, split_way = 2, debug = True)

model_entropy.fit(trainX, trainY)
model_error.fit(trainX, trainY)
model_entropy.score(testX, testY)
model_error.score(testX, testY)
m.count_leaves()
model_1 = MyDecisionTree(max_depth=3, min_error = 1e-15) 

model_2 = MyDecisionTree(max_depth=7, min_error = 1e-15) 

model_3 = MyDecisionTree(max_depth=15, min_error = 1e-15) 
model_1.fit(trainX, trainY)

model_2.fit(trainX, trainY)

model_3.fit(trainX, trainY)
print("model_1 training accuracy :", model_1.score(trainX, trainY))

print("model_2 training accuracy :", model_2.score(trainX, trainY))

print("model_3 training accuracy :", model_3.score(trainX, trainY))
print("model_1 testing accuracy :", model_1.score(testX, testY))

print("model_2 testing accuracy :", model_2.score(testX, testY))

print("model_3 testing accuracy :", model_3.score(testX, testY))
print("model_1 complexity is: ", model_1.count_leaves())

print("model_2 complexity is: ", model_2.count_leaves())

print("model_3 complexity is: ", model_3.count_leaves())
model_err1 = MyDecisionTree(max_depth=10, min_error = 0.6, debug = False) 

model_err2 = MyDecisionTree(max_depth=10, min_error = 0.4, debug = False) 

model_err3 = MyDecisionTree(max_depth=10, min_error = 0.2, debug = False) 
model_err1.fit(trainX, trainY)

model_err2.fit(trainX, trainY)

model_err3.fit(trainX, trainY)
print("model_1 training accuracy :", model_err1.score(trainX, trainY))

print("model_2 training accuracy :", model_err2.score(trainX, trainY))

print("model_3 training accuracy :", model_err3.score(trainX, trainY))
print("model_1 testing accuracy :", model_err1.score(testX, testY))

print("model_2 testing accuracy :", model_err2.score(testX, testY))

print("model_3 testing accuracy :", model_err3.score(testX, testY))
print("model_1 complexity is: ", model_err1.count_leaves())

print("model_2 complexity is: ", model_err2.count_leaves())

print("model_3 complexity is: ", model_err3.count_leaves())