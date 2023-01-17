%matplotlib inline
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import metrics



#data = pd.read_csv(os.path.join("data", "loan_sub.csv"), sep=',')
#data = pd.read_csv(os.path.join("/Users/jiadileng/Desktop/machine learning/jiuzhang/week4/input", "loan_sub.csv"), sep=',')
data = pd.read_csv(os.path.join("../input", "loan_sub.csv"), sep=',')
data.columns
data.head()
# safe_loans =  1 => safe
# safe_loans = -1 => risky
#TODO
data['safe_loans'] = data['bad_loans'].apply(lambda x: +1 if x == 0 else -1)
data = data.drop('bad_loans', axis=1)
data['safe_loans'].value_counts(normalize=True)
cols = ['grade', 'term','home_ownership', 'emp_length']
target = 'safe_loans'

data = data[cols + [target]]
data.head()
data['safe_loans'].value_counts()

# use the percentage of bad and good loans to undersample the safe loans.
bad_ones = data[data[target] == -1]# TODO
safe_ones = data[data[target] == 1]# TODO
percentage = len(bad_ones) / float(len(safe_ones))
#percentage = float(len(bad_ones) / len(safe_ones))


risky_loans = bad_ones
safe_loans = safe_ones.sample(frac=percentage, random_state=33)#根据刚才算出来的比例对好的贷款进行随机抽样

# combine two kinds of loans
data_set = pd.concat([risky_loans, safe_loans], axis=0)
data_set[target].value_counts(normalize=True)
#actually this is not dummies but one-hot encoding
def dummies(data, columns=['pclass','name_title','embarked', 'sex']): 
    for col in columns:
        data[col] = data[col].apply(lambda x: str(x))
        new_cols = [col + '_' + i for i in data[col].unique()]
        data = pd.concat([data, pd.get_dummies(data[col], prefix=col)[new_cols]], axis=1)
        del data[col]
    return data
#grade, home_ownership, target
cols = ['grade', 'term','home_ownership', 'emp_length']
data_set = dummies(data_set, columns=cols)
data_set.head()
train_data, test_data = train_test_split(data_set, test_size=0.2, random_state=33)#TODO
trainX, trainY = train_data[train_data.columns[1:]], pd.DataFrame(train_data[target])#TODO
testX, testY = test_data[test_data.columns[1:]], pd.DataFrame(test_data[target])

def entropy(labels_in_node):
    # 二分类问题: 0 or 1
    n = len(labels_in_node)
    s1 = (labels_in_node==1).sum()
    if s1 == 0 or s1 == n: # indicates the labels are the same~
        return 0
    
    p1 = float(s1) / n
    p0 = 1 - p1
    return -p0 * np.log2(p0) - p1 * np.log2(p1) #比较问题与底数是几无关


def best_split_entropy(data, features, target):
    
    best_feature = None
    best_info_gain = float('-inf') 
    num_data_points = float(len(data))
    # 计算划分之前数据集的整体熵值
    entropy_original = entropy(data[target])#划分前的熵值

    for feature in features:
        
        # 左分支对应当前特征为0的数据点
        left_split = data[data[feature]==0]
        
        # 右分支对应当前特征为1的数据点
        right_split = data[data[feature]==1] 
        
        # 计算左边分支的熵值
        left_entropy = entropy(left_split[target])           

        # 计算右边分支的熵值
        right_entropy = entropy(right_split[target])
            
        # 计算左边分支与右分支熵值的加权和（数据集划分后的熵值）
        entropy_split = len(left_split)/num_data_points * left_entropy + len(right_split)/num_data_points * right_entropy#TODO
        
        # 计算划分前与划分后的熵值差得到信息增益
        info_gain = entropy_original - entropy_split

        # 更新最佳特征和对应的信息增益的值
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = feature
    return best_feature
    
class TreeNode:
    def __init__(self, is_leaf, prediction, split_feature):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.split_feature = split_feature
        self.left = None
        self.right = None
    
        
        
from sklearn.base import BaseEstimator #模型中的父类
from sklearn.metrics import accuracy_score
class MyDecisionTree(BaseEstimator):
    
    def __init__(self, max_depth, min_error):
        self.max_depth = max_depth
        self.min_error = min_error
    
    def fit(self, X, Y, data_weights = None):
        
        data_set = pd.concat([X, Y], axis=1) #concatenate pandas objects along a particular axis
        features = X.columns
        target = Y.columns[0]#Y其实只有一列
        self.root_node = self.create_tree(data_set, features, target, current_depth=0, max_depth=self.max_depth, min_error=self.min_error)
    
    def create_tree(self, data, features, target, current_depth = 0, max_depth = 10, min_error=0):
        """
        探索三种不同的终止划分数据集的条件  
  
        termination 1, 当错误率降到min_error以下, 终止划分并返回叶子节点  
        termination 2, 当特征都用完了, 终止划分并返回叶子节点  
        termination 3, 当树的深度等于最大max_depth时, 终止划分并返回叶子节点
        """
        
    
        # 拷贝一下可用特征
        remaining_features = features[:]

        target_values = data[target]

        
        
        #################
        #  第一部分： 递归出口！
        #################
        # termination 1   bonus task
        #if count_errors(target_values) <= min_error:
            #print("Termination 1 reached.")     
            #return self.create_leaf(target_values) 

        # termination 2
        if len(remaining_features) == 0:
            print("Termination 2 reached.")    
            return self.create_leaf(target_values) #返回叶子节点

        # termination 3
        if current_depth >= max_depth: 
            print("Termination 3 reached.")
            return self.create_leaf(target_values)


        
        
        #################
        #  第二部分： 如果继续划分，划分数据集！
        #################
        
        # 选出最佳当前划分特征
        #split_feature = best_split(data, features, target)   #根据正确率划分   bonus task
        split_feature = best_split_entropy(data, features, target)  # 根据信息增益来划分

        # 选出最佳特征后，该特征为0的数据分到左边，该特征为1的数据分到右边
        left_split = data[data[split_feature]==0]
        right_split = data[data[split_feature]==1]

        # 剔除已经用过的特征
        remaining_features = remaining_features.drop(split_feature)
        print("Split on feature %s. (%s, %s)" % (split_feature, str(len(left_split)), str(len(right_split))))

        # 如果当前数据全部划分到了一边，就不需要再划分了，直接创建叶子节点返回即可
        if len(left_split) == len(data):
            print("Perfect split!")
            return self.create_leaf(left_split[target])
        if len(right_split) == len(data):
            print("Perfect split!")
            return self.create_leaf(right_split[target])

        # 递归上面的步骤
        left_tree = self.create_tree(left_split, remaining_features, target, current_depth + 1, max_depth, min_error)     
        right_tree = self.create_tree(right_split, remaining_features, target, current_depth + 1, max_depth, min_error)
        
        
        #################
        #  第三部分： 左右递归完之后，把当前结果返回！
        #################
        
        #生成当前的树节点
        result_node = TreeNode(False, None, split_feature)#False代表它不是叶子节点
        result_node.left = left_tree
        result_node.right = right_tree
        return result_node    
    
    
    
    def create_leaf(self, target_values):
        # 用于创建叶子的函数

        # 初始化一个树节点
        leaf = TreeNode(True, None, None)

        # 统计当前数据集里标签为+1和-1的个数，较大的那个即为当前节点的预测结果
        num_positive_ones = len(target_values[target_values == +1])
        num_negative_ones = len(target_values[target_values == -1])

        if num_positive_ones > num_negative_ones:
            leaf.prediction = 1
        else:
            leaf.prediction = -1

        # 返回叶子        
        return leaf     
    
    
        
    def predict(self, X):
        #某一行拿出来，给定结果，把结果应用到所有行
        prediction = X.apply(lambda row: self.predict_single_data(self.root_node, row), axis=1)
        return prediction
    
    
    
    def predict_single_data(self, tree, X, annotate = False):   
        # 如果已经是叶子节点直接返回叶子节点的预测结果
        if tree.is_leaf:
            if annotate: 
                print("leaf node, predicting %s" % tree.prediction)
            return tree.prediction
        else:
            # 查询x对应当前节点特征的值
            split_feature_value = X[tree.split_feature]

            if annotate: 
                print("Split on %s = %s" % (tree.split_feature, split_feature_value))
            if split_feature_value == 0:
                #如果x在该特征上的值为0，交给左子树来预测
                return self.predict_single_data(tree.left, X, annotate) #annotate---是否打印中间信息
            else:
                #如果x在该特征上的值为1，交给右子树来预测
                return self.predict_single_data(tree.right, X, annotate)
    
    
    
        
    def score(self, testX, testY):
        target = testY.columns[0]
        result = self.predict(testX)
        return accuracy_score(testY[target], result)


    def count_leaves(self):
        return self.count_leaves_helper(self.root_node)
    
    def count_leaves_helper(self, tree):
        if tree.is_leaf:
            return 1
        return self.count_leaves_helper(tree.left) + self.count_leaves_helper(tree.right)
    

m = MyDecisionTree(max_depth = 10, min_error = 1e-15)
m.fit(trainX, trainY)
m.score(testX, testY)
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
def count_errors(labels_in_node):
    if len(labels_in_node) == 0:
        return 0
    
    positive_ones = labels_in_node.apply(lambda x: x==1).sum()
    negative_ones = labels_in_node.apply(lambda x: x==-1).sum()
    
#错误数量的给出是基于major class prediction的，一个集合里有1有-1,如果做major class prediction以后，minor class的数量就是错误的数量    
    return min(positive_ones, negative_ones) 



def best_split(data, features, target):
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
        error = (left_misses + right_misses) * 1.0 / num_data_points

        # 更新应选特征和错误率，注意错误越低说明该特征越好
        if error < best_error:
            best_error = error
            best_feature = feature
    return best_feature
class TreeNode:
    def __init__(self, is_leaf, prediction, split_feature):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.split_feature = split_feature
        self.left = None
        self.right = None
from sklearn.base import BaseEstimator #模型中的父类
from sklearn.metrics import accuracy_score
class MyDecisionTree(BaseEstimator):
    
    def __init__(self, max_depth, min_error):
        self.max_depth = max_depth
        self.min_error = min_error
    
    def fit(self, X, Y, data_weights = None):
        
        data_set = pd.concat([X, Y], axis=1) #concatenate pandas objects along a particular axis
        features = X.columns
        target = Y.columns[0]#Y其实只有一列
        self.root_node = self.create_tree(data_set, features, target, current_depth=0, max_depth=self.max_depth, min_error=self.min_error)
    
    def create_tree(self, data, features, target, current_depth = 0, max_depth = 10, min_error=0):
        """
        探索三种不同的终止划分数据集的条件  
  
        termination 1, 当错误率降到min_error以下, 终止划分并返回叶子节点  
        termination 2, 当特征都用完了, 终止划分并返回叶子节点  
        termination 3, 当树的深度等于最大max_depth时, 终止划分并返回叶子节点
        """
        
    
        # 拷贝一下可用特征
        remaining_features = features[:]

        target_values = data[target]

        
        
        #################
        #  第一部分： 递归出口！
        #################
        # termination 1   bonus task
        if count_errors(target_values) <= min_error:
            print("Termination 1 reached.")     
            return self.create_leaf(target_values) 

        # termination 2
        if len(remaining_features) == 0:
            print("Termination 2 reached.")    
            return self.create_leaf(target_values) #返回叶子节点

        # termination 3
        if current_depth >= max_depth: 
            print("Termination 3 reached.")
            return self.create_leaf(target_values)


        
        
        #################
        #  第二部分： 如果继续划分，划分数据集！
        #################
        
        # 选出最佳当前划分特征
        split_feature = best_split(data, features, target)   #根据正确率划分   bonus task
        #split_feature = best_split_entropy(data, features, target)  # 根据信息增益来划分

        # 选出最佳特征后，该特征为0的数据分到左边，该特征为1的数据分到右边
        left_split = data[data[split_feature]==0]
        right_split = data[data[split_feature]==1]

        # 剔除已经用过的特征
        remaining_features = remaining_features.drop(split_feature)
        print("Split on feature %s. (%s, %s)" % (split_feature, str(len(left_split)), str(len(right_split))))

        # 如果当前数据全部划分到了一边，就不需要再划分了，直接创建叶子节点返回即可
        if len(left_split) == len(data):
            print("Perfect split!")
            return self.create_leaf(left_split[target])
        if len(right_split) == len(data):
            print("Perfect split!")
            return self.create_leaf(right_split[target])

        # 递归上面的步骤
        left_tree = self.create_tree(left_split, remaining_features, target, current_depth + 1, max_depth, min_error)     
        right_tree = self.create_tree(right_split, remaining_features, target, current_depth + 1, max_depth, min_error)
        
        
        #################
        #  第三部分： 左右递归完之后，把当前结果返回！
        #################
        
        #生成当前的树节点
        result_node = TreeNode(False, None, split_feature)#False代表它不是叶子节点
        result_node.left = left_tree
        result_node.right = right_tree
        return result_node    
    
    
    
    def create_leaf(self, target_values):
        # 用于创建叶子的函数

        # 初始化一个树节点
        leaf = TreeNode(True, None, None)

        # 统计当前数据集里标签为+1和-1的个数，较大的那个即为当前节点的预测结果
        num_positive_ones = len(target_values[target_values == +1])
        num_negative_ones = len(target_values[target_values == -1])

        if num_positive_ones > num_negative_ones:
            leaf.prediction = 1
        else:
            leaf.prediction = -1

        # 返回叶子        
        return leaf     
    
    
        
    def predict(self, X):
        #某一行拿出来，给定结果，把结果应用到所有行
        prediction = X.apply(lambda row: self.predict_single_data(self.root_node, row), axis=1)
        return prediction
    
    
    
    def predict_single_data(self, tree, X, annotate = False):   
        # 如果已经是叶子节点直接返回叶子节点的预测结果
        if tree.is_leaf:
            if annotate: 
                print("leaf node, predicting %s" % tree.prediction)
            return tree.prediction
        else:
            # 查询x对应当前节点特征的值
            split_feature_value = X[tree.split_feature]

            if annotate: 
                print("Split on %s = %s" % (tree.split_feature, split_feature_value))
            if split_feature_value == 0:
                #如果x在该特征上的值为0，交给左子树来预测
                return self.predict_single_data(tree.left, X, annotate) #annotate---是否打印中间信息
            else:
                #如果x在该特征上的值为1，交给右子树来预测
                return self.predict_single_data(tree.right, X, annotate)
    
    
    
        
    def score(self, testX, testY):
        target = testY.columns[0]
        result = self.predict(testX)
        return accuracy_score(testY[target], result)


    def count_leaves(self):
        return self.count_leaves_helper(self.root_node)
    
    def count_leaves_helper(self, tree):
        if tree.is_leaf:
            return 1
        return self.count_leaves_helper(tree.left) + self.count_leaves_helper(tree.right)
m = MyDecisionTree(max_depth = 10, min_error = 1e-15)
m.fit(trainX, trainY)
m.score(testX, testY)
m.count_leaves()
model_4 = MyDecisionTree(max_depth=3, min_error = 1e-15)
model_5 = MyDecisionTree(max_depth=7, min_error = 1e-15)
model_6 = MyDecisionTree(max_depth=15, min_error = 1e-15)
model_4.fit(trainX, trainY)
model_5.fit(trainX, trainY)
model_6.fit(trainX, trainY)
print("model_4 training accuracy :", model_4.score(trainX, trainY))
print("model_5 training accuracy :", model_5.score(trainX, trainY))
print("model_6 training accuracy :", model_6.score(trainX, trainY))
print("model_4 testing accuracy :", model_4.score(testX, testY))
print("model_5 testing accuracy :", model_5.score(testX, testY))
print("model_6 testing accuracy :", model_6.score(testX, testY))
print("model_4 complexity is: ", model_4.count_leaves())
print("model_5 complexity is: ", model_5.count_leaves())
print("model_6 complexity is: ", model_6.count_leaves())