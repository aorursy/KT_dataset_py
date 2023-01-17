# necessary imports
%matplotlib inline
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import cross_val_score

np.random.seed(19)
data_folder = "../input"
#data_folder = "./data"
data = pd.read_csv(os.path.join(data_folder, "mushrooms.csv"))
#data = pd.read_csv('./mushrooms.csv')
data['class'] = data.apply(lambda row: -1 if row[0] == 'e' else 1, axis=1)
def dummies(data, columns=['pclass','name_title','embarked', 'sex']):
    for col in columns:
        data[col] = data[col].apply(lambda x: str(x))
        new_cols = [col + '_' + i for i in data[col].unique()]
        data = pd.concat([data, pd.get_dummies(data[col], prefix=col)[new_cols]], axis=1)
        del data[col]
    return data
target = 'class'
cols = data.columns.drop(target)
data_set = dummies(data, columns = cols)

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data_set, test_size=0.3)
trainX, trainY = train_data[train_data.columns[1:]], pd.DataFrame(train_data[target])
testX, testY = test_data[test_data.columns[1:]], pd.DataFrame(test_data[target])


class TreeNode:
    def __init__(self, is_leaf, prediction, split_feature):
        self.is_leaf = is_leaf
        self.prediction = prediction
        self.split_feature = split_feature
        self.left = None
        self.right = None
def node_weighted_mistakes(targets_in_node, data_weights):
    # 计算lable 为+1的所有数据的权重和
    weight_positive = sum(data_weights[targets_in_node == 1]) # TODO
    
    # 如果全部预测为-1，那么预测错误的数据权重等于weight_positive
    #TODO
    weighted_mistakes_negative = weight_positive# TODO 
    
    # 计算lable 为-1的所有数据的权重和
    #TODO
    weight_negative = sum(data_weights[targets_in_node == -1])# TODO
    
    # 如果全部预测为+1，那么预测错误的数据权重等于weight_negative
    #TODO
    weighted_mistakes_positive = weight_negative# TODO
    
    #将加权错误和对应的预测标签一起输出
    #TODO
    if weighted_mistakes_negative < weighted_mistakes_positive:
        return(weighted_mistakes_negative, -1)
    else:
        return(weighted_mistakes_positive, 1)
#test
example_targets = np.array([-1, -1, 1, 1, 1])
example_data_weights = np.array([1., 2., 0.5, 1., 1.])
node_weighted_mistakes(example_targets, example_data_weights)

def best_split_weighted(data, features, target, data_weights):
    # return the best feature
    best_feature = None
    best_error = float("inf")
    num_data_points = float(len(data))  

    for feature in features:
        
        # 左分支对应当前特征为0的数据点
        left_split = data[data[feature] == 0]
        
        # 进入左分支数据点的权重
        left_data_weights = data_weights[data[feature] == 0]
        
        
        # 右分支对应当前特征为1的数据点
        right_split = data[data[feature] == 1] 
        
        # 进入右分支数据点的权重
        right_data_weights = data_weights[data[feature] == 1]
        
        # 重点！！
        # 计算左边分支里犯了多少错 (加权结果！！)
        left_misses, left_class = node_weighted_mistakes(left_split[target], left_data_weights)            

        # 计算右边分支里犯了多少错 (加权结果！！)
        right_misses, right_class = node_weighted_mistakes(right_split[target], right_data_weights)
            
        # 计算当前划分之后的分类犯错率
        error = (left_misses + right_misses) * 1.0 / sum(data_weights)

        # 更新应选特征和错误率，注意错误越低说明该特征越好
        if error < best_error:
            best_error = error
            best_feature = feature
    return best_feature
# test
# 根据之前的实现，最佳特征
features = data_set.columns.drop(target)
example_data_weights = np.array(len(train_data) * [1])
best_split_weighted(train_data, features, target, example_data_weights)




def create_leaf(target_values, data_weights):
    # 用于创建叶子的函数
    
    # 初始化一个树(叶子)节点
    leaf = TreeNode(True, None, None)
    
    # 直接调用node_weighted_mistakes得到叶子节点的预测结果
    weighted_error, prediction_class = node_weighted_mistakes(target_values, data_weights)# TODO
    
    leaf.prediction = prediction_class# TODO
        
    # 返回叶子        
    return leaf 
def create_weighted_tree(data, data_weights, features, target, current_depth = 0, max_depth = 10, min_error=1e-15):
    # 拷贝以下可用特征
    remaining_features = features[:]
    
    target_values = data[target]
    
    # termination 1
    if node_weighted_mistakes(target_values,data_weights)[0] <= min_error:
        print("Termination 1 reached.")     
        return create_leaf(target_values, data_weights)
    
    # termination 2
    if len(remaining_features) == 0:
        print("Termination 2 reached.")    
        return create_leaf(target_values, data_weights)    
    
    # termination 3
    if current_depth >= max_depth: 
        print("Termination 3 reached.")
        return create_leaf(target_values, data_weights)

    
    
    # 选出最佳当前划分特征
    split_feature = best_split_weighted(data, features, target, data_weights)  # 根据加权错误来划分
    
    # 选出最佳特征后，该特征为0的数据分到左边，该特征为1的数据分到右边
    left_split = data[data[split_feature] == 0]# TODO
    right_split = data[data[split_feature] ==1]# TODO
    
    # 将对应数据的权重也分到左边与右边
    left_data_weights = data_weights[data[split_feature] == 0]# TODO
    right_data_weights = data_weights[data[split_feature] == 1]# TODO
    
    
    # 剔除已经用过的特征
    remaining_features = remaining_features.drop(split_feature)
    print("Split on feature %s. (%s, %s)" % (split_feature, str(len(left_split)), str(len(right_split))))
    
    # 如果当前数据全部划分到了一边，直接创建叶子节点返回即可
    if len(left_split) == len(data):
        print("Perfect split!")
        return create_leaf(left_split[target],left_data_weights)
    if len(right_split) == len(data):
        print("Perfect split!")
        return create_leaf(right_split[target], right_data_weights)
        
    # 递归上面的步骤
    left_tree = create_weighted_tree(left_split, left_data_weights, remaining_features, target, current_depth + 1, max_depth, min_error)# TODO       
    right_tree = create_weighted_tree(right_split, right_data_weights, remaining_features, target, current_depth + 1, max_depth, min_error)# TODO
    
    
    #生成当前的树节点
    result_node = TreeNode(False, None, split_feature)
    result_node.left = left_tree
    result_node.right = right_tree
    return result_node
def count_leaves(tree):
    if tree.is_leaf:
        return 1
    return count_leaves(tree.left) + count_leaves(tree.right)
# test
example_data_weights =np.array([1.0 for i in range(len(train_data))])
small_data_decision_tree = create_weighted_tree(train_data,example_data_weights, features, target,max_depth=2)
count_leaves(small_data_decision_tree)
def predict_single_data(tree, x, annotate = False):   
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
            return predict_single_data(tree.left, x, annotate)
        else:
            #如果数据在该特征上的值不为0，交给右子树来预测
            return predict_single_data(tree.right, x, annotate)
def evaluate_accuracy(tree, data):
    # 将predict函数应用在数据data的每一行
    prediction = data.apply(lambda row: predict_single_data(tree, row), axis=1)
    # 返回正确率
    accuracy = (prediction == data[target]).sum() * 1.0 / len(data)
    return accuracy
# test 根据测试样例，输出应该至少是0.95以上
evaluate_accuracy(small_data_decision_tree, test_data)



from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
class WeightedDecisionTree(BaseEstimator):
    
    def __init__(self, max_depth, min_error):
        self.max_depth = max_depth# TODO
        self.min_error = min_error# TODO
    
    def fit(self, X, Y, data_weights = None):
        
        data_set = pd.concat([X, Y], axis=1)
        features = X.columns
        target = Y.columns[0]
        self.root_node = create_weighted_tree(data_set, data_weights, features, target, current_depth = 0, max_depth = self.max_depth, min_error = self.min_error)# TODO
        
        
    def predict(self, X):
        prediction = X.apply(lambda row: predict_single_data(self.root_node, row), axis=1)
        return prediction
        
        
    def score(self, testX, testY):
        target = testY.columns[0]
        result = self.predict(testX)
        return accuracy_score(testY[target], result)    
from sklearn.base import BaseEstimator
class MyAdaboost(BaseEstimator):
    
    def __init__(self, M):
        self.M = M# TODO
        
    def fit(self, X, Y):
        self.models = []# TODO
        self.model_weights = []# TODO
        self.target = Y.columns[0]
        
        N, _ = X.shape
        alpha = np.ones(N)/N# TODO    # data weights
        
        for m in range(self.M):
            tree = WeightedDecisionTree(max_depth=2,min_error=1e-15)# TODO
            #TODO fit your current model
            tree.fit(X, Y, data_weights=alpha)
            prediction = tree.predict(X)# TODO
            
            # 计算加权错误
            weighted_error = alpha.dot(prediction != Y[self.target])
            
            # 计算当前模型的权重
            model_weight = 0.5 * (np.log(1 - weighted_error) - np.log(weighted_error))
            
            # 更新数据的权重
            alpha = alpha * np.exp(-model_weight * Y[self.target] * prediction)
            
            # 数据权重normalize
            alpha = alpha / alpha.sum()
            
            self.models.append(tree)
            self.model_weights.append(model_weight)
            
    def predict(self, X):
        N, _ = X.shape
        result = np.zeros(N)
        for wt, tree in zip(self.model_weights, self.models):
            result += wt * tree.predict(X)
        print(result)
        
        return np.sign(result)

    def score(self, testX, testY):
        result = self.predict(testX)
        return accuracy_score(testY[self.target], result) 
m = MyAdaboost(20)
m.fit(trainX, trainY)
m.score(testX, testY)


