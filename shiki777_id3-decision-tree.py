import numpy as np

import time

import copy

from collections import Counter
def loadData(filename):

    fr = open(filename,'r')

    x,y = [],[]

    for line in fr.readlines():

        curline = line.strip().split(',')

        x.append([int(int(num) >= 128) for num in curline[1:]])

        y.append(int(curline[0]))

    x = np.array(x)

    y = np.array(y)

    return x,y
class Node:

    def __init__(self,label,dims,leaves):

        self.label = label

        self.dims = dims

        self.leaves = leaves
def major_label(labels):

    return Counter(labels).most_common(1)[0][0]
def cal_entropy(labels):

    entropy = 0

    labelset = set(labels)

    for i in labelset:

        entropy -= (np.sum(labels==i)/labels.shape[0]) * np.log2(np.sum(labels==i)/labels.shape[0])

    return entropy
def cal_conditional_entropy(features,labels):

    conditional_entropy = 0

    labelset = set(features)

    for i in labelset:

        conditional_entropy += np.sum(features == i) / labels.shape[0] * cal_entropy(labels[features == i])

    return conditional_entropy
def get_best_feature(x_train,y_train,features_left):

    best_feature = -1

    max_gain = float("-inf")

    for i in features_left:

        gain = cal_entropy(y_train)

        gain -= cal_conditional_entropy(x_train[:,i],y_train)

        if max_gain < gain:

            max_gain = gain

            best_feature = i

    return best_feature,max_gain
def build_Tree(x_train,y_train,features_left,threshold):

    if len(set(y_train)) == 1: return Node(y_train[0],-1,None)

    if not features_left: 

        return Node(major_label(y_train),-1,None)

    best_feature,max_gain = get_best_feature(x_train,y_train,features_left)

    if max_gain < threshold: 

        return Node(major_label(y_train),-1,None)

    features = set(x_train[:,best_feature])

    leaves = {}

    features_copy = copy.deepcopy(features_left)

    features_copy.remove(best_feature)

    for feature in features:

        leaves[feature] = build_Tree(x_train[x_train[:,best_feature] == feature],y_train[x_train[:,best_feature] == feature] \

                      ,features_copy,threshold)

    return Node(-1,best_feature,leaves)
def predict(x,root):

    cur_node = root

    while cur_node.dims != -1:

        cur_node = cur_node.leaves[x[cur_node.dims]]

    return cur_node.label
def test(x_val,y_val,root):

    size = x_val.shape[0]

    correct = 0

    for i in range(size):

        label = predict(x_val[i],root)

        if label == y_val[i]: correct += 1

    return correct / size
x_train,y_train = loadData('/kaggle/input/mnist-percetron/mnist_train.csv')

x_val,y_val = loadData('/kaggle/input/mnist-percetron/mnist_test.csv')
start = time.time()

root = build_Tree(x_train,y_train,[i for i in range(784)],0.05)

print("Building the decision tree consumes {:.2f} .".format(time.time() - start))
acc_train = test(x_train,y_train,root)

print("Accuracy of the train dataset is {:.2f}".format(acc_train))

acc_val = test(x_val,y_val,root)

print("Accuracy of the val dataset is {:.2f}".format(acc_val))