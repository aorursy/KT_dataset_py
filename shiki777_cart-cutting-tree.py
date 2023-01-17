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
    def __init__(self,label,dims,left,right,labels):
        self.label = label
        self.dims = dims
        self.left = left
        self.right = right
        self.labels = labels
def major_label(labels):
    return Counter(labels).most_common(1)[0][0]
def cal_Gini(y_train):
    label_set = set(y_train)
    Gini = 1
    for label in label_set:
        Gini -= np.square(np.sum(y_train == label) / y_train.shape[0])
    return Gini
def cal_Feature_Gini(features,labels):
    features_set = set(features)
    Gini_feature = 0
    for feature in features_set:
        Gini_feature += np.sum(features == feature) / features.shape[0] * cal_Gini(labels[features == feature])
    return Gini_feature
def get_best_feature(x_train,y_train,features_left):
    best_Gini = float("inf")
    best_feature = -1
    for feature in features_left:
        features = x_train[:,feature]
        gini = cal_Feature_Gini(features,y_train)
        if gini < best_Gini:
            best_Gini = gini
            best_feature = feature
    return best_Gini,best_feature
def build_Tree(x_train,y_train,features_left,sample_threshold,geni_threshold):
    if x_train.shape[0] <= sample_threshold: return Node(major_label(y_train),None,None,None,y_train)
    if cal_Gini(y_train) <= geni_threshold: return Node(major_label(y_train),None,None,None,y_train)
    if not features_left: return Node(major_label(y_train),None,None,None,y_train)
    
    min_Gini = float("inf")
    dims = (-1,-1)
    flag = False
    for index in features_left:
        features_values = set(x_train[:,index])
        for value in features_values:
            gini = cal_Feature_Gini((x_train[:,index] == value).astype(np.int),y_train)
            if gini < min_Gini:
                min_Gini = gini
                dims = (index,value)
                flag = len(features_values) <= 2
    if flag:
        left_copy = copy.deepcopy(features_left)
        right_copy = copy.deepcopy(features_left)
        left_copy.remove(dims[0])
        right_copy.remove(dims[0])
        return Node(major_label(y_train),dims, \
                    build_Tree(x_train[x_train[:,dims[0]] == dims[1]],y_train[x_train[:,dims[0]] == dims[1]],left_copy,sample_threshold,geni_threshold) \
                    ,build_Tree(x_train[x_train[:,dims[0]] != dims[1]],y_train[x_train[:,dims[0]] != dims[1]],right_copy,sample_threshold,geni_threshold),y_train)
    else:
        left_copy = copy.deepcopy(features_left)
        right_copy = copy.deepcopy(features_left)
        left_copy.remove(dims[0])
        return Node(major_label(y_train),dims,\
                    build_Tree(x_train[x_train[:,dims[0]] == dims[1]],y_train[x_train[:,dims[0]] == dims[1]],left_copy,sample_threshold,geni_threshold) \
                    ,build_Tree(x_train[x_train[:,dims[0]] != dims[1]],y_train[x_train[:,dims[0]] != dims[1]],right_copy,sample_threshold,geni_threshold),y_train)          
def predict(x,root):
    cur_node = root
    while cur_node.left or cur_node.right:
        if x[cur_node.dims[0]] == cur_node.dims[1]:
            cur_node = cur_node.left
        else:
            cur_node = cur_node.right
    return cur_node.label
def test(x_val,y_val,root):
    size = x_val.shape[0]
    correct = 0
    for i in range(size):
        pred = predict(x_val[i],root)
        if pred == y_val[i]: correct += 1
    return correct/size
def get_tree_gini(root):
    if not root: return 0
    if not root.left and not root.right: return cal_Gini(root.labels) * root.labels.shape[0]
    return get_tree_gini(root.left) + get_tree_gini(root.right)
def get_nodes_num(root):
    if not root: return 0
    return 1 + get_nodes_num(root.left) + get_nodes_num(root.right)
def get_G_t(root):
    C_T = get_tree_gini(root)
    C_t = cal_Gini(root.labels) * root.labels.shape[0]
    return (C_t - C_T) / (get_nodes_num(root) - 1)
class cut_Tree:
    def __init__(self,root):
        self.k = 0
        self.T = root
        self.alpha = float("inf")
        self.cut_node = None
        self.trees = [root]
        
        
    def get_tree(self,root):
        self.post_order(root)
        self.pre_order(root)
        self.cut_node.label = major_label(self.cut_node.labels)
        self.cut_node.left = None
        self.cut_node.right = None
        return root
    
    def get_trees(self,root):
        old_root = root
#         while old_root and old_root.left and old_root.right:
        for i in range(1000):
            self.alpha = float("inf")
            self.cut_node = None
            new_root = copy.deepcopy(old_root)
            old_root = self.get_tree(new_root)
            self.trees.append(old_root)
#             print(get_nodes_num(old_root))
        
    def post_order(self,root):
        if not root: return
        self.post_order(root.left)
        self.post_order(root.right)
        g_t = get_G_t(root)
        self.alpha = min(self.alpha,g_t)
    
    def pre_order(self,root):
        if not root: return
        if get_G_t(root) == self.alpha: 
            self.cut_node = root
        self.pre_order(root.left)
        self.pre_order(root.right)
x_train,y_train = loadData('/kaggle/input/mnist-percetron/mnist_train.csv')
x_val,y_val = loadData('/kaggle/input/mnist-percetron/mnist_test.csv')
start = time.time()
print("Start to build the tree...")
root = build_Tree(x_train[:50000],y_train[:50000],[i for i in range(784)],10,0.01)
print("Building the tree costs {:.2f}".format(time.time() - start))
acc_train = test(x_train[:50000],y_train[:50000],root)
print("Accuracy of train dataset is {:.4f}".format(acc_train))
acc_validation = test(x_train[50000:],y_train[50000:],root)
print("Accuracy of val dataset is {:.4f}".format(acc_validation))
acc_test = test(x_val,y_val,root)
print("Accuracy of test dataset is {:.4f}".format(acc_test))
print("The tree contains {} nodes".format(get_nodes_num(root)))
cut = cut_Tree(root)
start_cut = time.time()
print("Start to cut tree...")
cut.get_trees(root)
print("Cutting the tree costs {:.2f}".format(time.time() - start_cut))
print("The last tree contains {} nodes".format(get_nodes_num(cut.trees[-1])))
acc_val = float("-inf")
tree_res = None
for tree in cut.trees:
    if acc_val < test(x_train[50000:],y_train[50000:],tree):
        acc_val = test(x_train[50000:],y_train[50000:],tree)
        tree_res = tree
print("The res tree contains {} nodes".format(get_nodes_num(tree_res)))
acc_train = test(x_train[:50000],y_train[:50000],tree_res)
print("Accuracy of train dataset is {:.4f}".format(acc_train))
acc_validation = test(x_train[50000:],y_train[50000:],tree_res)
print("Accuracy of val dataset is {:.4f}".format(acc_validation))
acc_test = test(x_val,y_val,tree_res)
print("Accuracy of test dataset is {:.4f}".format(acc_test))
