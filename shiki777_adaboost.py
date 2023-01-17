# https://github.com/Sh1k17
import time

import numpy as np
def loadData(filename):

    fr = open(filename,'r')

    x,y = [],[]

    for line in fr.readlines():

        curline = line.strip().split(',')

        if int(curline[0]) in [0,1]:

            x.append([int(int(num) >= 128) for num in curline[1:]])

            if int(curline[0]) == 0:

                y.append(1)

            else:

                y.append(-1)

    x = np.array(x)

    y = np.array(y)

    return x,y
class Node:

    def __init__(self,feature,labels,rule,error_rate,div_point,alpha = 0):

        self.feature = feature

        self.labels = labels

        self.div_rule = rule

        self.error_rate = error_rate

        self.div_point = div_point

        self.alpha = alpha
class AdaBoost:

    def __init__(self,x_train,y_train,x_val,y_val,trees_num):

        self.x_train = x_train[:1000]

        self.y_train = y_train[:1000]

        self.x_val = x_val

        self.y_val = y_val

        self.m,self.n = self.x_train.shape

        self.trees_num = trees_num

    

    def get_error_rate(self,feature,D,div_point):

        labels = np.zeros(self.m)

        for idx in range(self.m):

            if self.x_train[idx,feature] > div_point:

                labels[idx] = 1

            else:

                labels[idx] = -1

        error = np.sum(D[labels != self.y_train])

        return labels,error

    

    def create_single_tree(self,D):

        best_error_rate = 1

        root = None

        for feature in range(self.n):

            for div_point in [-0.5,0.5,1.5]:

                labels,error_rate = self.get_error_rate(feature,D,div_point)

                if error_rate > 0.5:

                    error_rate = 1 - error_rate

                    if error_rate < best_error_rate:

                        best_error_rate = error_rate

                        labels = ~labels.astype(np.int) + np.ones(self.m)

                        root = Node(feature,labels,"lower_is_1",error_rate,div_point)

                else:

                    if error_rate < best_error_rate:

                        best_error_rate = error_rate

                        root = Node(feature,labels,"higher_is_1",error_rate,div_point)

        return root

    

    def create_trees(self,):

        trees = []

        train_pred = np.zeros(self.m)

        D = np.ones(self.m) / self.m

        for i in range(self.trees_num):

            start = time.time()

            root = self.create_single_tree(D)

            root.alpha = 1 / 2 * np.log((1 - root.error_rate) / root.error_rate) 

            D = np.multiply(D,np.exp(-1 * root.alpha * np.multiply(self.y_train,root.labels)))

            sum_ = np.sum(D)

            D = D / sum_

            trees.append(root)

            train_pred += root.alpha * root.labels

            error = np.sum(np.sign(train_pred) != self.y_train)

            if error == 0:

                print("Creating the tree costs {:.2f} seconds.The error rate is {:.4f}".format(time.time() - start,error / self.m))

                break

            print("Creating the tree costs {:.2f} seconds.The error rate is {:.4f}".format(time.time() - start,error / self.m))

        return trees

    

    def predict(self,x,trees):

        score = 0

        for i in range(len(trees)):

            tree = trees[i]

            if tree.div_rule == "higher_is_1":

                if x[tree.feature] > tree.div_point:

                    score += tree.alpha

                else:

                    score -= tree.alpha

            else:

                if x[tree.feature] < tree.div_point:

                    score += tree.alpha

                else:

                    score -= tree.alpha

        return np.sign(score)

    

    def test(self,x_val,y_val,trees):

        len_ = x_val.shape[0]

        correct = 0

        for i in range(len_):

            label = self.predict(x_val[i],trees)

            if label == y_val[i]: correct += 1

        return correct / len_
x_train,y_train = loadData('/kaggle/input/mnist-percetron/mnist_train.csv')

x_val,y_val = loadData('//kaggle/input/mnist-percetron/mnist_test.csv')
model = AdaBoost(x_train,y_train,x_val,y_val,40)

trees = model.create_trees()
acc_train = model.test(model.x_train,model.y_train,trees)

print("The accuracy of the train dataset is {:.4f}".format(acc_train))
acc_val = model.test(model.x_val,model.y_val,trees)

print("The accuracy of the train dataset is {:.4f}".format(acc_val))