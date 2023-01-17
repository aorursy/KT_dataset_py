import numpy as np

import time

import heapq

from heapq import *
def loadData(filename):

    fr = open(filename,'r')

    

    x,y = [],[]

    

    for line in fr.readlines():

        curline = line.strip().split(',')

        

        x.append([int(num) for num in curline[1:]])

        y.append(int(curline[0]))

    x = np.array(x)

    y = np.array(y)

    return x,y
class maxheap:

    def __init__(self,length):

        self.heap = [(-np.inf,0,None)] * length

        

    def pushpop(self,tup):

        heapq.heappushpop(self.heap,tup)

        

    def get_nsmallest(self,k):

        return heapq.nlargest(self.heap,k)
class Node:

    def __init__(self,data,label,sp,left,right):

        self.data = data

        self.label = label

        self.sp = sp

        self.left = left

        self.right = right
class KD_Tree:

    

    def __init__(self,x_train,y_train):

        self.tup = list(zip(x_train,y_train))

        self.root = self.create_KD_Tree(self.tup,0)

        

    def create_KD_Tree(self,data,sp):

        if len(data) == 0: return None

        

        data = sorted(data,key = lambda x:x[0][sp])

        mid = len(data) // 2

        

        d = data[mid]

        return Node(d[0],d[1],sp,self.create_KD_Tree(data[:mid],sp + 1),self.create_KD_Tree(data[mid + 1:],sp + 1))

    

    def get_nearest(self,target,k):

        heap = maxheap(k)

        def visit(node):

            if node:

                dis = target[node.sp] - node.data[node.sp]

                visit(node.left if dis < 0 else node.right)

                

                curdistance = np.linalg.norm(target - node.data)

                

                heap.pushpop((-curdistance,id(node),node))



                if - heap.heap[0][0] > abs(dis):  visit(node.right if dis < 0 else node.left)

        visit(self.root)

        return [element[2].label for element in heap.heap]
def test(x_train,y_train,x_val,y_val,k):

    

    start = time.time()

    tree = KD_Tree(x_train,y_train)

    correct = 0

    

#     for i in range(x_val.shape[0]):

    for i in range(100):

        pred_labels = tree.get_nearest(x_val[i],k)

        pred_label = max(pred_labels,key = pred_labels.count)

        if pred_label == y_val[i]: correct += 1

        

    print("Time of training consumes:{:.2f} Accuracy is:{:.2f}".format(time.time() - start , correct / 100))

    return correct / x_val.shape[0]
x_train,y_train = loadData('/kaggle/input/mnist-percetron/mnist_train.csv')

x_val,y_val = loadData('/kaggle/input/mnist-percetron/mnist_test.csv')

acc = test(x_train,y_train,x_val,y_val,25)