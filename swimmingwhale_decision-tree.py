import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
watermelon = pd.read_csv('../input/watermelon2.0.csv')
watermelon
X = watermelon.iloc[:,1:-1]
print(X.columns.values)
X.head()
y = watermelon.iloc[:,-1]
y.head()
class Node:
    def __init__(self, data_x,data_y, father = None):
        print(data_x)
        self.data_x = data_x
        self.data_y = data_y
        self.y = None
        self.leaf = False
        self.father = father
        self.chirlds = []
        (m,n) = self.data_x.shape
        self.m = m
        self.n = n
        unique, counts = np.unique(data_y, return_counts=True)
        self.unique_y = unique
        self.counts_y = counts
        
    def addChirlds(self, chirlds):
        self.chirlds.append(chirlds)
        
    def is_same_class(self):
        '''
        判断该节点样本是否属于同一类别
        '''
        return (self.data_y == self.data_y[0]).all()
    
    def is_legitimate(self):
        '''
        判断该节点在不同类别上的数据是否都相同
        '''
        sample = self.data_x[self.data_y == self.unique_y[0]]
        bools = []

        for i in self.unique_y[1:]:
            class_data = self.data_x[self.data_y == i]
            if class_data.shape == sample.shape:
                bools.append((class_data == sample).all())
            else:
                bools.append(False)

        return np.array(bools).all()

    def turnLeaf(self):
        '''
        将该分支标记为叶节点,其类别标记为data中样本最多的类
        '''
        
        self.leaf = True
        self.y = self.unique_y[self.counts_y.argmax()]
        
    def calcEnt(self,data_y):
        '''
        计算信息熵
        '''
        unique, counts = np.unique(data_y, return_counts=True)
        ent = 0
        for key,value in dict(zip(unique, counts)).items():
            p = value/len(data_y)
            ent += p*np.log2(p)
        return -ent
    
    def calcGini(self,data_y):
        '''
        计算基尼系数
        '''
        unique, counts = np.unique(data_y, return_counts=True)
        gini = 0
        for key,value in dict(zip(unique, counts)).items():
            p = value/len(data_y)
            gini += p**2
        return 1-gini
    
    def selectAttribute(self,algorithm = 'ID3'):
        '''
        选择最优划分属性,支持ID3,C4.5,CART
        '''
        # 计算信息熵
        if algorithm == 'CART':
            ent = 0
        else:
            ent = self.calcEnt(self.data_y)
            
        gains = []
        gains_radios = []
        ginis = []
        indexs = []
        # 循环所有属性,计算每个属性的信息增益
        for feature in range(self.n):
            gain = ent
            gini = 0
            iv = 0
            # 计算第j个属性的所有取值
            feature_data = self.data_x[:,feature]
            feature_values = np.unique(feature_data)
            index = []
            # 循环每个属性可能的取值
            for feature_value in feature_values:
                feature_value_index = (feature_data == feature_value)
                index.append(feature_value_index)
                radio = np.sum(feature_value_index)/self.m
                if algorithm == 'CART':
                    gini += radio*self.calcGini(self.data_y[feature_value_index])
                else:
                    iv -= radio*np.log2(radio)
                    gain -= radio*self.calcEnt(self.data_y[feature_value_index])
                
            indexs.append(index)
            if algorithm == 'ID3':
                gains.append(gain)
            elif algorithm == 'C4.5':
                if iv == 0:
                    gains_radio = 0
                else:
                    gains_radio = gain/iv
                gains_radios.append(gains_radio)
                gains.append(gain)
            elif algorithm == 'CART':
                ginis.append(gini)
            
        if algorithm == 'ID3':
            gains_max = np.argmax(gains)
        elif algorithm == 'C4.5':
            # 先找信息增益高于平均水平的 ,然后再找增益率最大的
            gains_radios = np.array(gains_radios)
            mean = np.mean(gains)
            gains_radios[gains < mean] =0
            gains_max = np.argmax(gains_radios)
        elif algorithm == 'CART':
            gains_max = np.argmin(ginis)
        return gains_max,indexs[gains_max]

    
    def __repr__(self):
        return str(self.data)
    
def TreeGenerate(X,y,father = None,deep = 1,columns = None):

    if deep >100:
        return
    print(deep)
    
    # 生成节点node
    node = Node(X,y)
    # 判断类别是否为空或者
    
    # 判断节点中的样本是否属于同一类别
    if node.is_same_class():
        node.turnLeaf()
        return
    # 判断节点中类别是否为空,或者x在y上取相同的值
    if node.is_legitimate():
        node.turnLeaf()
        return
    
    # 选择最优划分属性
    gains_max,indexs = node.selectAttribute(algorithm = 'ID3')
    print('按照'+columns[gains_max]+'划分')
    for index in indexs:
        TreeGenerate(X[index],y[index],node,deep+1,columns=columns)
    
    return node
tree = TreeGenerate(X.values,y.values,columns = X.columns.values)