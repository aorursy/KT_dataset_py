import pandas as pd

import math

from PIL import Image

import matplotlib.pyplot as plt
#创建结点，包含划分的属性以及阈值

class Node:

    def __init__(self,data_index,split_feature=None,split_feval=None,is_leaf=False,loss=None,deep=None):

        self.loss = loss

        self.split_feature = split_feature

        self.split_feval = split_feval

        self.is_leaf = is_leaf

        self.deep = deep

        self.predict_value = None

        self.data_index = data_index

        self.left_child = None

        self.right_child = None

    

    def updata_predict_value(self,targets):

        self.predict_value = targets.mean() #叶子结点的均值就是预测值

    

    def get_predict_value(self,instance):

        if self.is_leaf:

            return self.predict_value

        if instance[self.split_feature]<self.split_feval:

            return self.left_child.get_predict_value(instance)

        else:

            return self.right_child.get_predict_value(instance)

        

#建树

class Tree:

    def __init__(self,data,max_depth,features,target_name,loss):

        self.loss = loss

        self.max_depth = max_depth

        self.features = features

        self.target_name = target_name

        self.leaf_nodes = []

        self.remain_index = [True]*len(data)

        self.root_node = self.build_tree(data,self.remain_index,depth=0)

    

    def build_tree(self,data,remain_index,depth=0):

        nowdata = data[remain_index]

        

        if depth < self.max_depth-1 and len(data[self.target_name].unique()) > 1 and len(data[self.target_name]) > 1: #判断特征划分是否结束

            '''

            1.是否已经达到了设置的最大深度

            2.叶子结点的元素是否全部相同

            3.叶子结点是否只有一个元素

            '''

            

            se = None #计算平方损失

            split_feature = None

            split_feval = None

            lefttree_index_of_nowdata = None

            righttree_index_of_nowdata = None

            

            for feature in self.features:

                feature_values = nowdata[feature].unique()

                for feval in feature_values:

                    lefttree_index = list(nowdata[feature]<feval)

                    righttree_index = list(nowdata[feature]>=feval)

                    lefttree_se = calculate_se(nowdata[lefttree_index][self.target_name])

                    righttree_se = calculate_se(nowdata[righttree_index][self.target_name])

                    sum_se = lefttree_se+righttree_se

                    if se is None or sum_se<se:

                        split_feature = feature

                        split_feval = feval

                        se = sum_se

                        lefttree_index_of_nowdata = lefttree_index

                        righttree_index_of_nowdata = righttree_index

            

            node = Node(remain_index,split_feature,split_feval,is_leaf=False,loss=self.loss,deep=depth)

            #记录划分后的样本在原始数据中的索引

            left_index_of_all_data = []

            for i in remain_index:

                if i:

                    if lefttree_index_of_nowdata[0]:

                        left_index_of_all_data.append(True)

                        del lefttree_index_of_nowdata[0]

                    else:

                        left_index_of_all_data.append(False)

                        del lefttree_index_of_nowdata[0]

                else:

                    left_index_of_all_data.append(False)



            right_index_of_all_data = []

            for i in remain_index:

                if i:

                    if righttree_index_of_nowdata[0]:

                        right_index_of_all_data.append(True)

                        del righttree_index_of_nowdata[0]

                    else:

                        right_index_of_all_data.append(False)

                        del righttree_index_of_nowdata[0]

                else:

                    right_index_of_all_data.append(False)

            

            node.left_child = self.build_tree(data,left_index_of_all_data,depth+1)

            node.right_child = self.build_tree(data,right_index_of_all_data,depth+1)

            return node

        else:

            node = Node(remain_index,is_leaf=True,loss=self.loss,deep=depth)

            label_name = 'label'

            #更新结点的预测值

            node.updata_predict_value(nowdata[self.target_name])

            self.leaf_nodes.append(node)

            return node



def calculate_se(label):

    mean = label.mean()

    se = 0

    for y in label:

        se += (y - mean) **2

    return se

            

        
import abc



class AbstractBaseGradientBoosting(metaclass = abc.ABCMeta):

    def __init__(self):

        pass

    def fit(self,data):

        pass

    def predict(self,data):

        pass



class BaseGradientBoosting(AbstractBaseGradientBoosting):

    def __init__(self,loss,learning_rate=1,n_trees=5,max_depth=3):

        super().__init__()

        self.loss = loss

        self.learning_rate = learning_rate

        self.n_trees = n_trees

        self.max_depth = max_depth

        self.trees = {}

        self.features = None

        self.f_0 = {}

        

    def fit(self,data):

        self.features = list(data.columns)[1:-1]

        #初始化f_0（x）

        self.f_0 = self.loss.initialize_f_0(data)

        #生成多个回归树

        for iter in range(1,self.n_trees+1):

            print('--------------构建第%d棵树----------------'%iter)

            self.loss.calculate_gradient(data,iter)

            target_name = 'res_'+str(iter)

            self.trees[iter] = Tree(data,self.max_depth,self.features,target_name,self.loss)

            #更新学习器

            self.loss.update_f_m(data, self.trees, iter, self.learning_rate)

            

            plot_tree(self.trees[iter], self.max_depth,iter)

            
class GradientBoostingReg(BaseGradientBoosting):

    def __int__(self,loss,learning_rate,n_trees,max_depth):

        super().__init__(loss,learning_rate=1,n_trees=5,max_depth=3)

    

    def predict(self,data):

        data['f_0'] = self.f_0

        for iter in range(1,self.n_trees+1):

            f_prev_name = 'f_' + str(iter-1)

            f_m_name = 'f_' + str(iter)

            data[f_m_name] = data[f_prev_name] + self.learning_rate*data.apply(lambda x:self.trees[iter].root_node.get_predict_value(x),axis=1)

        data['predict_value']=data[f_m_name]

        print(data['predict_value'])

        

class LossFunction(metaclass=abc.ABCMeta):



    @abc.abstractmethod

    def initialize_f_0(self, data):

        """初始化 F_0 """



    @abc.abstractmethod

    def calculate_gradient(self, data, iter):

        """计算负梯度"""



    @abc.abstractmethod

    def update_f_m(self, data, trees, iter, learning_rate, logger):

        """计算 F_m """



#     @abc.abstractmethod

#     def update_leaf_values(self, targets, y):

#         """更新叶子节点的预测值"""



    @abc.abstractmethod

    def get_train_loss(self, y, f, iter, logger):

        """计算训练损失"""



class SquaresError(LossFunction):

    

    def initialize_f_0(self, data):

        data['f_0'] = data['label'].mean()

        return data['f_0']

    

    def calculate_gradient(self, data, iter):

        res_name = 'res_' + str(iter)

        f_prev_name = 'f_' + str(iter - 1)

        data[res_name] = data['label'] - data[f_prev_name] #平方损失的梯度

    

    def update_f_m(self, data, trees, iter, learning_rate):

        f_prev_name = 'f_' + str(iter - 1)

        f_m_name = 'f_' + str(iter)

        data[f_m_name] = data[f_prev_name]

        

        for leaf_node in trees[iter].leaf_nodes:

            data.loc[leaf_node.data_index, f_m_name] += learning_rate * leaf_node.predict_value

        # 打印每棵树的 train loss

        self.get_train_loss(data['label'], data[f_m_name], iter) 

        print(data)

    

    def get_train_loss(self, y, f, iter):

        loss = ((y - f) ** 2).mean()

        print('第%d棵树: mse_loss:%.4f' % (iter, loss))

    

decisionNode = dict(boxstyle="sawtooth", fc="0.8")

leafNode = dict(boxstyle="round4", fc="0.8")

arrow_args = dict(arrowstyle="<-")

#上面三行代码定义文本框和箭头格式

#定义决策树决策结果的属性，用字典来定义，也可写作 decisionNode={boxstyle:'sawtooth',fc:'0.8'}

#其中boxstyle表示文本框类型，sawtooth是波浪型的，fc指的是注释框颜色的深度

#arrowstyle表示箭头的样式



def plotNode(nodeTxt, centerPt, parentPt, nodeType):#该函数执行了实际的绘图功能

#nodeTxt指要显示的文本，centerPt指的是文本中心点，parentPt指向文本中心的点

    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',

             xytext=centerPt, textcoords='axes fraction',

             va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )





#获取叶节点的数目

def getNumLeafs(myTree):

    numLeafs=0

    firstStr=list(myTree.keys())[0]#字典的第一个键，也就是树的第一个节点

    secondDict=myTree[firstStr]#这个键所对应的值，即该节点的所有子树。

    for key in secondDict.keys():

        if type(secondDict[key]).__name__=='dict':#测试节点的数据类型是否为字典

            numLeafs+=getNumLeafs(secondDict[key])#递归,如果是字典的话，继续遍历

        else:numLeafs+=1#如果不是字典型的话，说明是叶节点，则叶节点的数目加1

    return numLeafs

#获取树的层数

def getTreeDepth(myTree):#和上面的函数结果几乎一致

    maxDepth=0

    firstStr=list(myTree.keys())[0]

    secondDict=myTree[firstStr]

    for key in secondDict.keys():

        if type(secondDict[key]).__name__ == 'dict':

            thisDepth=1+getTreeDepth(secondDict[key])#递归

        else:thisDepth=1#一旦到达叶子节点将从递归调用中返回，并将计算深度加1

        if thisDepth>maxDepth:maxDepth=thisDepth

    return maxDepth



#可视化

def plotMidText(cntrPt,parentPt,txtString):#计算父节点和子节点的中间位置，并在父子节点间填充文本信息

    # cntrPt文本中心点   parentPt 指向文本中心的点

    xMid=(parentPt[0]-cntrPt[0])/2.0+cntrPt[0]

    yMid=(parentPt[1]-cntrPt[1])/2.0+cntrPt[1]

    createPlot.ax1.text(xMid,yMid,txtString)



def plotTree(myTree,parentPt,nodeTxt):

    numLeafs=getNumLeafs(myTree)#调用getNumLeafs（）函数计算叶子节点数目（宽度）

    depth=getTreeDepth(myTree)#调用getTreeDepth（），计算树的层数（深度）

    firstStr=list(myTree.keys())[0]

    cntrPt=(plotTree.xOff+(1.0+float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)#plotTree.totalW表示树的深度

    plotMidText(cntrPt,parentPt,nodeTxt)#调用 plotMidText（）函数，填充信息nodeTxt

    plotNode(firstStr,cntrPt,parentPt,decisionNode)#调用plotNode（）函数，绘制带箭头的注解

    secondDict=myTree[firstStr]

    plotTree.yOff=plotTree.yOff-1.0/plotTree.totalD

    #因从上往下画，所以需要依次递减y的坐标值，plotTree.totalD表示存储树的深度

    for key in secondDict.keys():

        if type(secondDict[key]).__name__=='dict':

            plotTree(secondDict[key],cntrPt,str(key))#递归

        else:

            plotTree.xOff=plotTree.xOff+1.0/plotTree.totalW

            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)

            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))

    plotTree.yOff=plotTree.yOff+1.0/plotTree.totalD#h绘制完所有子节点后，增加全局变量Y的偏移。



def createPlot(inTree):

    fig=plt.figure(1,facecolor='white')#绘图区域为白色

    fig.clf()#清空绘图区

    axprops = dict(xticks=[], yticks=[])#定义横纵坐标轴

    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)

    #由全局变量createPlot.ax1定义一个绘图区，111表示一行一列的第一个，frameon表示边框,**axprops不显示刻度

    plotTree.totalW=float(getNumLeafs(inTree))

    plotTree.totalD=float(getTreeDepth(inTree))

    plotTree.xOff=-0.5/plotTree.totalW;plotTree.yOff=1.0;

    plotTree(inTree,(0.5,1.0),'')

    plt.show()



#构建两棵树，用来测试

def retrieveTree(i):

    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}},3: 'maybe'}},

                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}

                  ]

    return listOfTrees[i]



# createPlot(retrieveTree(0))







def plot_tree(tree: Tree, max_depth: int, iter: int):

    root = tree.root_node

    tree_dict = {}

    tree_to_dict(root,tree_dict)

    print(tree_dict)

    createPlot(tree_dict)

    

def tree_to_dict(tree_root,tree_dict):

    # 如果节点没有子节点，递归结束

    if tree_root:

        tree_dict[tree_root.split_feature] = {}

    else:

        return

    if not tree_root.left_child and not tree_root.right_child:

        return

    #下面是核心代码

    #如果有子节点，在对子节点进行操作

    if tree_root.left_child:

        # 如果tree_dict没有对应的节点地址键，

        #child.data的data对应的树节点的地址，比如河北省，北京市之类的，

        #那就给字典赋键值对，键就是data，值对应空字典

        if tree_root.left_child.is_leaf:

            tree_dict[tree_root.split_feature][tree_root.split_feature+'<'+str(tree_root.split_feval)] = {round(tree_root.left_child.predict_value,3)}

        else:

            if not tree_dict.get(tree_root.left_child.split_feature+'<'+str(tree_root.left_child.split_feval)):

                tree_dict[tree_root.split_feature][tree_root.left_child.split_feature+'<'+str(tree_root.left_child.split_feval)] = {}

                # 继续对child递归，这里的关键是tree_dict要传入tree_dict[child.data]，

                #也就是新的空字典，思想上就是不断的给字典赋值，赋的值仍然是字典，直至结束

                tree_to_dict(tree_root.left_child, tree_dict[tree_root.split_feature][tree_root.left_child.split_feature+'<'+str(tree_root.left_child.split_feval)])

            else:

                #如果tree_dict有对应的节点地址键，直接继续递归

                tree_to_dict(tree_root.left_child, tree_dict[tree_root.split_feature][tree_root.left_child.split_feature+'<'+str(tree_root.left_child.split_fevale)])

            



    if tree_root.right_child:

        # 如果tree_dict没有对应的节点地址键，

        #child.data的data对应的树节点的地址，比如河北省，北京市之类的，

        #那就给字典赋键值对，键就是data，值对应空字典

        if tree_root.right_child.is_leaf:

            tree_dict[tree_root.split_feature][tree_root.split_feature+'>='+str(tree_root.split_feval)] = {round(tree_root.right_child.predict_value,3)}

        else:

            if not tree_dict.get(tree_root.right_child.split_feature+'>='+str(tree_root.right_child.split_feval)):

                tree_dict[tree_root.split_feature][tree_root.right_child.split_feature+'>='+str(tree_root.right_child.split_feval)] = {}

                # 继续对child递归，这里的关键是tree_dict要传入tree_dict[child.data]，

                #也就是新的空字典，思想上就是不断的给字典赋值，赋的值仍然是字典，直至结束

                tree_to_dict(tree_root.right_child, tree_dict[tree_root.split_feature][tree_root.right_child.split_feature+'>='+str(tree_root.right_child.split_feval)])

            else:

                #如果tree_dict有对应的节点地址键，直接继续递归

                tree_to_dict(tree_root.right_child, tree_dict[tree_root.split_feature][tree_root.right_child.split_feature+'>='+str(tree_root.right_child.split_feval)])

    
data = {}

data['train'] = pd.DataFrame(data=[[1, 5, 20, 1.1],

                          [2, 7, 30, 1.3],

                          [3, 21, 70, 1.7],

                          [4, 30, 60, 1.8],], columns=['id', 'age', 'weight', 'label'])

data['test'] = pd.DataFrame(data = [[5,25,65],],columns=['id', 'age', 'weight',])

print(data['train'])

print(data['test'])
#定义模型

model = GradientBoostingReg(SquaresError(),learning_rate=0.1, n_trees=5, max_depth=3)

#训练模型

model.fit(data['train'])

#预测

model.predict(data['test'])