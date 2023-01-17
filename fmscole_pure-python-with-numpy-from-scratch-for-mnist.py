import numpy as np

from functools import reduce

import math

import pandas as pd

import time
class Conv2D(object):

    def __init__(self,output_channels, ksize=3, stride=1, method='VALID'):

        self.output_channels = output_channels

        self.stride = stride

        self.ksize = ksize

        self.method = method

    def OutShape(self,shape):

        self.input_shape=shape

        self.input_channels = shape[-1]

        weights_scale = math.sqrt(reduce(lambda x, y: x * y, shape) / self.output_channels)

        self.weights = np.random.standard_normal((self.ksize,self.ksize, self.input_channels, self.output_channels)) / weights_scale

        self.bias = np.random.standard_normal(self.output_channels) / weights_scale



        self.w_gradient = np.zeros(self.weights.shape)

        self.b_gradient = np.zeros(self.bias.shape)

        

        if (shape[1] - self.ksize) % self.stride != 0:

            print('input tensor width can\'t fit stride')

        if (shape[2] - self.ksize) % self.stride != 0:

            print('input tensor height can\'t fit stride')



        if self.method == 'VALID':

            return [shape[0], 

                    (shape[1] - self.ksize + 1) // self.stride, 

                    (shape[1] - self.ksize + 1) // self.stride,

                    self.output_channels]

        # self.method == 'SAME':

        return [shape[0], 

                shape[1]// self.stride, 

                shape[2]// self.stride, 

                self.output_channels]



        



    def forward(self, x):

        self.batchsize = x.shape[0]

        if self.method == 'SAME':

            x = np.pad(x, (

                (0, 0), (self.ksize // 2, self.ksize // 2), (self.ksize // 2, self.ksize // 2), (0, 0)),

                'constant', constant_values=0)

        self.col_image=self.split_by_strides(x)

        conv_out=np.tensordot(self.col_image,self.weights, axes=([3,4,5],[0,1,2]))

        return conv_out



    def backward(self, eta):

        self.eta = eta

        self.w_gradient=np.tensordot(self.col_image,self.eta,axes=([0,1,2],[0,1,2]))



        if self.method == 'VALID':

            pad_eta = np.pad(self.eta, (

                (0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0)),

                'constant', constant_values=0)



        if self.method == 'SAME':

            pad_eta = np.pad(self.eta, (

                (0, 0), (self.ksize // 2, self.ksize // 2), (self.ksize // 2, self.ksize // 2), (0, 0)),

                'constant', constant_values=0)



        pad_eta=self.split_by_strides(pad_eta)

        next_eta=np.tensordot(pad_eta,self.weights, axes=([3,4,5],[0,1,3]))

        return next_eta



    def gradient(self, alpha=0.00001, weight_decay=0.0004):

        self.weights -= alpha/self.batchsize * self.w_gradient

        self.bias -= alpha/self.batchsize * self.bias



        self.w_gradient = np.zeros(self.weights.shape)

        self.b_gradient = np.zeros(self.bias.shape)



    def split_by_strides(self, x):

        # 将数据按卷积步长划分为与卷积核相同大小的子集,当不能被步长整除时，不会发生越界，但是会有一部分信息数据不会被使用

        N, H, W, C = x.shape

        oh = (H - self.ksize) // self.stride + 1

        ow = (W - self.ksize) // self.stride + 1

        shape = (N, oh, ow, self.ksize, self.ksize, C)

        strides = (x.strides[0], x.strides[1] * self.stride, x.strides[2] * self.stride, *x.strides[1:])

        return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
class Dense(object):

    def __init__(self,output_num):

        self.output_num=output_num

    def OutShape(self,shape):

        self.input_len =reduce(lambda x, y: x * y, shape[1:])

        self.weights = np.random.normal(0.0, pow(self.input_len, -0.5), (self.input_len, self.output_num))

        self.bias = np.random.normal(0.0, pow(self.output_num, -0.5),self.output_num)

        self.w_gradient = np.zeros(self.weights.shape)

        self.b_gradient = np.zeros(self.bias.shape)



        return [shape[0],self.output_num]

        

    def forward(self, x):

        self.batchsize = x.shape[0]

        self.x = x.reshape(-1, self.input_len)

        output = np.dot(self.x, self.weights)+self.bias

        self.input_shape=x.shape

        return output



    def backward(self, eta):

        self.w_gradient=np.dot(self.x.T, eta)

        self.b_gradient=np.sum(eta,axis=0)

        next_eta = np.dot(eta, self.weights.T)

        next_eta = np.reshape(next_eta, self.input_shape)



        return next_eta



    def gradient(self, alpha, weight_decay=0.0004):

        alpha=alpha /self.batchsize

        self.weights -= alpha* self.w_gradient

        self.bias -= alpha * self.b_gradient



        self.w_gradient = np.zeros(self.weights.shape)

        self.b_gradient = np.zeros(self.bias.shape)
class MaxPooling(object):

    def __init__(self, size=2, **kwargs):

        '''

        size: Pooling的窗口大小，因为在使用中窗口大小与步长基本一致，所以简化为一个参数

        '''

        self.size = size

    def OutShape(self,shape):

        return [shape[0],shape[1]//self.size,shape[2]//self.size,shape[3]]



    def forward(self, x):

        # 首先将输入按照窗口大小划分为若干个子集

        #这个reshape方式非常精妙，把一个维度拆分为两个维度，并没有用滑动窗口的方式



        out = x.reshape(x.shape[0], x.shape[1]//self.size, self.size, x.shape[2]//self.size, self.size, x.shape[3])

        out = out.max(axis=(2, 4))

        #or: 上面的两行代码等价于滑动窗口方式:        

        # N, H, W, C = x.shape

        # oh = (H - self.size) // self.size + 1

        # ow = (W - self.size) // self.size + 1

        # reshape = (N, oh, ow, self.size, self.size, C)

        # strides = (x.strides[0], x.strides[1] * self.size, x.strides[2] * self.size, *x.strides[1:])

        # out = np.lib.stride_tricks.as_strided(x,shape=reshape,strides=strides)

        # out = out.max(axis=(3, 4))



        # 记录每个窗口中不是最大值的位置

        self.mask = out.repeat(self.size, axis=1).repeat(self.size, axis=2) != x

        return out

    def backward(self, eta):

        # 将上一层传入的梯度进行复制，使其shape扩充到forward中输入的大小

        eta = eta.repeat(self.size, axis=1).repeat(self.size, axis=2)

        # 将不是最大值的位置的梯度置为0

        eta[self.mask] = 0

        return eta


class Softmax(object):

    def OutShape(self,shape):

        return shape

    def forward(self, x):

        '''

        x.shape = (N, C)

        接收批量的输入，每个输入是一维向量

        计算公式为：

        a_{ij}=\frac{e^{x_{ij}}}{\sum_{j}^{C} e^{x_{ij}}}

        '''

        v = np.exp(x - x.max(axis=-1, keepdims=True))    

        return v / v.sum(axis=-1, keepdims=True)

    

    # 一般Softmax的反向传播和CrossEntropyLoss的放在一起

    #所以不需要定义backward

        

    def cal_loss(self, y,t):

        return y-t
class Relu(object):

    def OutShape(self,shape):

        return shape

    def forward(self, x):

        self.x = x

        return np.maximum(0, x)



    def backward(self, eta):

        eta[self.x<=0] = 0

        return eta

class BatchNormal:

    def __init__(self):

        self.gamma=1

        self.beta=0.0

        self.epsilon=1e-5

        self.mean=None

        self.var=None

        pass

    def OutShape(self,shape):

        return shape



    def forward(self,input,axis=3,momentum=0.95,training=True):

        '''

        axis: channel所在的维度,比如input为[batch,height,width,channel],则axis=3（或-1）。

        这样就是对整个batch的同一特征平面（feature）标准化。

        不是针对每个样本标准化.也不是对每个特征平面标准化，而是把整个batch的同一个特征平面放在一起标准化.

        在求和求平均值的时候，channel维度保留，其他三个维度坍缩为一个数，塌缩到channel上。

        '''

        if training:

            shape=list(input.shape)

            ax=list(np.arange(len(shape)))

            shape.pop(axis)

            ax.pop(axis)

            self.axis=tuple(ax)

            self.m=reduce(lambda x, y: x * y, shape)





            mu=np.mean(input,axis=self.axis,keepdims=True)

            self.xmu=input-mu

            var = np.var(input,axis=self.axis,keepdims=True)

            self.ivar=1/np.sqrt(var+self.epsilon)

            self.xhut=self.xmu*self.ivar



            if self.mean is None: self.mean=mu

            if self.var is None: self.var =var



            self.mean=self.mean*momentum+mu*(1-momentum)

            self.var = self.var * momentum + var * (1 - momentum)



            return self.gamma*self.xhut+self.beta

        else:

            return self.test(input=input)

    def test(self,input):

        xmu = input - self.mean

        ivar = 1 / np.sqrt(self.var + self.epsilon)

        xhut = xmu * ivar



        return self.gamma*xhut+self.beta



    def backward(self,dy,lr=0.09):

        '''

        lr:学习率

        '''

        dxhut=dy*self.gamma

        dx1=self.m*dxhut

        dx2=self.ivar**2*np.sum(dxhut*self.xmu,axis=self.axis,keepdims=True)*self.xmu

        dx3=np.sum(dxhut,axis=self.axis,keepdims=True)

        dx=self.ivar/self.m*(dx1-dx2-dx3)



        dbeta=np.sum(dy,axis=self.axis,keepdims=True)

        self.beta-=lr*dbeta #根据dy的符号情况，有的网络这里的+要改为-

        dgmama=np.sum(dy*self.xhut,axis=self.axis,keepdims=True)

        self.gamma-=lr*dgmama  #根据dy的符号情况，有的网络这里的+要改为-



        return dx
class Dropout(object):

    def __init__(self,  p):

        """

        A dropout regularization wrapper.



        During training, independently zeroes each element of the layer input

        with probability p and scales the activation by 1 / (1 - p) (to reflect

        the fact that on average only (1 - p) * N units are active on any

        training pass). At test time, does not adjust elements of the input at

        all (ie., simply computes the identity function).



        Parameters

        ----------

        wrapped_layer : `layers.LayerBase` instance

            The layer to apply dropout to.

        p : float in [0, 1)

            The dropout propbability during training

        """

        self.p = p

    def OutShape(self,shape):

        return shape

    def forward(self, X,trainable=True):

        self.trainable=trainable

        scaler, mask = 1.0, np.ones(X.shape).astype(bool)

        if trainable:

            scaler = 1.0 / (1.0 - self.p)

            mask = np.random.rand(*X.shape) >= self.p

            X = mask * X

        self.mask = mask

        return scaler * X



    def backward(self, dLdy):

        assert self.trainable, "Layer is frozen"

        dLdy=self.mask*dLdy

        dLdy *= 1.0 / (1.0 - self.p)

        return dLdy


class res_block(object):

    def __init__(self, ksize=3, stride=1):

        self.stride = stride

        self.ksize = ksize

    def OutShape(self,shape):

        self.conv1=Conv2D(output_channels=7,ksize=self.ksize,stride=self.stride,method="SAME")

        self.bn1=BN()

        self.relu1=Relu()

        self.conv2=Conv2D(output_channels=shape[-1],ksize=self.ksize,stride=self.stride,method="SAME")

        self.bn2=BN()

        self.relu2=Relu()



        out_shape=shape

        out_shape=self.conv1.OutShape(out_shape)

        out_shape=self.conv2.OutShape(out_shape)



        return out_shape



    def forward(self, x):

        out=x

        out=self.conv1.forward(out)

        out=self.bn1.forward(out)

        out=self.relu1.forward(out)

        out=self.conv2.forward(out)

        out=self.bn2.forward(out)

        out=self.relu2.forward(x+out)

        return out

    def backward(self, eta):

        out=eta

        out=self.relu2.backward(out)

        outx=out

        out=self.bn2.backward(out)

        out=self.conv2.backward(out)

        out=self.relu1.backward(out)

        out=self.bn1.backward(out)

        out=self.conv1.backward(out)

        return out+outx

    def gradient(self, alpha=0.00001, weight_decay=0.0004):

        self.conv1.gradient(alpha=alpha,weight_decay=weight_decay)

        self.conv2.gradient(alpha=alpha,weight_decay=weight_decay)
class res_block(object):

    def __init__(self, ksize=3, stride=1):

        self.stride = stride

        self.ksize = ksize

    def OutShape(self,shape):

        self.conv1=Conv2D(output_channels=7,ksize=self.ksize,stride=self.stride,method="SAME")

        self.bn1=BatchNormal()

        self.relu1=Relu()

        self.conv2=Conv2D(output_channels=shape[-1],ksize=self.ksize,stride=self.stride,method="SAME")

        self.bn2=BatchNormal()

        self.relu2=Relu()



        out_shape=shape

        out_shape=self.conv1.OutShape(out_shape)

        out_shape=self.conv2.OutShape(out_shape)



        return out_shape



    def forward(self, x):

        out=x

        out=self.conv1.forward(out)

        out=self.bn1.forward(out)

        out=self.relu1.forward(out)

        out=self.conv2.forward(out)

        out=self.bn2.forward(out)

        out=self.relu2.forward(x+out)

        return out

    def backward(self, eta):

        out=eta

        out=self.relu2.backward(out)

        outx=out

        out=self.bn2.backward(out)

        out=self.conv2.backward(out)

        out=self.relu1.backward(out)

        out=self.bn1.backward(out)

        out=self.conv1.backward(out)

        return out+outx

    def gradient(self, alpha=0.00001, weight_decay=0.0004):

        self.conv1.gradient(alpha=alpha,weight_decay=weight_decay)

        self.conv2.gradient(alpha=alpha,weight_decay=weight_decay)
class Net(object):

    def __init__(self, seq):

        self.seq=seq

        self.bkseq=seq[::-1]

        #最后一层一般与损失函数结合一起计算，不需要反向传播

        self.bkseq=self.bkseq[1:]

        self.shape_Computed=False

    def ComputeShape(self,input_shape):

        out_shape=input_shape

        for layer in self.seq:

            out_shape=layer.OutShape(out_shape)

        return out_shape



    def forward(self,imgs,training=True):

        if self.shape_Computed ==False:

            self.ComputeShape(imgs.shape)

            self.shape_Computed =True



        out=imgs

        for layer in self.seq:

            out=layer.forward(out)

        return out



    def backward(self,grad,training=True):

        for layer in self.bkseq:

            grad=layer.backward(grad)

        return grad



    def Gradient(self,alpha=0.001, weight_decay=0.001):

        for layer in self.seq:

            try:

                layer.gradient(alpha=alpha, weight_decay=weight_decay)

            except :

                pass 
class my_data_set:

    def __init__(self, kind='train'):

        if kind=='train':

            train_data = pd.read_csv('../input/digit-recognizer/train.csv')

            labels = train_data['label'].values

            X= train_data.drop(columns=['label']).values/255

            y = np.zeros((labels.shape[0], 10))

            for i in range(labels.shape[0]):

                y[i][labels[i]] = 1

            labels = np.array(y)

            self.labels=labels

        else: 

            test_data = pd.read_csv("../input/digit-recognizer/test.csv")

            X = test_data.values/255

            self.labels=np.zeros((X.shape[0], 10))

       

        self.images=X.reshape(X.shape[0],28,28,1)

        self.size=self.images.shape[0]

        self.i=0

        # return images, labels

    def next_batch(self,batch_size):

        if batch_size>=self.size: return self.images,self.labels



        i=self.i

        if i + batch_size<=self.size:

            batch_xs1 = self.images[i :i +  batch_size,...]

            batch_ys1 = self.labels[i :i + batch_size,...]

            self.i=self.i+batch_size

            if self.i==self.size:

                self.i=0

                # print (self.i,batch_xs1.shape[0])

            return batch_xs1,batch_ys1

        if i <self.size:

            batch_xs1 = self.images[i:,...]

            batch_ys1 = self.labels[i :,...]

            self.i =batch_size-self.size+self.i

            batch_xs1 = np.concatenate([batch_xs1,self.images[0:self.i ,...]],axis=0)

            batch_ys1 =np.concatenate([batch_ys1,self.labels[0:self.i ,...]],axis=0)

            # print (self.i,batch_xs1.shape[0])

            return batch_xs1,batch_ys1
data = my_data_set(kind='train')





#超参数

batch_size = 32

learning_rate = 0.01

# epochs=500

#定义网络结构

#不同的参数和结构会有不同的准确率，由于每次初始化也会影响到准确率，

#下面这个网络在epochs=5时准确率在98.6%到98.75%之间

#下面这个网络在epochs=87时准确率99%

#每个epoch耗时85秒



epochs=5

seq=[

    Conv2D(output_channels=32,ksize=5,stride=1, method='SAME'),

    BatchNormal(),

    Relu(),



    Conv2D(output_channels=32, ksize=5, stride=1, method='SAME'),

    BatchNormal(),

    Relu(),



    MaxPooling(),

    Dropout(p=0.25),



    Conv2D(output_channels=64,ksize=3,stride=1, method='SAME'),

    BatchNormal(),

    Relu(),



    Conv2D(output_channels=64, ksize=3, stride=1, method='SAME'),

    BatchNormal(),

    Relu(),



    MaxPooling(),

    Dropout(p=0.25),



    Dense(output_num=256),

    Relu(),

    Dense( output_num=10),

    # Sigmoid()

    Softmax()

]





#残差网络

#每个epoch耗时97秒



# epochs=5

# seq=[

#     Conv2D(output_channels=7,ksize=3,stride=1),

#     MaxPooling(),

#     res_block(),

#     res_block(),

#     Dropout(p=0.2),

#     Dense( output_num=10),

#     Softmax()

# ]





#input_shape必须是BHWC的顺序，如果不是，需要reshape和tanspose成NHWC顺序

net=Net(seq=seq)
def train():

    for epoch in range(epochs):

        start=time.time()





        batch_loss = 0

        batch_acc = 0

        val_acc = 0

        val_loss = 0



        # train

        total=0

        train_acc = 0

        train_loss = 0

        # imgs,labs=data.next_batch(batch_size)

        for i in range(60000//batch_size):

#             if i>100: 

#                 break

            

            imgs, labs = data.next_batch(batch_size)

            imgs=imgs



            #训练

            sf_out=net.forward(imgs)

            net.backward(sf_out-labs)

            net.Gradient(alpha=learning_rate, weight_decay=0.01)



            #统计

            for j in range(batch_size):

                if np.argmax(sf_out[j]) == np.argmax(labs[j]):

                    train_acc += 1

                total+=1



            mod = 10

            if i % mod == 0:

                print ("epoch=%d  batchs=%d   train_acc: %.4f  " % (epoch,i, train_acc / total))

                train_acc = 0

                total=0



        print("----------------------------------------------------------------------------------------------------")

        print("epoch=",epoch," batchs=%d      time is:  %5.5f (sec)"%(i,time.time()-start))

        start=time.time()

    #     test()

        print("----------------------------------------------------------------------------------------------------")

train()
import os

for dr in os.listdir("../input/digit-recognizer"):

    print(dr)
test=pd.read_csv("../input/digit-recognizer/test.csv")
test.head()
submission=pd.read_csv("../input/digit-recognizer/sample_submission.csv")

test_data = my_data_set(kind='test')
def  test():

    train_acc=0

    total=0

    batch_size=28

    for i in range(28000//batch_size):

        imgs, labs = test_data.next_batch(batch_size)

        sf=net.forward(imgs,training=False)

        for j in range(batch_size):

            index=i*batch_size+j

            submission.loc[index]["Label"]=np.argmax(sf[j])

            if index%batch_size==27:

                print(i,index,np.argmax(sf[j]))

            

    
test()
submission.to_csv("submission.csv",index=0)
!ls