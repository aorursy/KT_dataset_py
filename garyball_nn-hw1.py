import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#  tqdm 为显示训练进度的库
from tqdm import tqdm
from tqdm._tqdm import trange
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
pic = mpimg.imread('../input/pictures/p1.png') 
plt.imshow(pic) 
plt.axis('off')
plt.figure(dpi=150)
plt.show()
# useful functions:
def sigmoid(z):
    # Sigmoid activation function: 
    # f(z) = 1 / (1 + e^(-z))
    return 1 / (1 + np.exp(-z))

def deriv_sigmoid(z):
    # Derivative of sigmoid: 
    # f'(x) = f(x) * (1 - f(x))
    return sigmoid(z) * (1 - sigmoid(z))

def mse(y_true, y_pred):
    # mean squared error
    return ((y_true - y_pred) ** 2).mean()
class NN():
    """
    A neural network with:
      - 2 inputs
      - a hidden layer with 2 neurons (o1, o2)
      - an output layer with 1 neuron (o3)
    """
    def __init__(self):
        # weights initializing
        self.w1 = 0.1
        self.w2 = 0.1
        self.w3 = 0.1
        self.w4 = 0.1
        self.w5 = 0.1
        self.w6 = 0.1
        # biases
        self.b1 = 0.1
        self.b2 = 0.1
        self.b3 = 0.1
        
    def feedforward(self, x):
        # feed the two sample to the NN work together
        o1 = sigmoid(self.w1 * x[:,0] + self.w3 * x[:,1] + self.b1)
        o2 = sigmoid(self.w2 * x[:,0] + self.w4 * x[:,1] + self.b2)
        o3 = sigmoid(self.w5 * o1 + self.w6 * o2 + self.b3)
        return o3
    
    def train(self, x, y_true):
        """
        - x is a (2 x 2) numpy array for 2 samples
        - y_true is a numpy array with 2 elements,representing to the training data
        """
        learn_rate = 0.1
        epochs = 2 # number of times to loop through the entire dataset
        
        for epoch in range(epochs):
                
            # Do a feedforward to get values
            h1 = self.w1 * x[:,0] + self.w3 * x[:,1] + self.b1
            o1 = sigmoid(h1)
            
            h2 = self.w2 * x[:,0] + self.w4 * x[:,1] + self.b2
            o2 = sigmoid(h2)
                
            h3 = self.w5 * o1 + self.w6 * o2 + self.b3
            o3 = sigmoid(h3)
            # o3 equals to y_pred
                
            # Calculate partial derivatives.
            # Naming principle: d_L_d_w1 represents "partial L / partial w1"
            d_L_d_o3 = o3-y_true
                
            # cell o3
            d_L_d_w5 = d_L_d_o3 * deriv_sigmoid(h3) * o1
            d_L_d_w6 = d_L_d_o3 * deriv_sigmoid(h3) * o2
            d_L_d_b3 = d_L_d_o3 * deriv_sigmoid(h3)
                
            # cell o1
            d_L_d_w1 = d_L_d_o3 * deriv_sigmoid(h3) * self.w5 * deriv_sigmoid(h1) * x[:,0]
            d_L_d_w3 = d_L_d_o3 * deriv_sigmoid(h3) * self.w5 * deriv_sigmoid(h1) * x[:,1]
            d_L_d_b1 = d_L_d_o3 * deriv_sigmoid(h3) * self.w5 * deriv_sigmoid(h1)
                
            # cell o2
            d_L_d_w2 = d_L_d_o3 * deriv_sigmoid(h3) * self.w6 * deriv_sigmoid(h2) * x[:,0]
            d_L_d_w4 = d_L_d_o3 * deriv_sigmoid(h3) * self.w6 * deriv_sigmoid(h2) * x[:,1]
            d_L_d_b2 = d_L_d_o3 * deriv_sigmoid(h3) * self.w6 * deriv_sigmoid(h2)
                
            # update weights and biases
            grad_unit = (sum(d_L_d_w1)**2 +sum(d_L_d_w2)**2 +sum(d_L_d_b1)**2 
                         +sum(d_L_d_w3)**2 +sum(d_L_d_w4)**2 +sum(d_L_d_b2)**2
                         +sum(d_L_d_w5)**2 +sum(d_L_d_w6)**2 +sum(d_L_d_b3)**2)
            
            # cell o3
            self.w5 -= learn_rate * sum(d_L_d_w5)/np.sqrt(grad_unit)
            self.w6 -= learn_rate * sum(d_L_d_w6)/np.sqrt(grad_unit)
            self.b3 -= learn_rate * sum(d_L_d_b3)/np.sqrt(grad_unit)
                
            # cell o1
            self.w1 -= learn_rate * sum(d_L_d_w1)/np.sqrt(grad_unit)
            self.w3 -= learn_rate * sum(d_L_d_w3)/np.sqrt(grad_unit)
            self.b1 -= learn_rate * sum(d_L_d_b1)/np.sqrt(grad_unit)
                
            # cell o2
            self.w2 -= learn_rate * sum(d_L_d_w2)/np.sqrt(grad_unit)
            self.w4 -= learn_rate * sum(d_L_d_w4)/np.sqrt(grad_unit)
            self.b2 -= learn_rate * sum(d_L_d_b2)/np.sqrt(grad_unit)

            y_pred = self.feedforward(x)
            loss = mse(y_true, y_pred)
            print("Epoch: {} loss: {}".format(epoch, loss))
            
            print("w1:{},w2:{},w3:{},\nw4:{},w5:{},w6:{},\nb1:{},b2:{},b3:{}".format(
                self.w1,self.w2,self.w3,self.w4,self.w5,self.w6,self.b1,self.b2,self.b3))
                
# dataset definition
x = np.array([
    [0.1, 0.1], 
    [0.2, 0.2],  
])
y_trues = np.array([
    0.3, # sample 1
    0.5, # sample 2
])

# Train
net = NN()
net.train(x, y_trues)
data = pd.read_excel('../input/nn-hw1-data/data_hw1.xlsx', sheet_name=0)
data.head()
data_np = np.array(data.iloc[:,1:])
# 查看每天的数据数
day_sampling_len = int(len(data)/31)
print(day_sampling_len)
# 第i天的数据
day_num = 5
day_samp = data[288*day_num:288*(day_num+1)]
day_samp.plot()
from scipy.fftpack import fft,ifft
from matplotlib.pylab import mpl
import heapq

mpl.rcParams['font.sans-serif'] = ['SimHei']   #显示中文
mpl.rcParams['axes.unicode_minus']=False       #显示负号
def normalize(x):
    max_val = x.max(axis=0, keepdims=True)
    min_val = x.min(axis=0, keepdims=True)
    return (x-min_val)/(max_val-min_val),max_val,min_val

day_samp_nor,max_,min_ = normalize(np.array(day_samp.iloc[:,1:]))
len(day_samp_nor)
fft_y=fft(day_samp_nor[:,1])
print(len(fft_y))
print(fft_y[0:5])
fft_y[0] = 0
N= 288
x = np.arange(N)           # 频率个数
 
abs_y=np.abs(fft_y)                # 取复数的绝对值，即复数的模(双边频谱)
angle_y=np.angle(fft_y)              #取复数的角度
 
plt.figure()
plt.plot(x,abs_y)   
plt.title('Binary Spectrum')
 
plt.figure()
plt.plot(x,angle_y)   
plt.title('Binary Phase')
plt.show()
n = 6
max_indexs = heapq.nlargest(n, range(len(abs_y)), abs_y.take)
print(max_indexs)
import matplotlib.pyplot
%matplotlib inline
import imageio
import glob
np.random.seed(0)
# neural network with 3 layers
class MulitNN:
    # 神经网络初始化
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        # 定义网络 的输入节点数，隐藏层节点数，输出节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # learning rate
        self.lr = learningrate
        
        # 权重链接矩阵
        # 权重 w_i_j, 代表着从第i个节点到下一层第j个节点。
        # 比如： w12 w21 
        self.wih = (np.random.normal(0.0, pow(self.hnodes,-0.5), (self.hnodes,self.inodes) )  )
        self.who = (np.random.normal(0.0, pow(self.onodes,-0.5), (self.onodes,self.hnodes) )  )
        
        # 通过sigmoid函数激活
        # self.activation_function = lambda x: scipy.special.expit(x)
        self.activation_function = lambda x: sigmoid(x)
        pass
    
    # 网络训练
    def train(self,inputs_list,targets_list):      
        inputs = np.array(inputs_list,ndmin=2).T
        targets = np.array(targets_list,ndmin=2).T
        
        # 计算feedforward 
        # 计算输入到隐藏层，并激活
        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # 计算隐藏层到输出层，并激活
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        # 输出层误差，也就是对损失函数对输出层的输入final_inputs的求导
        output_errors = final_outputs - targets
        # 隐藏层误差，用于计算损失函数对隐藏层权重的导数
        hidden_errors = np.dot(self.who.T, output_errors * final_outputs * (1.0 - final_outputs))
        
        # 更新who
        self.who -= self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        
        # 更新wih
        self.wih -= self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        
        pass
    
    # 计算输出结果
    def predict(self,inputs_list):
        inputs = np.array(inputs_list,ndmin=2).T
        
        # 计算前向传递过程
        hidden_inputs = np.dot(self.wih,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs
data_len = int(len(data)/288)-1
data_input,max_val,min_val= normalize(np.array(data.iloc[:,1:]))
x_list = []
y = []
value_index = 0
for day_num in tqdm(range(data_len)):
    # 找到序号为day_num的数据，提取120个数据样本,并找到value_index = 2的数据
    day_samp = data_input[288*day_num:(288*day_num+120)]
    day_value = day_samp[:,value_index]
    x_list.append(day_value)
    y.append(data_input[288*day_num+121,value_index])
# 设置输入节点数为120，隐藏层200个节点，输出就1个点，建立神经网络
input_nodes = 120
hidden_nodes = 200
output_nodes = 1
# 学习率0.01
learning_rate = 0.001

nn_wave = MulitNN(input_nodes,hidden_nodes,output_nodes,learning_rate)
# 训练1000次（反正训练的快）
epochs = 1000
for epoch in tqdm(range(epochs)):
    for i in range(len(x_list)):
        # 设置输入输出，进行训练
        inputs = x_list[i]
        target = y[i]
        # 训练
        nn_wave.train(inputs,target)
        pass
    output = nn_wave.predict(x_list)
    loss = np.mean((y-output)**2)
    pass
pred_item = data_input[288*30:(288*30+120)]
output = nn_wave.predict(pred_item[:,0])
output
y_pred = output*(max_val[:,0]-min_val[:,0])+min_val[:,0]
y_true = data_input[288*30+121,0]*(max_val[:,0]-min_val[:,0])+min_val[:,0]
MAPE = abs(y_pred-y_true)/y_true
MAPE
x2_list = []
y2 = []
# 设置不同的列 序号
value_index = 1
for day_num in tqdm(range(data_len)):
    # 找到序号为day_num的数据，提取120个数据样本,并找到value_index = 2的数据
    day_samp = data_input[288*day_num:(288*day_num+120)]
    day_value = day_samp[:,value_index]
    x2_list.append(day_value)
    y2.append(data_input[288*day_num+121,value_index])
    
# 设置输入节点数为120，隐藏层200个节点，输出就1个点，建立神经网络
input_nodes = 120
hidden_nodes = 200
output_nodes = 1
# 学习率0.01
learning_rate = 0.001

nn_wave2 = MulitNN(input_nodes,hidden_nodes,output_nodes,learning_rate)
epochs = 1000
for epoch in tqdm(range(epochs)):
    for i in range(len(x_list)):
        inputs = x_list[i]
        target = y[i]
        nn_wave2.train(inputs,target)
        pass
    output = nn_wave2.predict(x_list)
    loss = np.mean((y-output)**2)
    pass
pred_item = data_input[288*30:(288*30+120)]
output = nn_wave2.predict(pred_item[:,value_index])
y_pred = output*(max_val[:,value_index]-min_val[:,value_index])+min_val[:,value_index]
y_true = data_input[288*30+121,value_index]*(max_val[:,value_index]-min_val[:,value_index])+min_val[:,value_index]
MAPE = abs(y_pred-y_true)/y_true
MAPE
x3_list = []
y3 = []
value_index = 2
for day_num in tqdm(range(data_len)):
    # 找到序号为day_num的数据，提取120个数据样本,并找到value_index = 2的数据
    day_samp = data_input[288*day_num:(288*day_num+120)]
    day_value = day_samp[:,value_index]
    x3_list.append(day_value)
    y3.append(data_input[288*day_num+121,value_index])
# 设置输入节点数为120，隐藏层200个节点，输出就1个点，建立神经网络
input_nodes = 120
hidden_nodes = 200
output_nodes = 1
# 学习率0.01
learning_rate = 0.001

nn_wave3 = MulitNN(input_nodes,hidden_nodes,output_nodes,learning_rate)

# 训练1000次（反正训练的快）
epochs = 1000
for epoch in tqdm(range(epochs)):
    for i in range(len(x_list)):
        record = x_list[i]
        inputs = record
        # create the target output values (all 0.01, except the desired label which is 0.99)
        target = y[i]
        # all_values[0] is the target label for this record
        nn_wave3.train(inputs,target)
        pass
    output = nn_wave3.predict(x_list)
    loss = np.mean((y-output)**2)
    pass
pred_item = data_input[288*30:(288*30+120)]
output = nn_wave3.predict(pred_item[:,value_index])
y_pred = output*(max_val[:,value_index]-min_val[:,value_index])+min_val[:,value_index]
y_true = data_input[288*30+121,value_index]*(max_val[:,value_index]-min_val[:,value_index])+min_val[:,value_index]
MAPE = abs(y_pred-y_true)/y_true
MAPE
from keras.models import Model, load_model
from keras.layers import Dense, Activation, Input, Reshape, Dropout, Concatenate
input_layer = Input(shape=(120,), name="input_data")
hidden1 = Dense(units=100, activation='tanh',name="hidden1")(input_layer)
hidden2 = Dense(units=10, name="hidden2")(hidden1)
output = Dense(units=1, name="output")(hidden2)

model = Model(inputs=input_layer,outputs=output)
model.compile(optimizer = 'Adam', loss='mse')
model.summary()
x_list = []
y =  []

for day_num in tqdm(range(data_len+1)):
    # 找到序号为day_num的数据，提取120个数据样本,并找到value_index = 2的数据
    day_samp = data_input[288*day_num:(288*day_num+120),2]
    x_list.append(day_samp)
    y.append(data_input[288*day_num+121,2])
x = np.array(x_list)
y = np.array(y)
x_train = x[0:30]
x_test = x[30:]
y_train = y[0:30]
y_test = y[30:]
model.fit(x = x_train,y = y_train, batch_size = 2,epochs = 500)
model.evaluate(x = x_test,y = y_test)
pred_result = model.predict(x_test)
y_pred = pred_result*(max_val[:,value_index]-min_val[:,value_index])+min_val[:,value_index]
y_true = y_test*(max_val[:,value_index]-min_val[:,value_index])+min_val[:,value_index]
MAPE = abs(y_pred-y_true)/y_true
MAPE