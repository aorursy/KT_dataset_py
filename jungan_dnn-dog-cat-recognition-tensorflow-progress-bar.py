import cv2 # 机器视觉库，安装请用pip3 install opencv-python
import numpy as np # 数值计算库
import os # 系统库
from random import shuffle # 随机数据库 
from tqdm import tqdm # 输出进度库
import matplotlib.pyplot as plt # 常用画图库
train_dir = '../input/train/'
test_dir = '../input/test/'
img_size = 50
# learning rate
lr = 1e-3
def label_img(img):
    # img: cat.0.jpg
    word_label = img.split('.')[-3]
    if word_label == 'cat': return [1,0]
    elif word_label == 'dog': return [0,1]
def create_train_data():
    training_data = []
    #tqdm 输出进度条
    for img in tqdm(os.listdir(train_dir)):
        if (not img.endswith('.jpg')):
            continue
        # label_img function return either [1, 0] OR [0, 1] 的one hot encoding type的label
        label = label_img(img)
        path = os.path.join(train_dir, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # IMREAD_GRAYSCALE 读入灰度图
        img = cv2.resize(img, (img_size, img_size) )  # 将图片变成统一大小 img_size = 50 i.e.50 * 50
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    return training_data
        
train_data = create_train_data()
train_data[:2]
train_data[1][0]
train_data[1][1]
# diff from create_train_data functin, there is no lable to be added to testing_data array
def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(test_dir)):
        if (not img.endswith('.jpg')):
            continue
        path = os.path.join(test_dir,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        # different from create_train_data function, for each image, you add img_num
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    return testing_data
import tflearn # 需要安装tensorflow，然后安装tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d  # 2维CNN以及最大采样
from tflearn.layers.core import input_data, dropout, fully_connected # 输入层，dropout，全连接层
from tflearn.layers.estimator import regression # cross entropy层
import tensorflow as tf
tf.reset_default_graph()
# image_size = 50,   shape: 50 * 50 * 1  1通道， None,表示不确定多少样本个数
convnet = input_data(shape = [None, img_size, img_size, 1], name = 'input')
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation = 'relu')
convnet = dropout(convnet, 0.8)

# softmax 输出每一个类别的概率
convnet = fully_connected(convnet, 2, activation='softmax')
# 预测层
convnet = regression(convnet, optimizer='adam', learning_rate = lr, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log')
# 从开始 到  导数500 个，（不包含第倒数500个）
train = train_data[:-500]
# 这里的最后500张图片用于validation purpose, 用在training 阶段
test = train_data[-500:]
X = np.array([i[0] for i in train], dtype=np.float64).reshape(-1, img_size, img_size, 1)
print(X[:1])
y = np.array([i[1] for i in train], dtype=np.float64)
print(y[:2])
Xtest = np.array([i[0] for i in test], dtype=np.float64).reshape(-1, img_size, img_size, 1)
ytest = np.array([i[1] for i in test], dtype=np.float64)
# "input" match to: convnet = input_data(shape = [None, img_size, img_size, 1], name = 'input')
# "targets" match to: convnet = regression(convnet, optimizer='adam', learning_rate = lr, loss='categorical_crossentropy', name='targets')
model.fit({'input': X}, {'targets': y}, n_epoch=3, validation_set=({'input': Xtest}, {'targets': ytest}), snapshot_step=500, show_metric=True, run_id='model' )
test_data = process_test_data()
print(test_data[:2])
fig = plt.figure()
for num,data in enumerate(test_data[:2]):
    img_num = data[1]
    print(num) 
    print(img_num)                                 
    img_data = data[0]
    print(img_data)
    
fig = plt.figure()
for num,data in enumerate(test_data[:16]):
    # data[1]就是testing data里的图片编号
    img_num = data[1]
    # data[0] 就是testing data里的实际图片数据 50 * 50 二维数组
    img_data = data[0]
    # 4 行， 4 列
    y = fig.add_subplot(4, 4, num+1)
    orig = img_data
    # 每一张图片的size 50 * 50 * 1  只有1个通道的灰度图
    data = img_data.reshape(img_size, img_size, 1)
    raw_model_out = model.predict([data])
    model_out = raw_model_out[0]
    print(model_out)
    # cat': return [1,0]
    # 'dog': return [0,1]
    # 这里的argmax return 的最大值的index , 所以==1 就是dog
    if np.argmax(model_out) == 1: 
        label = 'Dog'
    else: 
        label = 'Cat'
    # 每个图片就是原始的图片用来验证预测的正确与否
    y.imshow(orig, cmap='gray')
    # 每个图片的title就是预测的label
    plt.title(label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
    
plt.tight_layout()
plt.show()
