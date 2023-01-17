!pip install livelossplot #模型训练过程中的可视化工具
import os #输入输出工具，拼接路径
import cv2 #绘图工具（画飞机的线）
import ast #读文件
import numpy as np #最基础的python包，数据结构（list，array...）
import pandas as pd #数据处理工具（处理csv格式的文件）
import matplotlib.pyplot as plt #绘图工具（画很多飞机）
from tqdm import tqdm_notebook
from glob import glob
from keras.utils import to_categorical #读类别label
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy #评价指标
from keras.metrics import categorical_crossentropy #交叉熵损失函数
from keras.models import Sequential #序列模型（网络一层一层）
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint #回调函数（调整模型参数，保存最优模型）。reducelr，动态减小学习率
#from keras_tqdm import TQDMNotebookCallback
from livelossplot import PlotLossesKeras #为keras绘制loss图像的包
from keras.optimizers import Adam #一种引入动量的梯度下降优化器，Adam，（SGD随机梯度下降）为了调整模型参数。

#下一时刻的参数 = 上一时刻的参数 + （学习率 * 梯度） 学习率：调整梯度跨度的大小。

from keras.preprocessing import image #预处理图像的，img存储矩阵数据
from keras.models import Model #keras的基础模型。keras是一个顶层包。 汇编语言-（C语言）-python-tensorflow-keras
from keras import backend as K
%matplotlib inline #能显示你画的图

BASE_SIZE = 256 #图像矩阵
DP_DIR = '../input/inputdata/input/shuffle-csvs/'
INPUT_DIR = '../input/inputdata/input/quickdraw-doodle-recognition/'
NCSVS = 100 #验证集样本数
NCATS = 340 #总类别数

# 自定义函数，将笔划坐标数据转换为图像矩阵数据。所有图像一定要转换成矩阵数据。
def draw_cv2(raw_strokes, size=256, lw=6):
    #np.zeros((a,b)) 生成一个a行b列的矩阵，np.uint8是int的一种，8位的无符号整形。
    img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    for stroke in raw_strokes: #从3维到2维了
        for i in range(len(stroke[0]) - 1): #从第一笔开始 #len(a)求列表或者数组的长度
            #cv2.line(img, start, end, color, thickness) cv2是一个绘图库，cv2.line()是画线函数
            _ = cv2.line(img, (stroke[0][i], stroke[1][i]), (stroke[0][i + 1], stroke[1][i + 1]), (255,0,0), lw)
    if size != BASE_SIZE: #避免出错
        return cv2.resize(img, (size, size))
    else:
        return img
    
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
fig = plt.figure(figsize=(20, 11)) #制作一个图像，长20英寸宽11英寸。#画布
columns = 6 #子图列数
rows = 3 #子图行数
size = 256 #图像矩阵的行列数 CNN实质上是在处理size所在的数据 图像的本质是矩阵
filename = os.path.join(INPUT_DIR, 'train_simplified/airplane.csv')

#pd.sample(n) 随机选择n个样本，n行数据
df = pd.read_csv(filename).sample(columns*rows)

#dataframe将csv读入python的一种数据格式
#df.apply(func) 对dataframe中的每一行执行自定义函数func，这里的ast.literal_eval负责读入dataframe中的数据
df['drawing'] = df['drawing'].apply(ast.literal_eval) #确保迭代性

#创建三维数组x
x = np.zeros((len(df), size, size))
for i, raw_strokes in enumerate(df.drawing.values): 
    #enumerate(x)：给数据加索引，使其可以遍历。 例如enumerate(['dog','cat','human']) -> [(1,'dog'),(2,'cat'),(3,'human')]
    x[i] = draw_cv2(raw_strokes, size=size, lw=6)
x = x / 255. #0~255 归一化
x = x.reshape((len(df), size, size)).astype(np.float32) #（第一维，第二维，第三维）
y = np.array(df['word']) #将类别数据读入，并存在数组y里

for i in range(columns*rows):
    fig.add_subplot(rows,columns,i+1) #创建3X6子图中的第i+1个
    plt.imshow(x[i,]) #将第i个图像画出来，[i，]后面的两个维度都画出来
    plt.title(y[i]) #第i个图像标题是df['word']
plt.show() #展示所有图像
# 获取类别名称并对空格进行替换
classes_path = os.listdir(INPUT_DIR + 'train_simplified/')
classes_path = sorted(classes_path, key=lambda s: s.lower())
class_dict = {x[:-4].replace(' ', '_'):i for i, x in enumerate(classes_path)} #enumerate(x)：给数据加索引，使其可以遍历。
labels = {x[:-4].replace(' ', '_') for i, x in enumerate(classes_path)}

n_labels = len(labels)
print("Number of classes: {}".format(n_labels))   

n_files = n_labels # number of csv files same as labels due to the structure.
class Image_Generator: #生成训练数据和验证数据 白板演示
    
    def __init__(self, size, batchsize, ks, lw=6, ncsvs=NCSVS):
        self.size = size #图像尺寸
        self.bs = batchsize #Batch尺寸
        self.ks = ks #训练集所用的100个数据集索引 train_k{0-99}  ks ncsvs
        self.lw = lw #画线的粗细
        self.ncsvs = ncsvs #验证集所用的100个数据集索引
    
    def create_train(self): #创建训练集
        while True: #没有条件的循环
            for k in np.random.permutation(self.ks): #随机从100个train_k.csv文件中选择
                filename = os.path.join(DP_DIR, 'train_k{}.csv'.format(k))
                #filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(k))
                for df in pd.read_csv(filename, chunksize=self.bs): #从csv随机读入batch_size数据
                    x, y = Image_Generator.df_to_image_array(self, df)
                    yield x, y #等于return
    
    def create_valid(self): #创建验证集
        val_mark = Image_Generator.valid_mark(self)
        while True:
             #filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(val_mark))
            filename = os.path.join(DP_DIR, 'train_k{}.csv'.format(val_mark))
            for df in pd.read_csv(filename, chunksize=self.bs):
                x, y = Image_Generator.df_to_image_array(self, df)
                yield x, y  #等于return  
    
    def df_to_image_array(self, df):
        df['drawing'] = df['drawing'].apply(ast.literal_eval) #从dataframe中获取坐标数据
        x = np.zeros((len(df), self.size, self.size))
        for i, raw_strokes in enumerate(df.drawing.values):
            x[i] = draw_cv2(raw_strokes, size=self.size, lw=self.lw) #将笔划坐标转化为矩阵
        x = x / 255.
        x = x.reshape((len(df), self.size, self.size, 1)).astype(np.float32)  #uint8              
        y = to_categorical(df.y, num_classes=NCATS) #NCATS=340
        return x, y #x矩阵（图像）数据，y是标签或者叫类别名
    
    def valid_mark(self): #验证集不能和训练集重复
        for i in range(self.ncsvs):
            if i not in self.ks: 
                return i
size = 256
batchsize = 64
train_datagen = Image_Generator(size=size, batchsize=batchsize, ks=range(NCSVS - 1)).create_train()
x, y = next(train_datagen) #从一个batch的训练样本中随机选出18个。
fig = plt.figure(figsize=(20, 11))
for i in range(columns*rows):
    fig.add_subplot(rows,columns,i+1)
    plt.imshow(x[i,].reshape(size, size))
plt.show()
size = 80
batchsize = 512 
#NCSVS 验证集的索引数
#生成训练集和验证集
train_datagen = Image_Generator(size=size, batchsize=batchsize, ks=range(NCSVS - 1)).create_train() #create_train是类Image_Generator里面的方法
valid_datagen = Image_Generator(size=size, batchsize=batchsize, ks=range(NCSVS - 1)).create_valid() #

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D #卷积层，最大池化层，平均池化层
from keras.layers import Dense, Dropout, Flatten, Activation #全连接层，Dropout层（随机丢弃一些神经元，防止模型过拟合），偏平层（将数据变为一维），激活层
from keras.applications import MobileNet #一种CNN，MobileNet部署在手机上的，参数规模小400w

#input_shape=(第一维，第二维，第三维)
#input_shape=(2,2,1) 例子：[[[a,b],[c,d]]]
#input_shape=(3,2,1) 例子：[[[a,b,e],[c,d,f]]]
#input_shape=(2,3,1) 例子：[[[a,b],[c,d],[e,f]]]
#include_top：是否包括顶部3个Dense层。 weights：是否加载预训练的权重。
#只要size设置的和此项目一样，就可以套用
base_model = MobileNet(input_shape=(size, size, 1), include_top=False, weights=None, classes=n_labels) #封装好的模型，是一个高级的CNN，特点就是参数少，性能好。
#因为我们学校的设备有限，没有很好的GPU，只有1080Ti，所以我用这种轻量级的CNN网络。

x = base_model.output

x = GlobalAveragePooling2D()(x) #以x为此层的输入
x = Dense(1024, activation='relu')(x) #Dense(node) 512 or 1024
predictions = Dense(n_labels, activation='softmax')(x) #n_labels = 340， softmax就是归一化概率[0.2, 0.7, 0.3]. 

model = Model(inputs=base_model.input, outputs=predictions) #base_model对象继承于MobilNet，所以MobilNet的属性他都有，其中就包括.input和.output
model.summary()
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3) # top-3

model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy',
              metrics=[categorical_accuracy, top_3_accuracy]) #模型编译，设置优化器，设置损害函数，设置评价指标
model_name = '/kaggle/working/best_model.h5' #保存模型的路径，模型的存储都是.h5文件

callbacks = [ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=2),  #是在连续几个epoch之后，模型仍然不收敛。factor是衰减系数，factor*学习率，最小小到1e-6
             EarlyStopping(monitor='val_loss', patience=9, verbose=2), #verbose是可视化，0是不显示任何信息，1是每个iteration都显示，2是每个epoch显示一次
             ModelCheckpoint(model_name, save_best_only=True, save_weights_only=True), # save best model
             PlotLossesKeras()] #callbacks里的函数是自带的，里面的数据可以改
history = model.fit_generator(train_datagen, steps_per_epoch=1000, epochs=25, verbose=1, 
                              validation_data=valid_datagen, validation_steps=200, callbacks=callbacks) #fit是拟合模型，epochs：把训练数据学习多少次，自己调。
def gen_graph(history, title):#hi
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.plot(history.history['top_3_accuracy'])
    plt.plot(history.history['val_top_3_accuracy'])
    plt.title('Accuracy ' + title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation', 'Test top 3', 'Validation top 3'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss ' + title)
    plt.ylabel('MLogLoss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

gen_graph(history, "Your Name")
def df_to_image_array(df, size, lw=6):
    df['drawing'] = df['drawing'].apply(ast.literal_eval)
    x = np.zeros((len(df), size, size))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i] = draw_cv2(raw_strokes, size=size, lw=lw)
    x = x / 255.
    x = x.reshape((len(df), size, size, 1)).astype(np.float32)                        
    return x

model.load_weights('/kaggle/working/best_model.h5')
pred_results = []
chunksize = 10000
TEST_DIR = '../input/testdata/'
reader = pd.read_csv(TEST_DIR + 'test_simplified.csv', chunksize=chunksize)
#reader = pd.read_csv(INPUT_DIR + 'test_simplified.csv', chunksize=chunksize)
for chunk in tqdm_notebook(reader):
    imgs = df_to_image_array(chunk, size)
    pred = model.predict(imgs, verbose=0)
    top_3 =  np.argsort(-pred)[:, 0:3]  
    pred_results.append(top_3)
print("Finished test predictions...")
#prepare data for saving
reverse_dict = {v: k for k, v in class_dict.items()}
pred_results = np.concatenate(pred_results)
print("Finished data prep...")

preds_df = pd.DataFrame({'first': pred_results[:,0], 'second': pred_results[:,1], 'third': pred_results[:,2]})
preds_df = preds_df.replace(reverse_dict)

preds_df['words'] = preds_df['first'] + " " + preds_df['second'] + " " + preds_df['third']
sub = pd.read_csv(TEST_DIR + 'sample_submission.csv', index_col=['key_id'])
#sub = pd.read_csv(INPUT_DIR + 'sample_submission.csv', index_col=['key_id'])
sub['word'] = preds_df.words.values
sub.to_csv('submit.csv')
sub.head()
