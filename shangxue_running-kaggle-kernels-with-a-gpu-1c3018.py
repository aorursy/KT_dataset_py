import numpy as np #矩阵运算

import pandas as pd #特征工程

import os #os模块负责程序与操作系统的交互，提供了访问操作系统底层的接口

import cv2 #opencv 图像预处理

import matplotlib.pyplot as plt #画图

import matplotlib.patches as patches #生成常见图形对象如：矩形，椭圆，圆形，多边形

import random #生成随机变量

from sklearn.utils import shuffle #洗牌：某个集合的随机排列组合

from tqdm import tqdm_notebook #进度条



data = pd.read_csv('/kaggle/input/train_labels.csv') #读取csv文件转化为pandas dataframe

train_path = '/kaggle/input/train/' #训练集路径

test_path = '/kaggle/input/test/' #测试集路径

# quick look at the label stats 

data['label'].value_counts() #pd.Series.value_counts()  统计data['label']这个series的每种值的个数
def readImage(path):

    # OpenCV reads the image in bgr format by default   OpenCV默认以bgr格式读取图像，path是图像的地址

    bgr_img = cv2.imread(path)

    # We flip it to rgb for visualization purposes   我们将图片翻转为rgb格式用于观看

    # cv2.split函数分离得到各个通道的灰度值(单通道图像)  https://blog.csdn.net/eric_pycv/article/details/72887758

    b,g,r = cv2.split(bgr_img)

    # cv2.merge函数是合并单通道成多通道（不能合并多个多通道图像） 

    rgb_img = cv2.merge([r,g,b])

    return rgb_img
# random sampling训练标记随机抽样

shuffled_data = shuffle(data) 

# matpoltlib subplots :创建一系列的子图  nrows子图行的个数2，ncols子图列的个数5 ,figsize画布尺寸, axes子图的轴， fig图

fig, ax = plt.subplots(2,5, figsize=(20,8))

# 标题

fig.suptitle('Histopathologic scans of lymph node sections淋巴结切片的组织病理学扫描',fontsize=20)

# Negatives反例(没有转移癌)

for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 0]['id'][:5]): # idx是随机排列的前五个反例的图片名，i是0~4的索引

    path = os.path.join(train_path, idx) # 将训练集路径与反例图片名 拼接起来

    ax[0,i].imshow(readImage(path + '.tif')) # 在坐标为[0,i]的轴上画出这张反例图

    # Create a Rectangle patch # 画出蓝色中心虚线矩形框

    box = patches.Rectangle((32,32),32,32,linewidth=4,edgecolor='b',facecolor='none', linestyle=':', capstyle='round')

    ax[0,i].add_patch(box) # 在坐标为[0,i]的轴上画出这个矩形框

ax[0,0].set_ylabel('Negative samples反例样本(无肿瘤)', size='large') # 给坐标为[0,0]的轴画出其y轴标题

# Positives正例(有转移癌)

for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 1]['id'][:5]): # idx是随机排列的前五个正例的图片名，i是0~4的索引

    path = os.path.join(train_path, idx) # 将训练集路径与正例图片名 拼接起来

    ax[1,i].imshow(readImage(path + '.tif'))  # 在坐标为[1,i]的轴上画出这张正例图

    # Create a Rectangle patch # 画出红色中心虚线矩形框

    box = patches.Rectangle((32,32),32,32,linewidth=4,edgecolor='r',facecolor='none', linestyle=':', capstyle='round')

    ax[1,i].add_patch(box) # 在坐标为[0,i]的轴上画出这个矩形框

ax[1,0].set_ylabel(u"Tumor tissue samples肿瘤组织样本", size='large') # 给坐标为[1,0]的轴画出其y轴标题
import random # 生成随机变量

ORIGINAL_SIZE = 96      # original size of the images - do not change原始图片尺寸



# AUGMENTATION VARIABLES # 数据增强的变量

CROP_SIZE = 90          # final size after crop剪裁后的图片大小

RANDOM_ROTATION = 3    # range (0-180), 180 allows all rotation variations, 0=no change旋转角度



RANDOM_SHIFT = 2        #> center crop shift in x and y axes, 0=no change. This cannot be more than (ORIGINAL_SIZE - CROP_SIZE)//2 

# 随机平移0到2个像素值



RANDOM_BRIGHTNESS = 7  # range (0-100), 0=no change 随机亮度 给图像每个像素加上一个在(-1,1)间的随机值

RANDOM_CONTRAST = 5    # range (0-100), 0=no change 随机对比度 给图像每个像素乘上一个在(0,2)间的随机值

RANDOM_90_DEG_TURN = 1  # 0 or 1= random turn to left or right向左或右旋转90度



def readCroppedImage(path, augmentations = True):

    # augmentations parameter is included for counting statistics from images, where we don't want augmentations这个参数可以用来选择是否使用数据增强

      

    # OpenCV reads the image in bgr format by default 用opencv读取图片，通道格式默认是bgr

    bgr_img = cv2.imread(path)

    # We flip it to rgb for visualization purposes 我们将图片翻转为rgb格式用于观看

    # cv2.split函数分离得到各个通道的灰度值(单通道图像)  https://blog.csdn.net/eric_pycv/article/details/72887758

    b,g,r = cv2.split(bgr_img)

    # cv2.merge函数是合并单通道成多通道（不能合并多个多通道图像） 

    rgb_img = cv2.merge([r,g,b])

    

    if(not augmentations): # 如果augmentations等于false,即不做数据增强，只将图像像素的值归一化

        return rgb_img / 255

    

    #random rotation 随机旋转

    rotation = random.randint(-RANDOM_ROTATION,RANDOM_ROTATION)  #在(a,b)之间随机生成一个整数

    if(RANDOM_90_DEG_TURN == 1):#如果随机旋转90度开启为1，rotation就随机加上90度或0度或负90度

        rotation += random.randint(-1,1) * 90 

    M = cv2.getRotationMatrix2D((48,48),rotation,1)   # 第一个参数是旋转的中心点，第二个参数是旋转的角度，第三个参数是图像缩放因子，1

    rgb_img = cv2.warpAffine(rgb_img,M,(96,96)) # 第一个参数是输入图片，第二个参数是仿射变换矩阵，第三个是输出图片的尺寸

    

    #random x,y-shift根据x轴y轴随机平移

    x = random.randint(-RANDOM_SHIFT, RANDOM_SHIFT) # 在(a,-a)之间生成一个随机整数，用于x轴方向的平移

    y = random.randint(-RANDOM_SHIFT, RANDOM_SHIFT) # 在(a,-a)之间生成一个随机整数，用于y轴方向的平移

    # crop to center and normalize to 0-1 range 中心剪裁 并将图像像素值归一化

    start_crop = (ORIGINAL_SIZE - CROP_SIZE) // 2 # 开始剪裁的坐标。默认左上角坐标是(0,0)

    end_crop = start_crop + CROP_SIZE  # 终止剪裁的坐标。默认左上角坐标是(0,0)

    rgb_img = rgb_img[(start_crop + x):(end_crop + x), (start_crop + y):(end_crop + y)] / 255  # 除号之前的是对原图像做中心剪裁后的图像矩阵，除是为了归一化

    

    # Random flip 随机翻转

    flip_hor = bool(random.getrandbits(1)) # getrandbits(k)返回一个k比特位随机的整数.bool(1)是true,bool(0)是false。

    flip_ver = bool(random.getrandbits(1)) # getrandbits(k)返回一个k比特位随机的整数.bool(1)是true,bool(0)是false。

    if(flip_hor):

        rgb_img = rgb_img[:, ::-1] # [:,::-1]将图像每一行的像素做倒序，相当于水平方向翻转

    if(flip_ver):

        rgb_img = rgb_img[::-1, :] # [::-1, :] 将图像每一列的像素做倒序，相当于垂直方向翻转

        

    # Random brightness 随机亮度 给图像每个像素加上一个在(-1,1)间的随机值

    br = random.randint(-RANDOM_BRIGHTNESS, RANDOM_BRIGHTNESS) / 100.   

    rgb_img = rgb_img + br

    

    # Random contrast 随机对比度 给图像每个像素乘上一个在(0,2)间的随机值

    cr = 1.0 + random.randint(-RANDOM_CONTRAST, RANDOM_CONTRAST) / 100.

    rgb_img = rgb_img * cr

    

    # clip values to 0-1 range 

    rgb_img = np.clip(rgb_img, 0, 1.0) # 图像中像素值比0小的变为0，比1大的变为1

    

    return rgb_img
print("julyfan")
# matpoltlib subplots :创建一系列的子图  nrows子图行的个数，ncols子图列的个数 ,figsize画布尺寸, axes子图的轴， fig图

fig, ax = plt.subplots(2,5, figsize=(20,8))

fig.suptitle('Cropped histopathologic scans of lymph node sections 切除淋巴结切片的组织病理学扫描',fontsize=20) # 标题

# Negatives 反例(没有转移癌)

for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 0]['id'][:5]): # idx是随机排列的前五个反例的图片名，i是0~4的索引

    path = os.path.join(train_path, idx) # 将训练集路径与反例图片名 拼接起来

    ax[0,i].imshow(readCroppedImage(path + '.tif'))  # 在坐标为[0,i]的轴上画出这张反例图

ax[0,0].set_ylabel('Negative samples反例(没有转移癌)', size='large') # 给坐标为[0,0]的轴画出其y轴标题

# Positives 正例(有转移癌)

for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 1]['id'][:5]): # idx是随机排列的前五个正例的图片名，i是0~4的索引

    path = os.path.join(train_path, idx) # 将训练集路径与正例图片名 拼接起来

    ax[1,i].imshow(readCroppedImage(path + '.tif')) # 在坐标为[1,i]的轴上画出这张正例图

ax[1,0].set_ylabel('Tumor tissue samples', size='large') # 给坐标为[1,0]的轴画出其y轴标题
print("julyfan")
# matpoltlib subplots :创建一系列的子图  nrows子图行的个数，ncols子图列的个数 ,figsize画布尺寸, axes子图的轴， fig图

fig, ax = plt.subplots(1,5, figsize=(20,4))

fig.suptitle('Random augmentations to the same image对同一图像的随机增强',fontsize=20) # 标题

# Negatives 反例(没有转移癌)

for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 0]['id'][:1]): # idx是随机排列的第一个反例的图片名，i是索引0

    for j in range(5): # 随机图像增强五次，看效果

        path = os.path.join(train_path, idx) # 将训练集路径与反例图片名 拼接起来

        ax[j].imshow(readCroppedImage(path + '.tif')) # 调用之前自定义的readCroppedImage方法
print("julyfan")
# As we count the statistics, we can check if there are any completely black or white images 在统计数据时，我们可以检查是否有完全黑色或白色的图像

dark_th = 10 / 255      # If no pixel reaches this threshold, image is considered too dark 如果一张图片没有像素高于此阈值，则图像被认为太暗

bright_th = 245 / 255   # If no pixel is under this threshold, image is considerd too bright 如果一张图片没有像素低于此阈值，则认为图像太亮

too_dark_idx = [] #太暗图片的名字

too_bright_idx = []  #太亮图片的名字



x_tot = np.zeros(3)

x2_tot = np.zeros(3)

counted_ones = 0

for i, idx in tqdm_notebook(enumerate(shuffled_data['id']), 'computing statistics...(220025 it total)'): # tqdm_notebook可用于显示进度条，其第二个参数是提示用的

    path = os.path.join(train_path, idx) # 将训练集路径与反例图片名 拼接起来

    # 执行自定义的函数，不做数据增强，但做归一化，并将图像的形状从(96,96,3) 转成 (9216,3)

    imagearray = readCroppedImage(path + '.tif', augmentations = False).reshape(-1,3) 

    # is this too dark 如果图像太暗了

    if(imagearray.max() < dark_th): # 如果图像像素值最大的也比 设定的最小阈值小

        too_dark_idx.append(idx) # 把这张图片的名字记录在小本本上

        continue # do not include in statistics 跳过本次循环，不将其纳入求平均、求方差

    # is this too bright 如果图像太亮了

    if(imagearray.min() > bright_th): # 如果图像像素值最小的也比 设定的最大阈值大

        too_bright_idx.append(idx) # 把这张图片的名字记录在小本本上

        continue # do not include in statistics 跳过本次循环，不将其纳入求平均、求方差

    x_tot += imagearray.mean(axis=0) # 每个通道的像素值的平均之和 最后维度是(3,)

    x2_tot += (imagearray**2).mean(axis=0) # 每个通道的像素值的平方的平均之和 最后维度是(3,)

    counted_ones += 1 # 正常图片的个数

    

channel_avr = x_tot/counted_ones #每个通道的像素值平均值 最后维度是(3,)

#？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？

channel_std = np.sqrt(x2_tot/counted_ones - channel_avr**2) #每个通道的像素值标准差 最后维度是(3,)

#？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？

channel_avr,channel_std
print('There was {0} extremely dark image'.format(len(too_dark_idx))) # 太暗的图像的个数

print('and {0} extremely bright images'.format(len(too_bright_idx))) # 太亮的图像的个数

print('Dark one:') 

print(too_dark_idx) # 打印太暗图片的名字

print('Bright ones:') # 打印太亮图片的名字

print(too_bright_idx) 
fig, ax = plt.subplots(2,6, figsize=(25,9))

# matpoltlib subplots :创建一系列的子图  nrows子图行的个数，ncols子图列的个数 ,figsize画布尺寸, axes子图的轴， fig图

fig.suptitle('Almost completely black or white images',fontsize=20) # 标题

# Too dark 展示太暗的图片

i = 0

for idx in np.asarray(too_dark_idx)[:min(6, len(too_dark_idx))]: # 遍历太暗的图像 最多只展示6张图片，多了图片太小不好看了

    lbl = shuffled_data[shuffled_data['id'] == idx]['label'].values[0] # 太暗的图像的标记

    path = os.path.join(train_path, idx) # 将训练集路径与图片名 拼接起来

    ax[0,i].imshow(readCroppedImage(path + '.tif', augmentations = False)) # 展示这张图片，调用readCroppedImage的原因是，需要rgb格式的图像数据

    ax[0,i].set_title(idx + '\n label=' + str(lbl), fontsize = 8) # 给坐标为(0,i)图像设置标题

    i += 1 

ax[0,0].set_ylabel('Extremely dark images', size='large') # 给坐标为(0,0)图像设置y轴标题



for j in range(min(6, len(too_dark_idx)), 6): # 遍历[0,1]~[0,5]的轴

    ax[0,j].axis('off') # hide axes if there are less than 6 # 让这些没有图片的轴不显示

    

# Too bright 展示太亮的图片

i = 0

for idx in np.asarray(too_bright_idx)[:min(6, len(too_bright_idx))]: # 遍历太亮的图像 最多只展示6张图片，多了图片太小不好看了

    lbl = shuffled_data[shuffled_data['id'] == idx]['label'].values[0] # 太亮的图像的标记

    path = os.path.join(train_path, idx) # 将训练集路径与图片名 拼接起来

    ax[1,i].imshow(readCroppedImage(path + '.tif', augmentations = False)) # 展示这张图片，调用readCroppedImage的原因是，需要rgb格式的图像数据

    ax[1,i].set_title(idx + '\n label=' + str(lbl), fontsize = 8) # 给坐标为(0,i)图像设置标题

    i += 1

ax[1,0].set_ylabel('Extremely bright images', size='large') # 给坐标为(1,0)图像设置y轴标题

for j in range(min(6, len(too_bright_idx)), 6): # 遍历没有图像的轴(当然这里太亮的图片数量刚好是6，range(6,6)就不用遍历了)

    ax[1,j].axis('off') # hide axes if there are less than 6 # 让这些没有图片的轴不显示
print("fan")
from sklearn.model_selection import train_test_split # 分割训练集、测试集的方法



# we read the csv file earlier to pandas dataframe, now we set index to id so we can perform

# 我们之前读取了pandas dataframe的csv文件，现在我们将id列作为索引列，这样我们就可以接着往下执行了

train_df = data.set_index('id') 



# If removing outliers, uncomment the four lines below如果要删除离群值(异常值)，需要将下面四句话取消注释

# print('Before removing outliers we had {0} training samples.'.format(train_df.shape[0])) # 打印 在删除离群值之前，我们的训练集有多少条数据

# train_df = train_df.drop(labels=too_dark_idx, axis=0) # 删除训练集中太黑的图片

# train_df = train_df.drop(labels=too_bright_idx, axis=0) # 删除训练集中太亮的图片

# print('After removing outliers we have {0} training samples.'.format(train_df.shape[0])) @ 打印 在删除离群值之后，训练集还有多少条数据



train_names = train_df.index.values # 返回训练集的索引(图像名字)数组

train_labels = np.asarray(train_df['label'].values) # 返回训练图像对应的标记



# split, this function returns more than we need as we only need the validation indexes for fastai 

# split，这个函数返回的值 比我们需要的多，因为我们只需要验证集索引(val_idx)给fastai库

# Split the train and the validation set for the fitting将原训练集分割成训练集和验证集,test_size是占原来数据集的比例

# stratify是为了保持split前类的分布。 https://www.e-learn.cn/content/qita/780160

# random_state 可复现的随机切分方式

tr_n, tr_idx, val_n, val_idx = train_test_split(train_names, range(len(train_names)), test_size=0.1, stratify=train_labels, random_state=123)
print("julyfan")
print("julyfan")
# fastai 1.0

from fastai import * # 更快、更简单、更先进的深度学习库

from fastai.vision import * # 包含定义计算机视觉中的数据集和训练模型所必须的函数

from torchvision.models import *    # import *=all the models from torchvision  



arch = densenet169                  # specify model architecture, densenet169 seems to perform well for this data but you could experiment

# 指定已经训练好的模型结构，densenet169似乎对这些数据表现良好，不信您可以试试



BATCH_SIZE = 128                    # specify batch size, hardware restrics this one. Large batch sizes may run out of GPU memory

# 指定批量大小，这个参数对硬件有限制。 大批量可能会耗尽GPU内存。



sz = CROP_SIZE                      # input size is the crop size 输入的图像尺寸



MODEL_PATH = str(arch).split()[1]   # this will extrat the model name as the model file name e.g. 'resnet50' 分割字符串取第二个
print("julyfan")
# create dataframe for the fastai loader 创建一个dataframe 用于 定义fastai数据加载器ImageDataBunch

train_dict = {'name': train_path + train_names, 'label': train_labels} # 创建一个字典 name：训练集地址+图像名  label:图像名对应的标记

df = pd.DataFrame(data=train_dict) # 创建一个训练集dataframem，列名是name,label



# create test dataframe 创建测试集dataframe

test_names = []

for f in os.listdir(test_path): # os.listdidr返回一个列表包含某路径下所有文件名

    test_names.append(test_path + f) # 将测试集路径和测试图像名组合成一个字符串 追加到列表中

df_test = pd.DataFrame(np.asarray(test_names), columns=['name'])  # 构造一个dataframe，数据是test_names，列名是name
print("julyfan")
# Subclass ImageList to use our own image opening function 创建ImageList的子类，并自定义open函数加载图片，也是用于定义ImageDataBunch

class MyImageItemList(ImageList): # 创建类，继承自ImageList

    def open(self, fn:PathOrStr)->Image: # :PathOrStr,->Image 都是注释

        img = readCroppedImage(fn.replace('/./','').replace('//','/')) # 调用之前定义的读取图片的函数，str.replace 换字符串

        # This ndarray image has to be converted to tensor before passing on as fastai Image, we can use pil2tensor  

        # 这个ndarray图像必须在转变成fastai Image之前转换为张量tensor格式，我们可以使用pil2tensor方法实现

        return vision.Image(px=pil2tensor(img, np.float32)) # pil2tensor把pil格式的图片数组转换为torch格式的图片张量

        # https://docs.fast.ai/vision.image.html#Images

        # vision.Image在torch中是对每个图片进行封装
print("julyfan")
# https://docs.fast.ai/data_block.html#The-data-block-API 



# Create ImageDataBunch using fastai data block API  创建数据加载器ImageDataBunch

imgDataBunch = (MyImageItemList.from_df(path='/', df=df, suffix='.tif') # 根据训练集dataframe的列中的路径创建torch中的ItemList

        #Where to find the data?:

                

        .split_by_idx(val_idx) # ItemList.split_by_idx：根据val_idx拆分训练集合和验证集

        #How to split in train/valid?

                

        .label_from_df(cols='label') # 设置验证集的标签

        #Where are the labels?

                

        .add_test(MyImageItemList.from_df(path='/', df=df_test)) # 设置测试集

        #dataframe pointing to the test set?

                

            

        .transform(tfms=[[],[]], size=sz) # 数据增强 

        # We have our custom transformations implemented in the image loader but we could apply transformations also here

        # 我们在之前的图像加载器中实现了自定义的数据增强，但我们也可以在这里应用数据增强

        # Even though we don't apply transformations here, we set two empty lists to tfms. Train and Validation augmentations

        # 如果我们不在这里应用数据增强，需要将两个空列表设置给tfms。 



                

        .databunch(bs=BATCH_SIZE) # 将训练集、验证集、测试集捆绑成一个对象

        # convert to databunch

        # https://docs.fast.ai/basic_data.html#DataBunch

        

        .normalize([tensor([0.702447, 0.546243, 0.696453]), tensor([0.238893, 0.282094, 0.216251])])

        # Normalize with training set stats. These are means and std's of each three channel and we calculated these previously in the stats step.

        # 对数据标准化。 这些参数是图像三个通道的均值和标准，我们先前在统计步骤中计算了这些。

       )
print("julyfan")
# check that the imgDataBunch is loading our images ok   检查imageDataBunch是否正确加载您的图像

imgDataBunch.show_batch(rows=2, figsize=(4,4)) 
print("julyfan")
# Next, we create a convnet learner object 创建一个卷积神经网络学习器对象

# ps = dropout percentage (0-1) in the final layer  ps是最后一层随机失活的比例

def getLearner(): 

    return create_cnn(imgDataBunch, arch, pretrained=True, path='.', metrics=accuracy, ps=0.5, callback_fns=ShowGraph)

#  `create_cnn` becomes `cnn_learner` 这个方法在新版本中改为cnn_learner 获得一个用于迁移学习的模型 。imgDataBunch是数据加载器。 arch = densenet169 。dropout，pretrained如果是false则参数随机初始化。 .

learner = getLearner() # 创建一个卷积神经网络学习器

# learner是一个Learner对象

print("julyfan")
# 1cycle策略，得到在不同权重衰减系数和学习率下 的损失值

# We can use lr_find with different weight decays and record all losses so that we can plot them on the same graph  

# 我们可以使用具有不同衰减权重的lr_find并记录所有损失值，以便我们可以在同一图表上绘制它们

# Number of iterations is by default 100, but at this low number of itrations, there might be too much variance

# 默认情况下，迭代次数为100，但如果次数比较少，可能会有比较高的方差

# from random sampling that makes it difficult to compare WD's. I recommend using an iteration count of at least 300 for more consistent results.

# 随机抽样，很难比较WD(权重衰减)的效果。 我建议使用至少300的迭代计数以获得更一致的结果。

lrs = [] # 学习率

losses = [] # 损失值

wds = [] # 衰减权重

iter_count = 600 # 迭代轮数



# WEIGHT DECAY = 1e-6   衰减权重是1e-6

learner.lr_find(wd=1e-6, num_it=iter_count) # 帮助你找到一个模型的最佳学习率 wd权重衰减  num_it 迭代次数 https://github.com/fastai/fastai/search?q=lr_find&unscoped_q=lr_find

# recorder : 记录了epoch轮次、loss损失值、opt优化器、metric指标    https://docs.fast.ai/basic_train.html#Recorder

lrs.append(learner.recorder.lrs) # 把训练的学习率保存到lrs数组中

losses.append(learner.recorder.losses) # 把训练的损失值保存到losses数组中

wds.append('1e-6') # 把训练的衰减权重保存到wds数组中

learner = getLearner() #reset learner - this gets more consistent starting conditions 重设学习器





# WEIGHT DECAY = 1e-4   衰减权重是1e-4

learner.lr_find(wd=1e-4, num_it=iter_count)

lrs.append(learner.recorder.lrs)

losses.append(learner.recorder.losses)

wds.append('1e-4')

learner = getLearner() #reset learner - this gets more consistent starting conditions 重设学习器





# WEIGHT DECAY = 1e-2   衰减权重是1e-2

learner.lr_find(wd=1e-2, num_it=iter_count)

lrs.append(learner.recorder.lrs)

losses.append(learner.recorder.losses)

wds.append('1e-2')

learner = getLearner() #reset learner
print("fan")
# Plot weight decays 画出三种权重衰减值下的 学习率与损失值的关系图

_, ax = plt.subplots(1,1) # 创建一个子图

min_y = 0.5 

max_y = 0.55 

for i in range(len(losses)):

    ax.plot(lrs[i], losses[i]) # 在图中画出学习率-损失值对

    min_y = min(np.asarray(losses[i]).min(), min_y) # 找到最小损失下界

ax.set_ylabel("Loss") # y轴标签是Loss

ax.set_xlabel("Learning Rate") # x轴标签是Learning Rate

ax.set_xscale('log') # x轴尺度是log对数

#ax ranges may need some tuning with different model architectures 当使用不同的模型时，ax的范围可能需要做调整

ax.set_xlim((1e-3,3e-1)) # 设置x轴上界和下界 

ax.set_ylim((min_y - 0.02,max_y)) # 设置y轴上界和下界

ax.legend(wds) # 图例 (1e-6 1e-4 1e-2) 

ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e')) # 设置x轴标签文本的格式
print("julyfan")
max_lr = 2e-2 # 最大学习率

wd = 1e-4 # 最大权重衰减值

# 1cycle policy

learner.fit_one_cycle(cyc_len=8, max_lr=max_lr, wd=wd) # 根据1cycle策略拟合模型  https://docs.fast.ai/train.html#fit_one_cycle
print("julyfan")
# plot learning rate of the one cycle 画出1cycle策略的学习率曲线 由小变大、再由大变小

learner.recorder.plot_lr() #  https://docs.fast.ai/basic_train.html#Recorder.plot_lr
# and plot the losses of the first cycle 

learner.recorder.plot_losses() # 画出训练集和测试集的损失曲线  https://docs.fast.ai/basic_train.html#Recorder.plot_losses
# predict the validation set with our model 预测模型的验证集

interp = ClassificationInterpretation.from_learner(learner) # 分类模型的效果解释函数 https://docs.fast.ai/vision.learner.html#ClassificationInterpretation

interp.plot_confusion_matrix(title='Confusion matrix') # 画出混淆矩阵 https://docs.fast.ai/vision.learner.html#_cl_int_plot_multi_top_losses
# before we continue, lets save the model at this stage  在我们继续之前，让我们保存模型这个阶段的模型为1

learner.save(MODEL_PATH + '_stage1')
print("julyfan")
# load the baseline model 加载之前保存的基线模型权重

learner.load(MODEL_PATH + '_stage1')

 

# unfreeze and run learning rate finder again 解冻所有神经网络层并再次运行寻找最佳学习率方法

learner.unfreeze() 

learner.lr_find(wd=wd) # 之前设置的最大衰减权重



# plot learning rate finder results 画出寻找学习率的结果

learner.recorder.plot()  
print("julyfan")
# Now, smaller learning rates. This time we define the min and max lr of the cycle 现在，学习率较低。 这次我们定义循环的最小和最大学习率

learner.fit_one_cycle(cyc_len=12, max_lr=slice(4e-5,4e-4)) # 根据1cycle策略拟合模型  https://docs.fast.ai/train.html#fit_one_cycle
print("julyfan")
learner.recorder.plot_losses() # 画出训练集和验证集的损失 https://docs.fast.ai/basic_train.html#Recorder.plot_losses
print("julyfan")
# lets take a second look at the confusion matrix. See if how much we improved. 让我们再看一下混淆矩阵。 看看我们改进了多少。

interp = ClassificationInterpretation.from_learner(learner)  # 分类模型的效果解释函数 https://docs.fast.ai/vision.learner.html#ClassificationInterpretation

interp.plot_confusion_matrix(title='Confusion matrix')  # 画出混淆矩阵 https://docs.fast.ai/vision.learner.html#_cl_int_plot_multi_top_losses
print("julyfan")
# Save the finetuned model

learner.save(MODEL_PATH + '_stage2') # 保存微调的模型
print("julyfan")
# if the model was better before finetuning, uncomment this to load the previous stage 如果在微调之前的模型更好，则取消注释 加载前一阶段的模型

#learner.load(MODEL_PATH + '_stage1') 
print("julyfan")
preds,y, loss = learner.get_preds(with_loss=True) # 预测、预测标记、损失值 https://docs.fast.ai/basic_train.html#Learner.get_preds

# get accuracy

acc = accuracy(preds, y) # 计算精度 https://docs.fast.ai/metrics.html#accuracy

print('The accuracy is {0} %.'.format(acc))
print("julyfan")
# I modified this from the fastai's plot_top_losses (https://github.com/fastai/fastai/blob/master/fastai/vision/learner.py#L114)

from random import randint # 随机生成一个整数



def plot_overview(interp:ClassificationInterpretation, classes=['Negative','Tumor']): 

    # interp 是 ClassificationInterpretation 分类模型的效果解释函数 https://docs.fast.ai/vision.learner.html#ClassificationInterpretation

    

    # top losses will return all validation losses and indexes sorted by the largest first

    # top_losses 将返回所有 验证集的损失值(tl_val)和其索引(tl_idx) 按降序排序

    tl_val,tl_idx = interp.top_losses()

    # https://docs.fast.ai/vision.learner.html#ClassificationInterpretation.top_losses

    

    #classes = interp.data.classes

    fig, ax = plt.subplots(3,4, figsize=(16,12))

    # matpoltlib subplots : 创建一系列的子图  nrows子图行的个数，ncols子图列的个数 ,figsize画布尺寸, ax子图的轴， fig图

    

    fig.suptitle('Predicted / Actual / Loss / Probability',fontsize=20) #标题

    

    

    

    # Random展示随机抽样的图例

    for i in range(4): # 只展示四张图

        random_index = randint(0,len(tl_idx)) # 在(0,len(tl_idx))之间随机生成一个整数

        idx = tl_idx[random_index] # 随机得到一张图片的索引

        

        # interp.data   :   https://docs.fast.ai/basic_train.html#See-results

        # dl  : 返回一些数据用于验证、训练或者测试 https://docs.fast.ai/basic_data.html#DataBunch.dl

        # https://docs.fast.ai/basic_train.html#Learner.dl

        # DatasetType.Valid : https://docs.fast.ai/basic_data.html#DatasetType

        # im 是图片 cl 是类别class,值是0/1，0指代负例，1指代肿瘤

        im,cl = interp.data.dl(DatasetType.Valid).dataset[idx] 

        

        im = image2np(im.data) # 将图片从pytorch格式转为numpy格式  https://docs.fast.ai/vision.image.html#image2np

        cl = int(cl) # 转换为整数

        ax[0,i].imshow(im)  # 在第一行第i个轴上展示这张图片

        ax[0,i].set_xticks([]) # 第一行第i个轴的x轴没有刻度

        ax[0,i].set_yticks([]) # 第一行第i个轴的y轴没有刻度

        ax[0,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}') # 设置标题

        # classes[interp.pred_class[idx]]是第idx的预测类别 classes[cl]是其真实标记  interp.losses[idx]是损失值  interp.probs是置信度？？

    ax[0,0].set_ylabel('Random samples', fontsize=16, rotation=0, labelpad=80) # 给第一行第一个设置y轴标签

    

    

    # Most incorrect or top losses 展示前四个最不正确(损失值最大)的图例

    for i in range(4): # 只展示四张图

        idx = tl_idx[i] # tl_idx的前四个是损失值最高的

        im,cl = interp.data.dl(DatasetType.Valid).dataset[idx]  # im 是图片 cl 是类别class,值是0/1，0指代负例，1指代肿瘤

        cl = int(cl) # 转换为整数

        im = image2np(im.data) # 将图片从pytorch格式转为numpy格式  https://docs.fast.ai/vision.image.html#image2np

        ax[1,i].imshow(im) # 在第二行第i个轴上展示这张图片

        ax[1,i].set_xticks([]) # 第二行第i个轴的x轴没有刻度

        ax[1,i].set_yticks([]) # 第二行第i个轴的y轴没有刻度

        ax[1,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}') # 设置标题

    ax[1,0].set_ylabel('Most incorrect\nsamples', fontsize=16, rotation=0, labelpad=80) # 给第二行第一个设置y轴标签



    

    # Most correct or least losses  展示前四个最正确(损失值最小)的图例

    for i in range(4): # 只展示四张图

        idx = tl_idx[len(tl_idx) - i - 1] # tl_idx的后四个是损失值最高的

        im,cl = interp.data.dl(DatasetType.Valid).dataset[idx]  # im 是图片 cl 是类别class,值是0/1，0指代负例，1指代肿瘤

        cl = int(cl) # 转换为整数

        im = image2np(im.data) # 将图片从pytorch格式转为numpy格式  https://docs.fast.ai/vision.image.html#image2np

        ax[2,i].imshow(im) # 在第三行第i个轴上展示这张图片

        ax[2,i].set_xticks([]) # 第三行第i个轴的x轴没有刻度

        ax[2,i].set_yticks([]) # 第三行第i个轴的y轴没有刻度

        ax[2,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}') # 设置标题

    ax[2,0].set_ylabel('Most correct\nsamples', fontsize=16, rotation=0, labelpad=80) # 给第三行第一个设置y轴标签
print("julyfan")
#interp = ClassificationInterpretation.from_learner(learner) 这是之前定义的

plot_overview(interp, ['Negative','Tumor']) # 调用上面这个函数
print("julyfan")
from fastai.callbacks.hooks import *



# hook into forward pass 前向传播时自动调用这个钩子，用于检查和修改每一层的输出

# https://docs.fast.ai/callbacks.hooks.html#Hook

# https://pytorch.org/tutorials/beginner/former_torchies/nn_tutorial.html#forward-and-backward-function-hooks



# fastai 回调 ： https://pouannes.github.io/blog/callbacks-fastai/

def hooked_backward(m, oneBatch, cat):

    # we hook into the convolutional part = m[0] of the model

    with hook_output(m[0]) as hook_a:  # 创建一个保存了模型激活函数输出的钩子 https://docs.fast.ai/callbacks.hooks.html#hook_output

        with hook_output(m[0], grad=True) as hook_g:

            preds = m(oneBatch) # 输入oneBatch给构造的模型做预测

            preds[0,int(cat)].backward() # 反向传播

    # 两个解释，我暂时看不懂(通过上下文管理器删除钩子？)：

    # https://forums.fast.ai/t/lesson-6-advanced-discussion/31442/6

    # https://forums.fast.ai/t/lesson-6-advanced-discussion/31442/2

    return hook_a,hook_g
print("julyfan")
# We can create a utility function for getting a validation image with an activation map

# 工具函数：画出一个验证集图片的热图

def getHeatmap(val_index):

    """Returns the validation set image and the activation map"""

    # 返回验证集图像和激活映射(热图)

    

    # this gets the model

    m = learner.model.eval()  

    # 评估模型

    # https://pytorch.org/tutorials/beginner/saving_loading_models.html

    # Remember that you must call model.eval() to set dropout and batch normalization 

    # layers to evaluation mode before running inference（推理预测）. 

    # Failing to do this will yield inconsistent inference results.



    tensorImg,cl = imgDataBunch.valid_ds[val_index]

    # valid_ds 是 validation的缩写 https://docs.fast.ai/dev/abbr.html

    # 输出图片列表tensorImg,类别列表cl，

    # https://dejanbatanjac.github.io/2019/03/15/ImageDataBunch.html

    # valid_ds 是 fastai v0.7 https://forums.fast.ai/t/how-to-get-data-val-ds-fnames-in-fastai-v1-0/37231



    

    # create a batch from the one image

    oneBatch,_ = imgDataBunch.one_item(tensorImg)

    # one_item  Get item into a batch. Optionally detach and denorm.  https://docs.fast.ai/basic_data.html#DataBunch.one_item

    # one_batch   Get one batch of from the DataBunch. Returns x,y with size of the batch_size (if bs=128 then there is a list of 128 elements).    https://docs.fast.ai/basic_data.html#DataBunch.one_batch

    # #get tensor from Image tensorfImg https://dhruvs.space/posts/grad-cam-heatmaps-along-resnet-34/



    

    oneBatch_im = vision.Image(imgDataBunch.denorm(oneBatch)[0])

    # denorm是denormalize的偏函数  https://docs.fast.ai/vision.data.html#denormalize

    # 反归一化denormalize的理解  https://stackoverflow.com/questions/4684622/how-to-normalize-denormalize-a-vector-to-range-11

    

    # convert batch tensor image to grayscale image with opencv从一个颜色空间到另一个颜色空间

    cvIm = cv2.cvtColor(image2np(oneBatch_im.data), cv2.COLOR_RGB2GRAY)

    # cv2.cvtColor https://docs.opencv.org/2.4.13.7/modules/imgproc/doc/miscellaneous_transformations.html#cv2.cvtColor

    # image2np  将图片从pytorch格式转为numpy格式  https://docs.fast.ai/vision.image.html#image2np

    

    # attach hooks 附上钩子？

    hook_a,hook_g = hooked_backward(m, oneBatch, cl) # 前面自定义的方法

    # get convolutional activations and average from channels

    

    acts = hook_a.stored[0].cpu()

    #avg_acts = acts.mean(0)

    # 这个钩子把激活的值存储到了stored里面，并放置到cpu里面



    # Grad-CAM

    grad = hook_g.stored[0][0].cpu()  #[0][0]??

    # 这个钩子把激活的值存储到了stored里面，并放置到cpu里面

    

    grad_chan = grad.mean(1).mean(1) # 求均值

    grad.shape,grad_chan.shape 

    mult = (acts*grad_chan[...,None,None]).mean(0)

    return mult, cvIm

    #mult是热点  cvIm是背景

# 生成热图的其他例子，方法挺类似的，有很多解释，有时间了细看，相信能解决很多疑惑：

# https://dhruvs.space/posts/grad-cam-heatmaps-along-resnet-34/
print("julyfan")
# Then, modify our plotting func a bit 终于开始画图函数啦（与前面的类似）

def plot_heatmap_overview(interp:ClassificationInterpretation, classes=['Negative','Tumor']):

    # interp 分类模型的效果解释函数 https://docs.fast.ai/vision.learner.html#ClassificationInterpretation

    tl_val,tl_idx = interp.top_losses()

    #classes = interp.data.classes

    # top_losses 将返回所有 验证集的损失值(tl_val)和其索引(tl_idx) 按降序排序

    # top losses will return all validation losses and indexes sorted by the largest first

    

    fig, ax = plt.subplots(3,4, figsize=(16,12))

    # matpoltlib subplots : 创建一系列的子图  nrows子图行的个数，ncols子图列的个数 ,figsize画布尺寸, ax子图的轴， fig图

    

    fig.suptitle('Grad-CAM\nPredicted / Actual / Loss / Probability',fontsize=20) #标题

    

    # Random 随机抽样

    for i in range(4):

        random_index = randint(0,len(tl_idx)) # 在(0,max_offset)之间随机生成一个整数

        idx = tl_idx[random_index] # 该验证集的索引

        act, im = getHeatmap(idx) # 得到该图片的热图和背景

        H,W = im.shape # 图片长宽

        _,cl = interp.data.dl(DatasetType.Valid).dataset[idx] # _ 是图片 cl 是类别class,值是0/1，0指代负例，1指代肿瘤

        cl = int(cl) # 转换为整数型

        ax[0,i].imshow(im) # 在第1行，第i+1列上展示背景图

        ax[0,i].imshow(im, cmap=plt.cm.gray) # 颜色图为灰度图

        ax[0,i].imshow(act, alpha=0.5, extent=(0,H,W,0), 

              interpolation='bilinear', cmap='inferno') 

        # 画出热图

        # interpolation插值：代表图片像素颜色没有过渡   https://blog.csdn.net/goldxwang/article/details/76855200

        # 某个点的值越大，颜色越深

        # alpha透明度

        # cmp : color map颜色图

        # extent 拉伸图像 https://www.cnblogs.com/lijiazhang/p/10105722.html

        # https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imshow.html

        ax[0,i].set_xticks([]) # 第一行第i个轴的x轴没有刻度

        ax[0,i].set_yticks([]) # 第一行第i个轴的y轴没有刻度

        ax[0,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}')# 设置标题

        # f-string https://blog.csdn.net/qq_33453253/article/details/79653546

    ax[0,0].set_ylabel('Random samples', fontsize=16, rotation=0, labelpad=80) # 给第一行第一个设置y轴标签



    # Most incorrect or top losses 展示前四个最不正确(损失值最大)的图例

    for i in range(4): # 只展示四张图

        idx = tl_idx[i] # # tl_idx的前四个是损失值最高的

        act, im = getHeatmap(idx) # 得到该图片的热图和背景

        H,W = im.shape # 图片长宽

        _,cl = interp.data.dl(DatasetType.Valid).dataset[idx] # _ 是图片 cl 是类别class,值是0/1，0指代负例，1指代肿瘤

        cl = int(cl) # 转换为整数型

        ax[1,i].imshow(im) # 在第1行，第i+1列上展示背景图

        ax[1,i].imshow(im, cmap=plt.cm.gray) # 颜色图为灰度图

        ax[1,i].imshow(act, alpha=0.5, extent=(0,H,W,0),

              interpolation='bilinear', cmap='inferno')

        # 同上

        ax[1,i].set_xticks([]) # 第二行第i个轴的x轴没有刻度

        ax[1,i].set_yticks([]) # 第二行第i个轴的y轴没有刻度

        ax[1,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}') # 设置标题

        # f-string https://blog.csdn.net/qq_33453253/article/details/79653546

    ax[1,0].set_ylabel('Most incorrect\nsamples', fontsize=16, rotation=0, labelpad=80) # 给第二行第一个设置y轴标签

    

    # (同上)

    # Most correct or least losses 展示前四个最正确(损失值最小)的图例

    for i in range(4):

        idx = tl_idx[len(tl_idx) - i - 1]

        act, im = getHeatmap(idx)

        H,W = im.shape

        _,cl = interp.data.dl(DatasetType.Valid).dataset[idx]

        cl = int(cl)

        ax[2,i].imshow(im)

        ax[2,i].imshow(im, cmap=plt.cm.gray)

        ax[2,i].imshow(act, alpha=0.5, extent=(0,H,W,0),

              interpolation='bilinear', cmap='inferno')

        ax[2,i].set_xticks([])

        ax[2,i].set_yticks([])

        ax[2,i].set_title(f'{classes[interp.pred_class[idx]]} / {classes[cl]} / {interp.losses[idx]:.2f} / {interp.probs[idx][cl]:.2f}')

    ax[2,0].set_ylabel('Most correct\nsamples', fontsize=16, rotation=0, labelpad=80)
print("julyfan")
plot_heatmap_overview(interp, ['Negative','Tumor']) # 调用上面的方法，画出热图
print("julyfan")
from sklearn.metrics import roc_curve, auc

# probs from log preds # 

probs = np.exp(preds[:,1]) # 计算每个元素的指数e^x

# Compute ROC curve # 计算ROC曲线

fpr, tpr, thresholds = roc_curve(y, probs, pos_label=1)

# fpr 假阳性率 tpr：真阳性率 ,thresholds：阈值列表  y:真实标签 ， probs预测值，pos_label=1标签为1的是正例，其都是反例

# https://blog.csdn.net/u014264373/article/details/80487766

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html#sklearn-metrics-roc-curve



# Compute ROC area # 计算曲面下面积

roc_auc = auc(fpr, tpr)

# https://scikit-learn.org/stable/modules/classes.html#classification-metrics

print('ROC area is {0}'.format(roc_auc)) # ROC area is 0.9942634117111718
print("julyfan")
plt.figure() #matplotlib plt.figure:创建一个新图

plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)

#画出ROC曲线

plt.plot([0, 1], [0, 1], color='navy', linestyle='--') # 画出对角线

plt.xlim([-0.01, 1.0]) # 设置x轴上界和下界

plt.ylim([0.0, 1.01]) # 设置y轴上界和下界

plt.xlabel('False Positive Rate') # x轴标签是

plt.ylabel('True Positive Rate') # y轴标签是

plt.title('Receiver operating characteristic') # 图的上方标题

plt.legend(loc="lower right") # 设置图例
print("julyfan")
# make sure we have the best performing model stage loaded 加载之前训练的效果最好的模型参数

learner.load(MODEL_PATH + '_stage2')



# Fastai has a function for this but we don't want the additional augmentations it does (our image loader has augmentations) so we just use the get_preds

#preds_test,y_test=learner.TTA(ds_type=DatasetType.Test)

#虽然fastai有测试时增强，但是我们之前加载数据的时候就做了数据增强，所以就不用再做了



# We do a fair number of iterations to cover different combinations of flips and rotations.

# 我们做了相当数量的迭代 来执行图像翻转和旋转的不同组合的操作。

# The predictions are then averaged.

# 然后对预测结果取平均

n_aug = 12

# 增强次数

preds_n_avg = np.zeros((len(learner.data.test_ds.items),2))

# 设置一个预测值矩阵，初始值为0，形状是(len(learner.data.test_ds.items):测试集数目 , 2：反例和正例概率



for n in tqdm_notebook(range(n_aug), 'Running TTA...'): # tqdm_notebook可用于显示进度条，其第二个参数是提示用的

    preds,y = learner.get_preds(ds_type=DatasetType.Test, with_loss=False)

    # preds预测值、y预测标记  with_loss=False不计损失值  https://docs.fast.ai/basic_train.html#Learner.get_preds    

    preds_n_avg = np.sum([preds_n_avg, preds.numpy()], axis=0)

    # (2, 57458, 2) -> (57458, 2) 对第一个维度聚合相加，也可以简单理解为两个矩阵的元素对应相加

preds_n_avg = preds_n_avg / n_aug # 对这几次概率之和求平均
print("julyfan")
# Next, we will transform class probabilities to just tumor class probabilities

print('Negative and Tumor Probabilities: ' + str(preds_n_avg[0])) # 第一行 第一个样本的反例和正例的概率

tumor_preds = preds_n_avg[:, 1] # 概率矩阵第二列 即癌症的概率

print('Tumor probability: ' + str(tumor_preds[0])) # 第一个样本是正例(癌症)的概率

# If we wanted to get the predicted class, argmax would get the index of the max

class_preds = np.argmax(preds_n_avg, axis=1) 

# 顺着第一个维度(行)求最大值的坐标，即求每个样本的概率最大的坐标(0或1)

# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.argmax.html



classes = ['Negative','Tumor'] # preds_n_avg 0 对应了Negative , 1对应了 Tumor

print('Class prediction: ' + classes[class_preds[0]])
print("fan")
# get test id's from the sample_submission.csv and keep their original order

# 从提交模板中取得id并保持其顺序

# 从sample_submission.csv这个文件中，提取测试集的id，并保持原有的顺序

SAMPLE_SUB = '/kaggle/input/sample_submission.csv'

sample_df = pd.read_csv(SAMPLE_SUB) #pandas 读csv文件:提交文件

sample_list = list(sample_df.id) # dataframe.columns   提交文件的id列 ;Series转换为list



# List of tumor preds. 生成癌症预测概率列表

# These are in the order of our test dataset and not necessarily in the same order as in sample_submission

# 这些预测概率是按照我们测试集中的顺序排列的，稍后要重新排序

pred_list = [p for p in tumor_preds] #numpy.ndarray -> list  python列表生成式



# To know the id's, we create a dict of id:pred 把我们的id和预测值 组成 一个字典

pred_dic = dict((key, value) for (key, value) in zip(learner.data.test_ds.items, pred_list))

# python zip 把多个序列的元素对应编织在一起  https://www.python.org/dev/peps/pep-0201/#id10



# Now, we can create a new list with the same order as in sample_submission

# 现在，我们要创建一个和提交模板顺序相同的列表

pred_list_cor = [pred_dic['///kaggle/input/test/' + id + '.tif'] for id in sample_list]

# 循环地提取sample_list中的id，找到其在pred_dice字典中对应的值，生成一个规定顺序的预测概率列表



# Next, a Pandas dataframe with id and label columns.

df_sub = pd.DataFrame({'id':sample_list,'label':pred_list_cor})

# 生成一个dataframe，第一列是提交模板的id，第二列label是按照规定顺序排列的预测概率



# Export to csv

df_sub.to_csv('{0}_submission.csv'.format(MODEL_PATH), header=True, index=False)

# dataframe to_csv 将dataframe转换为csv文件，header=True保留列名，index=False不要保存索引
# This is what the first 10 items of submission look like 查看提交数据的前十行

df_sub.head(10)

# pandas dataframe head 
print("julyfan")
# This will create an export.pkl file that you'll need to copy with your model file if you want to deploy it on another device.

# This saves the internal information (classes, etc) need for inference in a file named 'export.pkl'. 

imgDataBunch.export(file='./export.pkl') # 存储模型推理预测所需要的最少的信息

# https://docs.fast.ai/data_block.html#LabelList.export

# https://github.com/fastai/fastai/blob/c15331a7b8b9fa908dbe6c0bcb38ba124e0d2768/fastai/data_block.py#L663
######## RUN THIS ON A NEW MACHINE请在另一台新机器上运行下面的代码 ##########

# from fastai.vision import * # 包含定义计算机视觉中的数据集和训练模型所必须的函数

# from fastai import * # 更快、更简单、更先进的深度学习库

# from torchvision.models import *  # pytorch视觉模块

#arch = densenet169       # specify model architecture 所用的模型架构

#MODEL_PATH = str(arch).split()[1] + '_stage2' # 训练的模型权重

#empty_data = ImageDataBunch.load_empty('./') #这会自动搜索export.pkl文件：this will look for a file named export.pkl in the specified path

#learner = create_cnn(empty_data, arch).load(MODEL_PATH)

# 定义一个卷积神经网络，空数据，densenet169,权重采用MODEL_PATH
print("fan")
## And then we are ready to do predictions 然后我们就可以开始推理预测了

import cv2 # opencv

sz = 68



# This function will convert image to the prediction format #转换图像格式到模型想要的格式

def imageToTensorImage(path):

    # OpenCV reads the image in bgr format by default 用opencv读取图片，通道格式默认是bgr

    bgr_img = cv2.imread(path) 

    # cv2.split函数分离得到各个通道的灰度值(单通道图像)  https://blog.csdn.net/eric_pycv/article/details/72887758

    b,g,r = cv2.split(bgr_img)

    # cv2.merge函数是合并单通道成多通道（不能合并多个多通道图像） 

    rgb_img = cv2.merge([r,g,b])

    # crop to center to the correct size and convert from 0-255 range to 0-1 range

    # 剪裁图像的重心到正确的尺寸，然后将像素值从0-255 缩放到 0-1 的范围

    H,W,C = rgb_img.shape # 图像长、宽，通道数

    rgb_img = rgb_img[(H-sz)//2:(sz +(H-sz)//2),(H-sz)//2:(sz +(H-sz)//2),:] / 256

    # (H-sz)//2是中心小图的x轴的切片的起点，(sz +(H-sz)//2是x轴的切片的终点。

    # (H-sz)//2是中心小图的y轴的切片的起点，(sz +(H-sz)//2)是y轴的切片的终点。

    # 保持通道数不变

    

    return vision.Image(px=pil2tensor(rgb_img, np.float32))

    # This ndarray image has to be converted to tensor before passing on as fastai Image, we can use pil2tensor  

    # 这个ndarray图像必须在传递为fastai Image之前转换为张量tensor格式，我们可以使用pil2tensor方法实现

    # pil2tensor把pil格式的图片数组转换为torch格式的图片张量

    # https://docs.fast.ai/vision.image.html#Images



        

img = imageToTensorImage('/kaggle/input/test/0eb051700fb6b1bf96188f36c8e4889598c6a157.tif')

# 调用上面定义的函数，生成了一个Image对象  https://docs.fast.ai/vision.image.html#Image



## predict and visualize 预测和图像可视化

img.show(y=learner.predict(img)[0]) # y是图的标题

# 展示图像 https://docs.fast.ai/vision.image.html#Image.show



# learner.predict(img) 推理预测，三个返回值是predicted class, label and probabilities

# 第一个返回值y=learner.predict(img)[0]是预测的类别，0对应了反例negative即正常，1对应了正例即癌症tumor

# https://docs.fast.ai/vision.learner.html#Learner.predict

classes = ['negative', 'tumor']

print('This is a ' + classes[int(learner.predict(img)[0])] + ' tissue image.')

# 当int(learner.predict(img)[0])是0时，classes[int(learner.predict(img)[0])]是negative。

# 当int(learner.predict(img)[0])是1时，classes[int(learner.predict(img)[0])]是tumor。
