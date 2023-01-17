import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import os
import cv2
from skimage import color
from skimage import io
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy
import random
import csv as csv
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 

TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test/'

IMG_SIZE = 50 #image size 50x50
def read_image(file_path):        
    img= cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img= cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    return img

def prep_data(images):
    count = len(images)
    data = np.ndarray((count, IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        if i%5000 == 0: print('Processed {} of {}'.format(i, count))
    
    return data

train_cats = sorted(glob.glob(os.path.join(TRAIN_DIR, 'cat*.jpg')))
train_dogs = sorted(glob.glob(os.path.join(TRAIN_DIR, 'dog*.jpg')))
train_all = train_dogs+train_cats 

random.Random(4).shuffle(train_all)

test_all = sorted(glob.glob(os.path.join(TEST_DIR, '*.jpg')))

#X变成矩阵形式，Y变成0/1向量形式
X_train = prep_data([path for path in train_all])
Y_train = np.array([[1., 0.] if 'dog' in name else [0., 1.] for name in train_all])

X_test = prep_data([path for path in test_all])
Y_test = np.array([[1., 0.] if 'dog' in name else [0., 1.] for name in test_all])
#定义神经网络，None---batch size大小不固定，IMG_SIZE--固定，1---图像通道数（灰度图显示，没有彩色图片）
#feature map的数量一步步变多，又一步步变少（32-64-128-64-32）
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
#第二层对前一个input做卷积（32个不同的5x5卷积），卷积的结果会通过relu凼数激活
#这里没有padding, 其实卷积是可以padding的
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)#卷积之后做5x5的max pooling
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
#最后一层卷积做完后加入全连接（大小是1024）
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8) #随机丢20%的参数
convnet = fully_connected(convnet, 2, activation='softmax') #再使用全连接
#train model
convnet = regression(convnet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets')

#用DNN(深度神经网络)来fit这个model
model = tflearn.DNN(convnet, tensorboard_dir='log')
model.fit({'input': X_train}, {'targets': Y_train}, n_epoch=3, validation_set=({'input': X_test}, {'targets': Y_test}), 
    snapshot_step=500, show_metric=True, run_id='dog-cat')
def pyramid(image, scale=1.5, minSize=(30, 30)): #图像金字塔
    # yield the original image
    yield image
    
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)#不停地缩小这张图
        h = int(image.shape[0] / scale)
        image = cv2.resize(image, (w, h))

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image
        
def sliding_window(image, stepSize, windowSize): #怎么找sliding_window的四个大小
    for y in range(0, image.shape[0] - windowSize[1], stepSize):
        for x in range(0, image.shape[1] - windowSize[0], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def get_grey_and_color_image(path): #把灰度图和彩色图都显示出来
    return cv2.imread(path), cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
def run_inference(image):
    (w, h) = (50, 50)
    # loop over the image pyramid
    scale = 1.
    cats_score = []
    cats_bbox = []
    dogs_score = []
    dogs_bbox = []
    for resized in pyramid(image, scale=1.5): #先枚举所有不同大小的图片
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(w, h)):
            input_data = np.ndarray((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)
            input_data[0] = window.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
            out = model.predict(input_data)[0]

            if np.argmax(out) == 1:
                if out[1] > 0.8:#选出来的score一定会大于0.8
                    cats_score.append(out[1])
                    cats_bbox.append((int(x * scale), int(y * scale), int(w * scale), int(h * scale)))
            else:
                if out[0] > 0.8:
                    dogs_score.append(out[0])
                    dogs_bbox.append((int(x * scale), int(y * scale), int(w * scale), int(h * scale)))

        scale = scale * 1.5
        #输出是猫的score和bbox和是狗的score和bbox
    return cats_score, cats_bbox, dogs_score, dogs_bbox
#可视化是猫的score和bbox和是狗的score和bbox
#这里没有用RCN的selective search, 用图像金字塔和sliding window解决的问题
def showWindows(image, dogs_bbox, cats_bbox):
    for (x, y, w, h) in dogs_bbox:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for (x, y, w, h) in cats_bbox:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(np.array(image).astype('uint8'))
    plt.show()
    
THRESHOLD = 0.2
def nms(scores, bboxes):#score和bounding box一一对应的
    #先排序（从高到低），把所有的score和bounding box都拿出来
    sortedRes = sorted(zip(scores, bboxes), key = lambda x: x[0], reverse=True)
    scores = [score for (score, bbox) in sortedRes]
    bboxes = [bbox for (score, bbox) in sortedRes]
    
    suppress = [0] * len(scores)
    result = []
    for i in range(len(bboxes)): #枚举所有的bounding box
        if suppress[i] == 1: #先判断bounding box有没有被删除
            continue
        
        #如果没被删除，计算下并集和交集，再相除下
        area = bboxes[i][2] * bboxes[i][3] #选择的bounding box的面积是多少
        result.append(bboxes[i]) #把这个设置为框A（ppt32）
        for j in range(i+1, len(bboxes)): #枚举到的另一个框（B），如果IOU比较大，就把框B删除
            if suppress[j] == 1:
                continue
            
            otherArea = bboxes[j][2] * bboxes[i][3] #枚举到的另一个框的大小，并且算下它和框A的交集
            xx1 = np.maximum(bboxes[i][0], bboxes[j][0])
            yy1 = np.maximum(bboxes[i][1], bboxes[j][1])
            xx2 = np.minimum(bboxes[i][0] + bboxes[i][2] - 1, bboxes[j][0] + bboxes[j][2] - 1)
            yy2 = np.minimum(bboxes[i][1] + bboxes[i][3] - 1, bboxes[j][1] + bboxes[j][3] - 1)
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h #交集的大小
            ovr = inter / min(area,otherArea) #IOU #除以并集或者是两个框的最小值
            if ovr > THRESHOLD:
                suppress[j] = 1  #如果> THRESHOLD，就把框B删除
    return result
color_image, image = get_grey_and_color_image(test_all[1])
cats_score, cats_bbox, dogs_score, dogs_bbox = run_inference(image)
showWindows(color_image.copy(), dogs_bbox, cats_bbox)
new_dogs_bboxes = nms(dogs_score, dogs_bbox)
new_cats_bboxes = nms(cats_score, cats_bbox)
showWindows(color_image.copy(), new_dogs_bboxes, new_cats_bboxes)