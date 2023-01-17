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

IMG_SIZE = 50
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

# [1., 0.] -> dog, [0., 1.] -> cat, 相当于对label 做了one-hot-encoding
X_train = prep_data([path for path in train_all])
Y_train = np.array([[1., 0.] if 'dog' in name else [0., 1.] for name in train_all])

X_test = prep_data([path for path in test_all])
Y_test = np.array([[1., 0.] if 'dog' in name else [0., 1.] for name in test_all])
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
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
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')
model.fit({'input': X_train}, {'targets': Y_train}, n_epoch=3, validation_set=({'input': X_test}, {'targets': Y_test}), 
    snapshot_step=500, show_metric=True, run_id='dog-cat')
# construct 图像金字塔，from big size -> small size
def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image # ??? 为什么 yield 可以生产 image
    
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale) # while True, 所以会一直缩小下去，直到 < minSize
        h = int(image.shape[0] / scale)
        image = cv2.resize(image, (w, h))

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        yield image # yield the next image in the pyramid

# image size list: (374, 500), (249, 333), (166, 222), (110, 148), (73, 98), (48, 65), (32, 43)
# step size: 32, windowSize: (50, 50)
# 返回每个window/box 的 position (i.e. x, y), w, h
def sliding_window(image, stepSize, windowSize):
    # windowSize=(w, h) windowSize[0] = w, windowSize[1] =h
    #                   image.shape[0] = w, image.shape[1] = h,
    # ??? image.shape[0] - windowSize[1], w, h mapping 不对？
    for y in range(0, image.shape[0] - windowSize[1], stepSize):
        for x in range(0, image.shape[1] - windowSize[0], stepSize):
            # yield the current window. window (x, y) -> position
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def get_grey_and_color_image(path):
    return cv2.imread(path), cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
def run_inference(image):
    (w, h) = (50, 50)
    # loop over the image pyramid
    scale = 1.
    cats_score = []
    cats_bbox = []
    dogs_score = []
    dogs_bbox = []
    for resized in pyramid(image, scale=1.5): # generate图像金字塔, then loop over每一种size的图片
        print("scale")
        print(scale)
        # loop over the sliding window for each layer of the pyramid. "resized": represent each size of image
        #resized.shape: (w, h)(374, 500), (249, 333), (166, 222), (110, 148), (73, 98), (48, 65), (32, 43)
        for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(w, h)): # windowSize(50, 50)
            input_data = np.ndarray((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.uint8)
            input_data[0] = window.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
            out = model.predict(input_data)[0]

            if np.argmax(out) == 1: # argmax output index of largest number
                if out[1] > 0.8: # even column_1 大， 还加了一个threshold 0.8 ，只有大于0.8 才判断是 cat,才保留box/window
                    cats_score.append(out[1])
                    # from (x, y)， we got the position. w, h determine the size of the window/box
                    cats_bbox.append((int(x * scale), int(y * scale), int(w * scale), int(h * scale)))
            else:
                if out[0] > 0.8:
                    dogs_score.append(out[0])
                    dogs_bbox.append((int(x * scale), int(y * scale), int(w * scale), int(h * scale)))
        
        # 由于在生产图像金字塔的时候 (i.e.def pyramid function里)，每一次的图片都缩小了1.5倍，
        # 这里从金字塔顶端往下处理图片的时候，keep track of this scale, 这样才能 用x * scale,  来还原box在原始图里面的位置
        scale = scale * 1.5
        
        
    return cats_score, cats_bbox, dogs_score, dogs_bbox
# 在原始的图片上 画出windows/boxes
def showWindows(image, dogs_bbox, cats_bbox):
    for (x, y, w, h) in dogs_bbox:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for (x, y, w, h) in cats_bbox:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(np.array(image).astype('uint8'))
    plt.show()
    
THRESHOLD = 0.2
def nms(scores, bboxes):
    sortedRes = sorted(zip(scores, bboxes), key=lambda x: x[0], reverse=True)
    scores = [score for (score, bbox) in sortedRes] # scores 就是PPT nms算法部分提到的置信度，倒序排序了
    bboxes = [bbox for (score, bbox) in sortedRes]
    
    suppress = [0] * len(scores) # 生成和scores数组样长的 0 数组 i.e. [0,0,0...0,0]
    result = [] # 返回的结果。里面存了没有被suppressed 的box
    
    """
        go through each box, decide whether we should suppress this box
        由于上面sortedRes 根据分数大小从大到小排序了， 所以第一box的分数最大（i.e. 置信度）
    """
    for i in range(len(bboxes)): 
        if suppress[i] == 1:
            continue
        
        area = bboxes[i][2] * bboxes[i][3] #box(x, y, w, h) get area from w * h
        result.append(bboxes[i]) # 用其他的box与上面第一个for loop 选中的box来比较
        for j in range(i+1, len(bboxes)):
            if suppress[j] == 1:
                continue
            
            otherArea = bboxes[j][2] * bboxes[j][3] # 计算面积
            
            # 就是比较坐标，得到两个box重叠部分坐标
            xx1 = np.maximum(bboxes[i][0], bboxes[j][0])
            yy1 = np.maximum(bboxes[i][1], bboxes[j][1])
            xx2 = np.minimum(bboxes[i][0] + bboxes[i][2] - 1, bboxes[j][0] + bboxes[j][2] - 1)
            yy2 = np.minimum(bboxes[i][1] + bboxes[i][3] - 1, bboxes[j][1] + bboxes[j][3] - 1)
            
            # 计算重叠部分的w, h 然后计算面积
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            #??? 根据nms 公式: inter / A并B, 这里为什么除以 min (A, B)
            ovr = inter / min(area, otherArea)
            
            if ovr > THRESHOLD:
                suppress[j] = 1
                
    return result
color_image, image = get_grey_and_color_image(test_all[1])
cats_score, cats_bbox, dogs_score, dogs_bbox = run_inference(image)
showWindows(color_image.copy(), dogs_bbox, cats_bbox)
new_dogs_bboxes = nms(dogs_score, dogs_bbox)
new_cats_bboxes = nms(cats_score, cats_bbox)
showWindows(color_image.copy(), new_dogs_bboxes, new_cats_bboxes)