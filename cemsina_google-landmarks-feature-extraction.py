import numpy as np 
import pandas as pd
from keras.optimizers import Adam
from keras.layers import *
from keras.models import *
import keras.preprocessing.image as image
from keras.utils import to_categorical
import math
from tqdm.notebook import tqdm
from keras.applications.densenet import DenseNet121
from keras import backend as K
import threading
import gc
import tensorflow as tf
IMG_WIDTH  = 224
IMG_HEIGHT = 224
READ_SIZE = 100
train = pd.read_csv('../input/landmark-recognition-2020/train.csv')
test = pd.read_csv('../input/landmark-recognition-2020/sample_submission.csv')
w = open("google_landmark_features.csv","w")
train_ids = train.id.values
test_ids = test.id.values
def get_data_X(image_name,folder):
    img = image.load_img(f'../input/landmark-recognition-2020/{folder}/{image_name[0]}/{image_name[1]}/{image_name[2]}/{image_name}.jpg',target_size=(IMG_WIDTH, IMG_HEIGHT))
    X = image.img_to_array(img) / 255
    img.close()
    return np.array(X)
inp = Input((IMG_WIDTH,IMG_HEIGHT,3))
x = DenseNet121(input_tensor=inp, include_top=False)
x = x.output
x = GlobalAveragePooling2D()(x)
x = Lambda(lambda x: K.expand_dims(x,axis=-1))(x)
x = AveragePooling1D(4)(x) # feature count reduction
out = Lambda(lambda x: x[:,:,0])(x)
model = Model(inp,out)    

def append_to_file(ids,folder):
    X = np.array([get_data_X(idx,folder) for idx in ids])
    y = model.predict(X)
    for i,idx in enumerate(ids):
        features = idx+","+",".join([str(e) for e in y[i]])
        w.write(features+"\n")
    w.flush()
    gc.collect()
    return

def chunks(arr, n):
    n = max(1, n)
    return [arr[i:i+n] for i in range(0, len(arr), n)]

w.write(',')
w.write(','.join([str(i) for i in range(0,256)])+"\n")

train_chunks = chunks(train_ids, READ_SIZE)
test_chunks = chunks(test_ids, READ_SIZE)

for chunk in tqdm(train_chunks,desc="Train"):
    p = threading.Thread(target=append_to_file, args=(chunk,"train"))
    p.start()
    p.join()

for chunk in tqdm(test_chunks,desc="Test"):
    p = threading.Thread(target=append_to_file, args=(chunk,"test"))
    p.start()
    p.join()

w.close()
print("OK")