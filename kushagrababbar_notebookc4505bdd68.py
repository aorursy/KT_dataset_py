import tensorflow as tf
import keras
from keras.layers import Input , Conv2D , MaxPool2D , BatchNormalization , LeakyReLU , GlobalAveragePooling2D  , UpSampling2D
from keras.layers import Dense , Flatten , add , Concatenate
from keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

def focal_loss(true, pred, shape, eps=10e-10, alpha=0.75, gama=2.0):
    b, h, w, c = shape
    fl = - tf.math.log(tf.clip_by_value(pred, eps, 1.)) * true * alpha * tf.pow(1 - pred, gama)             - tf.math.log(tf.clip_by_value(1 - pred, eps, 1.)) * (1 - true) * (1 - alpha) * tf.pow(pred, gama)
    fl = tf.reshape(fl, (b, h * w * c))
    fl = tf.reduce_sum(fl, axis=-1)
    return fl

def l2(true, pred, shape):
    b, h, w, c = shape
    verts = tf.reshape(tf.square(true-pred), (b, h*w*c))
    verts = tf.reduce_sum(verts, 1)
    return verts

def focal_multi_class_entropy_loss(true, pred, shape, eps=10e-10, alpha=0.75, gama=2.0):
    b, h, w, c = shape
    cl = tf.clip_by_value(pred, eps, 1.)
    cl = -tf.math.log(cl) * true * tf.pow(1 - cl, gama) * alpha
    cl = tf.reshape(cl, (b, h * w * c))
    cl = tf.reduce_sum(cl, axis=1)
    return cl

def total_loss(Y_true, Y_pred):
    
    shape = tf.shape(Y_true)
    b , h , w = shape[0] , shape[1] , shape[2]
    y_true_loc = tf.reshape(Y_true[..., 0], (b, h, w, 1))
    y_pred_loc = tf.reshape(Y_pred[..., 0], (b, h, w, 1))
    y_true_vertex = Y_true[..., 1:5]
    y_pred_vertex = Y_pred[..., 1:5]
    #y_true_class = Y_true[..., 5:]
    #y_pred_class = Y_pred[..., 5:]
    region_lp = tf.reshape(y_true_loc, (b, h, w, 1))
    
    loc_loss = focal_loss(y_true_loc, y_pred_loc, (b, h, w, 1)) * 1.0
    vertex_loss = l2(y_true_vertex * region_lp, y_pred_vertex * region_lp, (b, h, w, 4)) * 1.0
    #class_loss = focal_multi_class_entropy_loss(y_true_class, y_pred_class, (b, h, w, 2)) * 1.0
    
    return loc_loss + vertex_loss #+ class_loss

def bn_and_activation_layer(x):
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    return x

def res_block(x , out_ch ):
    if list(x.shape)[-1] == out_ch:
        res = x
    else :
        res = Conv2D(out_ch , kernel_size=(1,1) , padding = 'same')(x)
        res = bn_and_activation_layer(res)
    x = Conv2D(int(out_ch / 2) , kernel_size=(1,1) , padding = 'same')(x)
    x = bn_and_activation_layer(x)
    x = Conv2D(int(out_ch / 2) , kernel_size=(3,3) , padding = 'same')(x)
    x = bn_and_activation_layer(x)
    x = Conv2D(out_ch , kernel_size=(1,1) , padding = 'same')(x)
    x = BatchNormalization()(x)
    x = add([res , x])
    x = LeakyReLU()(x)
    return x

def hourglass(x):
    ch = list(x.shape)[-1]
    
    d1 = res_block(x , ch)
    y = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(d1)
    d2 = res_block(y , ch)
    y = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(d2)
    d3 = res_block(y , ch)
    y = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(d3)
    d4 = res_block(y , ch)
    
    u4 = res_block(d4 , ch)
    u4 = res_block(u4 , ch)
    u4 = res_block(u4 , ch)
    d4 = res_block(d4 , ch)
    u4 = add([u4 , d4])
    u3 = UpSampling2D()(u4)
    d3 = res_block(d3 , ch)
    u3 = add([u3 , d3])
    u3 = res_block(u3 , ch)
    u2 = UpSampling2D()(u3)
    d2 = res_block(d2 , ch)
    u2 = add([u2 , d2])
    u2 = res_block(u2 , ch)
    u1 = UpSampling2D()(u2)
    d1 = res_block(d1 , ch)
    u1 = add([u1 , d1])
    u1 = res_block(u1 , ch)
    
    y = Conv2D(ch , kernel_size=(1,1) , padding = 'same')(u1)
    y = bn_and_activation_layer(y)
    
    return x , y

def join_hourglass(x , y ):
    ch = list(x.shape)[-1]
    y1 = Conv2D(ch , kernel_size=(1,1) , activation='linear' , padding = 'same')(y)
    y2 = Conv2D(ch , kernel_size=(1,1) , activation='linear' , padding = 'same')(y)
    y2 = Conv2D(ch , kernel_size=(1,1) , activation='linear' , padding = 'same')(y2)
    x = add([y1 , y2 ,x])
    return x

def create_heads(out):
    head_loc = Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')(out)
    head_vertex = Conv2D(4, kernel_size=(3, 3), activation='linear', padding='same')(out)
    #head_class = Conv2D(2, kernel_size=(3, 3), activation='softmax', padding='same')(out)
    out = Concatenate(3)([head_loc, head_vertex])#, head_class])
    return out

def model(out_channels = 256 , no_of_hgs = 2 , inp = Input((None,None,3))):
    x = Conv2D(64 , kernel_size=(7,7) , strides=(2,2)  , padding = 'same')(inp)
    x = bn_and_activation_layer(x)
    x = res_block(x , out_channels//2 )
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = res_block(x , out_channels//2 )
    x = res_block(x , out_channels )
    out = None
    for i in range(no_of_hgs):
        x , out = hourglass(x)
        if i<no_of_hgs-1:
            x = join_hourglass(x , out)
    out = create_heads(out)
    model = Model(inp , out)
    loss = total_loss
    #lr = 0.000001
    #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(lr , decay_steps=20000 , decay_rate = 0.96 , staircase = True)
    #opt = Adam(lr_schedule)
    model.compile(loss=loss, optimizer='adam')#opt)
    #model.summary()
    return model
def iou(box1, box2):
    
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2
        
    xi1 = max(box1_x1,box2_x1)
    yi1 = max(box1_y1,box2_y1)
    xi2 = min(box1_x2,box2_x2)
    yi2 = min(box1_y2,box2_y2)
    inter_width = xi2-xi1
    inter_height = yi2-yi1
    inter_area = max(inter_height, 0)*max(inter_width, 0)
    
    box1_area = (box1_y2 - box1_y1)*(box1_x2 - box1_x1) 
    box2_area = (box2_y2 - box2_y1)*(box2_x2 - box2_x1)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area

    return iou

def nms(detections, thresh=.5):

    if len(detections) == 0:
        return []
    
    detections = sorted(detections, key=lambda detections: detections[4],
            reverse=True)
    
    new_detections=[]
    new_detections.append(detections[0])
    del detections[0]
    for index, detection in enumerate(detections):
        for new_detection in new_detections:
            if iou(detection[:4], new_detection[:4]) > thresh:
                del detections[index]
                break
        else:
            new_detections.append(detection)
            del detections[index]
    return new_detections
inp = Input((256 , 256 , 3))
lp = model(256 , 3 , inp)
lp.summary()
lp.load_weights('../input/weights-file/lp300000.h5')
import os
import cv2
paths = os.listdir('../input/global-wheat-detection/test/')
def test(img):
    pred = lp.predict(np.expand_dims(img , 0))[0]
    def scale_points(box , scale1 , scale2):
            pts = [int(int(box[0])/scale2*scale1) , int(int(box[1])/scale2*scale1) , int(int(box[2])/scale2*scale1) , int(int(box[3])/scale2*scale1)]
            return np.array(pts)
    objs=[]
    for y in range(pred.shape[0]):
        for x in range(pred.shape[1]):
            prob = pred[y,x,0]
            if prob>.1:
                pts = pred[y,x,1:5]
                pts = scale_points(pts, 256 , 64)
                for i in range(4):
                    pts[i] = np.clip(pts[i], 0, 256)
                objs.append(np.array([pts[0] , pts[1] , pts[2] , pts[3] ,prob]))          
    objs = np.array(objs)
    if len(objs)>0:
        aft_nms = np.array(nms(objs,0.0))
        return aft_nms
results=[]
for path in paths:
    img = cv2.imread('../input/global-wheat-detection/test/'+path)
    img = cv2.resize(img , (256,256))
    box_scores = test(img/255.0)
    ind = np.argsort(-box_scores[:,4])
    box_scores = box_scores[ind]
    sub=[]
    for box in box_scores:
        sub.append(f'{box[4]:.2f} {int(box[0])*4} {int(box[1])*4} {int(box[2] - box[0])*4} {int(box[3] - box[1])*4}')
    sub = " ".join(sub)
    result={'image_id':path[:-4] , 'PredictionString': sub}
    results.append(result)
import pandas as pd
test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
test_df.head()
test_df.to_csv('submission.csv', index=False)
results
