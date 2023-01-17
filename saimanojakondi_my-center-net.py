#installing efficientnet library
!pip install efficientnet
#importing libraries
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from random import shuffle,choice,choices
from tensorflow.keras.layers import *
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau,LearningRateScheduler
from efficientnet import tfkeras as efn
from tensorflow.keras.models import Model
#train and test directories
train_dir = '../input/global-wheat-detection/train'
train_csv = '../input/global-wheat-detection/train.csv'
test_dir = '../input/global-wheat-detection/test'


n_epochs = 40
BATCH_SIZE = 2
DISPLAY = 1
lr = 0.001
INPUT_SIZE = 512
stride = 2
OUTPUT_SIZE = INPUT_SIZE//4
n_category = 1
output_layer_n = n_category + 4
train_df = pd.read_csv(train_csv)
train_df.head()
image_names = train_df['image_id'].unique()
#adding noise
def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.
        sigma = 1.0
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image/255 + gauss
        return noisy*255
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.4
        out = np.copy(image)
      # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[tuple(coords)] = 1

      # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[tuple(coords)] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy
#functions to extract the given data
def get_bb(string):
    bbox = []
    string = string[1:-1]
    strings = string.split(',')
    for char in strings:
        bbox.append(int(float(char)))
    return bbox
#random visualization of bboxes
def random_visualize(train_dir,image_names,train_df,k):
    k = k//2
    fig,ax = plt.subplots(k,2,figsize = (10,10))
    images = choices(image_names,k=k*2)

    for i,image in enumerate(images):
        
        img = cv2.imread(train_dir+'/'+image+'.jpg')
        ax[int(i//2)][int(i%2)].imshow(np.asarray(noisy('s&p',image=img)))
        bboxes = train_df[train_df['image_id']==image]['bbox'].to_list()
        for bbox in bboxes:
            bbox = get_bb(bbox)
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[0]+bbox[2]
            ymax = bbox[1]+bbox[3]
            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,255),4)

    plt.show()
random_visualize(train_dir,image_names,train_df,4)
'''
widths = []
heights = []
for image in image_names:
    image_dir = os.path.join(train_dir,image+'.jpg')
    img = cv2.imread(image_dir)
    im_w,im_h,c = img.shape[0],img.shape[1],img.shape[2]
    bboxes = train_df[train_df['image_id']==image]['bbox'].to_list()
    for bbox in bboxes:
        bbox = get_bb(bbox)
        widths.append(bbox[2]/im_w)
        heights.append(bbox[3]/im_h)
'''
'''
print(plt.hist(widths,bins=10)[0])
print(plt.hist(widths,bins=10)[1])
plt.show()
plt.hist(heights,bins=10)
plt.show()
'''
#generator for sequence
class My_generator(Sequence):
    def __init__(self,image_names,train_df,train_dir=train_dir,input_size = INPUT_SIZE,batch_size = BATCH_SIZE,is_train = False,stride = stride):
        self.image_names = image_names
        self.train_df = train_df
        self.stride = stride
        self.train_dir = train_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = self.input_size//self.stride
        self.is_train = is_train
        if self.is_train:
            self.on_epoch_end()
    def __len__(self):
        return int(np.ceil(len(self.image_names)/float(self.batch_size)))
    def on_epoch_end(self):
        if self.is_train:
            shuffle(image_names)
    def __getitem__(self,idx):
        batch_x = self.image_names[self.batch_size*idx:self.batch_size*(idx+1)]
        if self.is_train:
            return self.train_generator(batch_x)
        else:
            return self.valid_generator(batch_x)
    def train_generator(self,batch_x):
        X = []
        heatmaps = []
        Y = []
        output_height,output_width = self.output_size,self.output_size
        for image_name in list(batch_x):
            image_dir = os.path.join(self.train_dir,image_name+'.jpg')
            img = cv2.imread(image_dir)
            im_w,im_h,c = img.shape[0],img.shape[1],img.shape[2]
            img = cv2.resize(img,(self.input_size,self.input_size))
            bboxes = train_df[train_df['image_id']==image_name]['bbox'].to_list()
            output1 = np.zeros((self.output_size,self.output_size,output_layer_n+n_category))
            output2 = np.zeros((self.output_size,self.output_size,output_layer_n+n_category))
            output3 = np.zeros((self.output_size,self.output_size,output_layer_n+n_category))
            output4 = np.zeros((self.output_size,self.output_size,output_layer_n+n_category))

            for bbox in bboxes:
                bbox = get_bb(bbox)
                xc = bbox[0] + (bbox[2]/2)
                yc = bbox[1] + (bbox[3]/2)
                width = bbox[2]
                height = bbox[3]
                xc,yc,width,height = xc*(output_width/im_w),yc*(output_height/im_h),width*(output_width/im_w),height*(output_height/im_h)
                category = 0
                heatmap1=((np.exp(-(((np.arange(output_width)-xc)/(width/10))**2)/2)).reshape(1,-1)*(np.exp(-(((np.arange(output_height)-yc)/(height/10))**2)/2)).reshape(-1,1))
                output1[:,:,0] = np.maximum(heatmap1[:,:],output1[:,:,0])
                for i in range(n_category):
                    output1[int(yc//1),int(xc//1),i+1] = 1
                    output1[int(yc//1),int(xc//1),n_category+1] = yc%1
                    output1[int(yc//1),int(xc//1),n_category+2] = xc%1
                    output1[int(yc//1),int(xc//1),n_category+4] = width/output_width
                    output1[int(yc//1),int(xc//1),n_category+3] = height/output_height
                    
                xc2 =  xc 
                yc2 = self.output_size-yc
                heatmap2=((np.exp(-(((np.arange(output_width)-xc2)/(width/10))**2)/2)).reshape(1,-1)*(np.exp(-(((np.arange(output_height)-yc2)/(height/10))**2)/2)).reshape(-1,1))
                output2[:,:,0] = np.maximum(heatmap2[:,:],output2[:,:,0])
                for i in range(n_category):
                    output2[int(yc2//1),int(xc2//1),i+1] = 1
                    output2[int(yc2//1),int(xc2//1),n_category+1] = yc2%1
                    output2[int(yc2//1),int(xc2//1),n_category+2] = xc2%1
                    output2[int(yc2//1),int(xc2//1),n_category+4] = width/output_width
                    output2[int(yc2//1),int(xc2//1),n_category+3] = height/output_height
                    
                xc3 = self.output_size - xc 
                yc3 = yc
                heatmap3=((np.exp(-(((np.arange(output_width)-xc3)/(width/10))**2)/2)).reshape(1,-1)*(np.exp(-(((np.arange(output_height)-yc3)/(height/10))**2)/2)).reshape(-1,1))
                output3[:,:,0] = np.maximum(output3[:,:,0],heatmap3[:,:])
                for i in range(n_category):
                    output3[int(yc3//1),int(xc3//1),i+1] = 1
                    output3[int(yc3//1),int(xc3//1),n_category+1] = yc3%1
                    output3[int(yc3//1),int(xc3//1),n_category+2] = xc3%1
                    output3[int(yc3//1),int(xc3//1),n_category+4] = width/output_width
                    output3[int(yc3//1),int(xc3//1),n_category+3] = height/output_height

                xc4 = self.output_size-xc 
                yc4 = self.output_size-yc
                heatmap4=((np.exp(-(((np.arange(output_width)-xc4)/(width/10))**2)/2)).reshape(1,-1)*(np.exp(-(((np.arange(output_height)-yc4)/(height/10))**2)/2)).reshape(-1,1))
                output4[:,:,0] = np.maximum(output4[:,:,0],heatmap4[:,:])
                for i in range(n_category):
                    output4[int(yc4//1),int(xc4//1),i+1] = 1
                    output4[int(yc4//1),int(xc4//1),n_category+1] = yc4%1
                    output4[int(yc4//1),int(xc4//1),n_category+2] = xc4%1
                    output4[int(yc4//1),int(xc4//1),n_category+4] = width/output_width
                    output4[int(yc4//1),int(xc4//1),n_category+3] = height/output_height
            image2 = cv2.flip(img,0)
            image3 = cv2.flip(img,1)
            image4 = cv2.flip(img,-1)
            X.append(image2)
            X.append(img)
            X.append(image3)
            X.append(image4)
            Y.append(output2)
            Y.append(output1)
            Y.append(output3)
            Y.append(output4)
        X = np.asarray(X, np.float32)/255
        Y = np.asarray(Y, np.float32)
        return X,Y
    def valid_generator(self,batch_x):
        X = []
        heatmaps = []
        Y = []
        output_height,output_width = self.output_size,self.output_size
        for image_name in list(batch_x):
            image_dir = os.path.join(self.train_dir,image_name+'.jpg')
            img = cv2.imread(image_dir)
            im_w,im_h,c = img.shape[0],img.shape[1],img.shape[2]
            img = cv2.resize(img,(self.input_size,self.input_size))
            bboxes = train_df[train_df['image_id']==image_name]['bbox'].to_list()
            output1 = np.zeros((self.output_size,self.output_size,output_layer_n+n_category))
            output2 = np.zeros((self.output_size,self.output_size,output_layer_n+n_category))
            output3 = np.zeros((self.output_size,self.output_size,output_layer_n+n_category))
            output4 = np.zeros((self.output_size,self.output_size,output_layer_n+n_category))

            for bbox in bboxes:
                bbox = get_bb(bbox)
                xc = bbox[0] + (bbox[2]/2)
                yc = bbox[1] + (bbox[3]/2)
                width = bbox[2]
                height = bbox[3]
                xc,yc,width,height = xc*(output_width/im_w),yc*(output_height/im_h),width*(output_width/im_w),height*(output_height/im_h)
                category = 0
                heatmap1=((np.exp(-(((np.arange(output_width)-xc)/(width/10))**2)/2)).reshape(1,-1)*(np.exp(-(((np.arange(output_height)-yc)/(height/10))**2)/2)).reshape(-1,1))
                output1[:,:,0] = np.maximum(heatmap1[:,:],output1[:,:,0])
                for i in range(n_category):
                    output1[int(yc//1),int(xc//1),i+1] = 1
                    output1[int(yc//1),int(xc//1),n_category+1] = yc%1
                    output1[int(yc//1),int(xc//1),n_category+2] = xc%1
                    output1[int(yc//1),int(xc//1),n_category+4] = width/output_width
                    output1[int(yc//1),int(xc//1),n_category+3] = height/output_height
            X.append(img)
            Y.append(output1)
        X = np.asarray(X, np.float32)/255
        Y = np.asarray(Y, np.float32)
        #print(X.shape,Y.shape)
        return X,Y
#testing the generator
def test(i):
    mygen = My_generator(image_names,train_df,train_dir=train_dir,batch_size=1, is_train = True)
    #print(mygen)
    X,Y = mygen.__getitem__(1)
    X = X[i]
    Y = Y[i]
    X = np.asarray(np.ceil(X*255) ,np.uint8)
    heatmap = Y[:,:,0]
    points = np.argwhere(Y[:,:,1]==1)
    #print(points)
    for y,x in points:
        offy = Y[y,x,2]
        offx = Y[y,x,3]
        width = Y[y,x,5]*(INPUT_SIZE/stride)
        height = Y[y,x,4]*(INPUT_SIZE/stride)
        xc = x+offx
        yc = y+offy
        xmin = int((xc-(width/2))*stride)
        ymin = int((yc-(height/2))*stride)
        xmax = int((xc+(width/2))*stride)
        ymax = int((yc+(height/2))*stride)
        cv2.rectangle(X, (xmin, ymin), (xmax, ymax), (0,255,255), 2)
            #cv2.circle(X, (int(xc*4),int(yc*4)), 5, (0,0,255), 2) 
        #cv2.imshow('djpg',y[:,:,1]*255)
        #cv2.imshow('drawjpg',x)
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))


        #print(X.shape)
    ax[0].imshow(X)
    ax[1].imshow(heatmap)


test(0)
test(1)
test(2)
test(3)
#fpn block and weighted-fpn block
def build_fpn(features,num_channels,wbifpn,kernel_size=2):
    p4,p5,p6,p7 = features
    #column1
    p6 = Conv2D(num_channels,(3,3),kernel_initializer='glorot_uniform',activation='relu',padding='same')(p6)
    p5 = Conv2D(num_channels,(3,3),kernel_initializer='glorot_uniform',activation='relu',padding='same')(p5)
    p4 = Conv2D(num_channels,(3,3),kernel_initializer='glorot_uniform',activation='relu',padding='same')(p4)
    
    p7 = Conv2D(num_channels,(3,3),kernel_initializer='glorot_uniform',activation='relu',padding='same')(p7)
    p7_resize = BatchNormalization()(p7)
    p7_resize = MaxPool2D((kernel_size,kernel_size))(p7_resize)
    if wbifpn:
        p6_td = Fuse()([p6,p7_resize])
    else:
        p6_td = Add()([p6,p7_resize])
    p6_td = Conv2D(num_channels,(3,3),kernel_initializer = 'glorot_uniform',activation='relu',padding='same')(p6_td)
    p6_td = BatchNormalization()(p6_td)
    p6_td = MaxPool2D((2,2),padding = 'same',strides = 1)(p6_td)
    p6_td_resize = Conv2D(num_channels,(3,3),kernel_initializer='glorot_uniform',activation='relu',padding='same')(p6_td)
    p6_td_resize = BatchNormalization()(p6_td_resize)
    
    p6_td_resize = MaxPool2D((kernel_size,kernel_size))(p6_td_resize) 
    if wbifpn:
        p5_td = Fuse()([p5,p6_td_resize])
    else:
        p5_td = Add()([p5,p6_td_resize])
    p5_td = Conv2D(num_channels,(3,3),kernel_initializer='glorot_uniform',activation='relu',padding='same')(p5_td)
    p5_td = BatchNormalization()(p5_td)
    p5_td = MaxPool2D((2,2),padding='same',strides = 1)(p5_td)
    p5_td_resize = Conv2D(num_channels,(3,3),kernel_initializer='glorot_uniform',activation='relu',padding='same')(p5_td)
    p5_td_resize = BatchNormalization()(p5_td_resize)
    p5_td_resize = MaxPooling2D((kernel_size,kernel_size))(p5_td_resize)
    if wbifpn:
        p4_td = Fuse()([p4,p5_td_resize])
    else:
        p4_td = Add()([p4,p5_td_resize])
    p4_td = Conv2D(num_channels,(3,3),kernel_initializer='glorot_uniform',activation='relu',padding='same')(p4_td)
    p4_td = MaxPool2D((2,2),padding='same',strides = 1)(p4_td)
    p4_U = Conv2D(num_channels,(3,3),kernel_initializer='glorot_uniform',activation='relu',padding='same')(p4_td)
    p4_U = BatchNormalization()(p4_U)
    p5_U = UpSampling2D((kernel_size,kernel_size))(p4_U)
    if wbifpn:
        p5_U = Fuse()([p5,p5_td,p5_U])
    else:
        p5_U = Add()([p5,p5_td,p5_U])
    p5_U = Conv2D(num_channels,(3,3),kernel_initializer='glorot_uniform',activation='relu',padding='same')(p5_U)
    p5_U = BatchNormalization()(p5_U)
    p6_U = UpSampling2D((kernel_size,kernel_size))(p5_U)
    if wbifpn:
        p6_U = Fuse()([p6,p6_td,p6_U])
    else:
        p6_U = Add()([p6,p6_td,p6_U])
    p6_U = Conv2D(num_channels,(3,3),kernel_initializer='glorot_uniform',activation='relu',padding='same')(p6_U)
    p6_U = BatchNormalization()(p6_U)
    p7_U = UpSampling2D((kernel_size,kernel_size))(p6_U)
    if wbifpn:
        p7_U = Fuse()([p7,p7_U])
    else:
        p7_U = Add()([p7,p7_U])
    p7_U = Conv2D(num_channels,(3,3),kernel_initializer='glorot_uniform',activation='relu',padding='same')(p7_U)
    p7_U = BatchNormalization()(p7_U)
    return (p4_U,p5_U,p6_U,p7_U)


class Fuse(tf.keras.layers.Layer):
    '''Fusion layer'''
    def __init__(self, epsilon=1e-4, **kwargs):
        super(Fuse, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                 shape=(num_in,),
                                 initializer=tf.keras.initializers.constant(1 / num_in),
                                 trainable=True,
                                 dtype=tf.float32)

    def call(self, inputs, **kwargs):
        w = tf.keras.activations.relu(self.w)
        x = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis=0)
        x = x / (tf.reduce_sum(w) + self.epsilon)
        return x
    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(Fuse, self).get_config()
        config.update({
            'epsilon': self.epsilon
        })
        return config
    
def create_model(input_shape ,wbifpn=False):
    '''model'''
    effnet = efn.EfficientNetB4(input_shape=input_shape,weights=None,include_top = False)
    p4 = effnet.get_layer('block2a_activation').output
    p5 = effnet.get_layer('block3a_activation').output
    p6 = effnet.get_layer('block4a_activation').output
    p7 = effnet.get_layer('block7a_activation').output
    features = (p7,p6,p5,p4)
    features = build_fpn(features,16,wbifpn)
    features = build_fpn(features,32,wbifpn)
    features = build_fpn(features,64,wbifpn)
    features = build_fpn(features,81,wbifpn)
    features = list(features)

    for i in range(1,4):
        feature_curr = features[i]
        feature_past = features[i-1]
        feature_past_up = UpSampling2D((2,2))(feature_past)
        feature_past_up = Conv2D(81,(3,3),padding='same',activation='relu',kernel_initializer='glorot_uniform')(feature_past_up)
        if wbifpn:
            feature_final = Fuse(name='final{}'.format(str(i)))([feature_curr,feature_past_up])
        else:
            feature_final = Add(name='final{}'.format(str(i)))([feature_curr,feature_past_up])
        features[i] = feature_final
    if stride == 2:
        features[-1] = UpSampling2D((2,2))(features[-1])
        features[-1] = Conv2D(128,(3,3),activation='relu',padding='same',kernel_initializer='glorot_uniform')(features[-1])
    out = Conv2D(5,(3,3),activation='sigmoid',kernel_initializer='glorot_uniform',padding='same')(features[-1])
    zeros = tf.expand_dims(tf.zeros_like(out[...,0]),axis=-1)
    out_concat = tf.concat([zeros,out],axis = -1)
    prediction_model=tf.keras.models.Model(inputs=[effnet.input],outputs=out)
    model = Model(inputs = [effnet.input],outputs = out_concat)
    return model,prediction_model
model,prediction_model = create_model(input_shape=(512,512,3))
model.summary()
def focal_loss(gamma,gamma2,y_true,y_pred,heatmaps):
    '''focal loss'''
    y_pred = K.clip(y_pred,1e-5,1-1e-5)
    loglik = 2*y_true*((1-y_pred)**gamma)*K.log(y_pred) +(1-y_true)*((1-heatmaps)**gamma2)*(y_pred**gamma)*K.log(1-y_pred)
    #heatloss=-K.sum(heatmap_true*((1-heatmap_pred)**alpha)*K.log(heatmap_pred+1e-6)+(1-heatmap_true)*((1-heatmap_true_rate)**beta)*(heatmap_pred**alpha)*K.log(1-heatmap_pred+1e-6))

    cls_loss = -K.sum(loglik,axis=0)
    return cls_loss

def loss_fn(gamma1,gamma2):
    '''final combined loss and supporting functions'''
    def final_loss(y_true,y_pred):
        mask = K.batch_flatten(K.sign(y_true[...,4]))
        N = K.sum(mask)
        #heatmap loss
        heatmaps = K.batch_flatten(y_true[...,0])
        cls_pred = K.batch_flatten(y_pred[...,1])
        cls_pred = K.clip(cls_pred,1e-7,1-1e-7)
        cls_true = K.batch_flatten(y_true[...,1])

        cls_loss = K.sum(focal_loss(gamma1,gamma2,cls_true,cls_pred,heatmaps))/N
        
        #offset loss
        offy_pred = K.batch_flatten(y_pred[...,2])
        offx_pred = K.batch_flatten(y_pred[...,3])
        offy_true = K.batch_flatten(y_true[...,2])
        offx_true = K.batch_flatten(y_true[...,3])
        offloss = K.abs(offx_pred*mask-offx_true) + K.abs(offy_pred*mask-offy_true)
        offloss = K.sum(offloss)/N
        
        #size loss
        sizey_true = K.batch_flatten(y_true[...,4])
        sizey_pred = K.batch_flatten(y_pred[...,4])
        sizex_true = K.batch_flatten(y_true[...,5])
        sizex_pred = K.batch_flatten(y_pred[...,5])
        y_mask = tf.cast(sizey_pred>0.4,dtype=tf.float32)
        x_mask = tf.cast(sizex_pred>0.4,dtype=tf.float32)
        y_weight = y_mask*sizey_pred
        x_weight = x_mask*sizex_pred
        '''loss is penalized by 1+x_weight, 1+y_weight'''
        size_loss = K.sum(K.abs(sizex_pred*mask-sizex_true)*(1+x_weight)+K.abs(sizey_pred*mask-sizey_true)*(1+y_weight))/N 
        loss = (1.5*cls_loss+1*offloss+10*size_loss)
        return loss
    return final_loss
def cls_metric(gamma1,gamma2):
    def closs(y_true,y_pred):
        mask = K.batch_flatten(K.sign(y_true[...,4]))
        N = K.sum(mask)
        heatmaps = K.batch_flatten(y_true[...,0])
        cls_pred = K.batch_flatten(y_pred[...,1])
        cls_pred = K.clip(cls_pred,1e-7,1-1e-7)
        cls_true = K.batch_flatten(y_true[...,1])
        cls_loss = K.sum(focal_loss(gamma1,gamma2,cls_true,cls_pred,heatmaps))/N
        return cls_loss
    return closs
    
def off_loss(y_true,y_pred):
    mask = K.batch_flatten(K.sign(y_true[...,4]))
    N = K.sum(mask)
    offy_pred = K.batch_flatten(y_pred[...,2])
    offx_pred = K.batch_flatten(y_pred[...,3])
    offy_true = K.batch_flatten(y_true[...,2])
    offx_true = K.batch_flatten(y_true[...,3])
    offloss = K.abs(offx_pred*mask-offx_true) + K.abs(offy_pred*mask-offy_true)
    offloss = K.sum(offloss)/N
    return offloss
def size_metric():
    def sloss(y_true,y_pred):
        mask = K.batch_flatten(K.sign(y_true[...,4]))
        N = K.sum(mask)
        sizey_true = K.batch_flatten(y_true[...,4])
        sizey_pred = K.batch_flatten(y_pred[...,4])
        sizex_true = K.batch_flatten(y_true[...,5])
        sizex_pred = K.batch_flatten(y_pred[...,5])
        y_mask = tf.cast(sizey_pred>0.4,dtype=tf.float32)
        x_mask = tf.cast(sizex_pred>0.4,dtype=tf.float32)
        y_weight = y_mask*sizey_pred
        x_weight = x_mask*sizex_pred
        size_loss = K.sum(K.abs(sizex_pred*mask-sizex_true)*(1+x_weight)+K.abs(sizey_pred*mask-sizey_true)*(1+y_weight))/N#size_loss2 = K.sum(K.abs(sizex_pred*mask-sizex_true)*(1-sizex_true)*mask+K.abs(sizey_pred*mask-sizey_true)*(1-sizey_true)*mask,axis=-1)
        #size_loss = 0.8*size_loss1+0.2*size_loss2
        return size_loss
    return sloss
def build_model(input_shape,gamma1=1.5,gamma2=3.0,lr=lr):
    model,_ = create_model(input_shape)
    optimizer = Adam(lr = lr)
    model.compile(loss = loss_fn(gamma1,gamma2),optimizer = optimizer,metrics = [off_loss,size_metric(),cls_metric(gamma1,gamma2)])
    return model
#learning rate scheduler
def lrs(epoch):
    lr = 0.001
    if epoch >= 20: lr = 0.0002
    return lr

lr_schedule = LearningRateScheduler(lrs)
early_stopping = EarlyStopping(monitor = 'val_loss', min_delta=0, patience = 5, verbose = 1)
print('no of available datapoints : {}'.format(4*len(image_names)))
#splitting
curT = image_names[300:]
curV = image_names[:300]
#generators
train_gen = My_generator(curT,train_df,batch_size = BATCH_SIZE,is_train=True)
val_gen = My_generator(curV,train_df,batch_size = BATCH_SIZE,is_train = False)
model = build_model((INPUT_SIZE,INPUT_SIZE,3))
STEPS_PER_EPOCH = curT.shape[0]//(BATCH_SIZE)
name = 'trained.h5'
checkpoint = ModelCheckpoint(name,monitor = 'val_loss', save_best_only = True, verbose = 1, period = 1)
#training
history = model.fit_generator(train_gen,epochs = n_epochs ,verbose = DISPLAY,steps_per_epoch = STEPS_PER_EPOCH,validation_data = val_gen,shuffle = True,validation_steps = curV.shape[0]//(BATCH_SIZE),callbacks = [lr_schedule,checkpoint,early_stopping])
