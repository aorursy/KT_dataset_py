import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from skimage.measure import regionprops
class BoundBox:
    def __init__(self, class_num, x=0., y=0., w = 0., h = 0., c = None):
        self.x, self.y, self.w, self.h, self.c = x,y,w,h,c if c is not None else 0.
        self.probs = np.zeros((class_num,))
        if c is not None:
            self.probs[c] = 1
        self.normed = False
        
    def norm_dims(self, image):
        if not self.normed:
            self.x, self.y, self.w, self.h = self.x/image.shape[1], self.y/image.shape[0], self.w/image.shape[1], self.h/image.shape[0]
        self.normed = True
    
    def unnorm_dims(self, image):
        if self.normed:
            self.x, self.y, self.w, self.h = self.x*image.shape[1], self.y*image.shape[0], self.w*image.shape[1], self.h*image.shape[0]
        self.normed = False
    
    def iou(self, box):
        intersection = self.intersect(box)
        union = self.w*self.h + box.w*box.h - intersection
        return intersection/union
        
    def intersect(self, box):
        width  = self.__overlap([self.x-self.w/2, self.x+self.w/2], [box.x-box.w/2, box.x+box.w/2])
        height = self.__overlap([self.y-self.h/2, self.y+self.h/2], [box.y-box.h/2, box.y+box.h/2])
        return width * height
        
    def __overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2,x4) - x3
            

def sigmoid(x):
    return 1. / (1.  + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)           
%matplotlib inline
base_x, base_y = 128, 128
# base_x, base_y = 416, 416 # original size
MAX_OBJ = 8
CLASS_NUM = 4
class_color = lambda class_lab: plt.cm.nipy_spectral((class_lab+1)/(CLASS_NUM+1))[:3]
def generate_image(make_bbox_img = False, threshold = 0.2):
    base_img = np.ones((base_x, base_y, 3))
    
    bbox_list = []
    rng_func = lambda n: np.linspace(-1,1,n)
    xx, yy = np.meshgrid(rng_func(base_img.shape[0]),
               rng_func(base_img.shape[1]))
    for i in range(1+np.random.choice(range(MAX_OBJ))):
        xc, yc = np.random.uniform(-1.1,1.1, size = 2)
        rc = np.random.uniform(0.1, 0.5)
        c_class = np.random.choice(range(CLASS_NUM))
        mask_img = np.power(xx-xc,2)+np.power(yy-yc,2) < np.power(rc,2)
        
        base_img[mask_img] = class_color(c_class)
        c_prop = regionprops(mask_img.astype(int))
        if len(c_prop)>0:
            c_bbox = c_prop[0].bbox
            ymin, xmin, ymax, xmax = c_bbox
            bbox_list += [BoundBox(class_num = CLASS_NUM, 
                                   x = 0.5*(xmin+xmax),
                                   y = 0.5*(ymin+ymax),
                                   w = xmax-xmin,
                                   h = ymax-ymin,
                                   c = c_class)]
    
    bbox_img = base_img.copy()
    if make_bbox_img:
        # draw the boxes using a threshold
        for box in bbox_list:
            box.norm_dims(bbox_img)
            max_indx = np.argmax(box.probs)
            max_prob = box.probs[max_indx]
            print(max_indx, max_prob)

            if max_prob > threshold:
                xmin  = int((box.x - box.w/2) * bbox_img.shape[1])
                xmax  = int((box.x + box.w/2) * bbox_img.shape[1])
                ymin  = int((box.y - box.h/2) * bbox_img.shape[0])
                ymax  = int((box.y + box.h/2) * bbox_img.shape[0])

                cv2.rectangle(bbox_img, (xmin,ymin), (xmax,ymax), class_color(max_indx), 2)
                cv2.putText(bbox_img, '%d' % max_indx, 
                            (xmin, ymin - 12), 0, 
                            4e-3 * bbox_img.shape[0], 
                            (0,255,0), 2)

    return base_img, bbox_list, bbox_img
plt.imshow(generate_image(True,-0.1)[2])
from keras.models import Sequential
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD
import tensorflow as tf

%matplotlib inline
SCALE_NOOB, SCALE_CONF, SCALE_COOR, SCALE_PROB = 0.5, 5.0, 5.0, 1.0
BATCH_SIZE = 8
BOX = 5
ANCHORS = '1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52'
ANCHORS = [float(ANCHORS.strip()) for ANCHORS in ANCHORS.split(',')]
NORM_H, NORM_W = base_x, base_y
model = Sequential()

# Layer 1
model.add(Conv2D(16, (3,3), strides=(1,1), padding='same', use_bias=False, input_shape=(NORM_H,NORM_W,3)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2 - 5
for i in range(0,4):
    model.add(Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 6
model.add(Conv2D(512, (3,3), strides=(1,1), padding='same', use_bias=False))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same'))

# Layer 7 - 8
for _ in range(0,2):
    model.add(Conv2D(1024, (3,3), strides=(1,1), padding='same', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.1))

# Layer 9
model.add(Conv2D(BOX * (4 + 1 + CLASS_NUM), (1, 1), strides=(1, 1), kernel_initializer='he_normal'))
model.add(Activation('linear'))
_, GRID_H, GRID_W, _ = model.get_output_shape_at(0)
model.add(Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS_NUM)))
model.summary()
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
def show_net(in_mod, name):
    out_model = model_to_dot(in_mod, show_shapes=True, show_layer_names=True)
    out_model.write_svg(name)
    return SVG(name)
try:
    show_net(model, 'yolo.svg')
except:
    pass
def interpret_netout(image, netout, threshold):
    boxes = []

    # interpret the output by the network
    for row in range(GRID_H):
        for col in range(GRID_W):
            for b in range(BOX):
                box = BoundBox(CLASS_NUM)

                # first 5 weights for x, y, w, h and confidence
                box.x, box.y, box.w, box.h, box.c = netout[row,col,b,:5]

                box.x = (col + sigmoid(box.x)) / GRID_W
                box.y = (row + sigmoid(box.y)) / GRID_H
                box.w = ANCHORS[2 * b + 0] * np.exp(box.w) / GRID_W
                box.h = ANCHORS[2 * b + 1] * np.exp(box.h) / GRID_H
                box.c = sigmoid(box.c)

                # last 20 weights for class likelihoods
                classes = netout[row,col,b,5:]
                box.probs = softmax(classes) * box.c
                box.probs *= box.probs > threshold

                boxes.append(box)

    # suppress non-maximal boxes
    for c in range(CLASS_NUM):
        sorted_indices = list(reversed(np.argsort([box.probs[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            if boxes[index_i].probs[c] == 0: 
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    
                    if boxes[index_i].iou(boxes[index_j]) >= 0.4:
                        boxes[index_j].probs[c] = 0

    # draw the boxes using a threshold
    for box in boxes:
        max_indx = np.argmax(box.probs)
        max_prob = box.probs[max_indx]
        
        if max_prob > threshold:
            xmin  = int((box.x - box.w/2) * image.shape[1])
            xmax  = int((box.x + box.w/2) * image.shape[1])
            ymin  = int((box.y - box.h/2) * image.shape[0])
            ymax  = int((box.y + box.h/2) * image.shape[0])


            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), class_color(max_indx), 2)
            cv2.putText(image, '%d' % max_indx, (xmin, ymin - 12), 0, 4e-3 * image.shape[0], (0,255,0), 2)
            
    return image
def show_single_prediction(threshold, seed = None):
    if seed is not None:
        np.random.seed(seed)
    test_img, _, real_output = generate_image(True, 0.2)
    test_pred = model.predict(np.expand_dims(test_img,0))
    fig, (ax_input, ax_pred, ax_gt) = plt.subplots(1,3, figsize = (12, 3))
    ax_input.imshow(test_img)
    ax_input.set_title('Input')
    ax_pred.imshow(interpret_netout(test_img.copy(), test_pred[0], threshold))
    ax_pred.set_title('Prediction')
    ax_gt.imshow(real_output)
    ax_gt.set_title('Ground Truth')
    return fig
_ = show_single_prediction(0.15, seed = 0)
def custom_loss(y_true, y_pred):
    ### Adjust prediction
    # adjust x and y      
    pred_box_xy = tf.sigmoid(y_pred[:,:,:,:,:2])
    
    # adjust w and h
    pred_box_wh = tf.exp(y_pred[:,:,:,:,2:4]) * np.reshape(ANCHORS, [1,1,1,BOX,2])
    pred_box_wh = tf.sqrt(pred_box_wh / np.reshape([float(GRID_W), float(GRID_H)], [1,1,1,1,2]))
    
    # adjust confidence
    pred_box_conf = tf.expand_dims(tf.sigmoid(y_pred[:, :, :, :, 4]), -1)
    
    # adjust probability
    pred_box_prob = tf.nn.softmax(y_pred[:, :, :, :, 5:])
    
    y_pred = tf.concat([pred_box_xy, pred_box_wh, pred_box_conf, pred_box_prob], 4)
    
    ### Adjust ground truth
    # adjust x and y
    center_xy = .5*(y_true[:,:,:,:,0:2] + y_true[:,:,:,:,2:4])
    center_xy = center_xy / np.reshape([(float(NORM_W)/GRID_W), (float(NORM_H)/GRID_H)], [1,1,1,1,2])
    true_box_xy = center_xy - tf.floor(center_xy)
    
    # adjust w and h
    true_box_wh = (y_true[:,:,:,:,2:4] - y_true[:,:,:,:,0:2])
    true_box_wh = tf.sqrt(true_box_wh / np.reshape([float(NORM_W), float(NORM_H)], [1,1,1,1,2]))
    
    # adjust confidence
    pred_tem_wh = tf.pow(pred_box_wh, 2) * np.reshape([GRID_W, GRID_H], [1,1,1,1,2])
    pred_box_area = pred_tem_wh[:,:,:,:,0] * pred_tem_wh[:,:,:,:,1]
    pred_box_ul = pred_box_xy - 0.5 * pred_tem_wh
    pred_box_bd = pred_box_xy + 0.5 * pred_tem_wh
    
    true_tem_wh = tf.pow(true_box_wh, 2) * np.reshape([GRID_W, GRID_H], [1,1,1,1,2])
    true_box_area = true_tem_wh[:,:,:,:,0] * true_tem_wh[:,:,:,:,1]
    true_box_ul = true_box_xy - 0.5 * true_tem_wh
    true_box_bd = true_box_xy + 0.5 * true_tem_wh
    
    intersect_ul = tf.maximum(pred_box_ul, true_box_ul) 
    intersect_br = tf.minimum(pred_box_bd, true_box_bd)
    intersect_wh = intersect_br - intersect_ul
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect_area = intersect_wh[:,:,:,:,0] * intersect_wh[:,:,:,:,1]
    
    iou = tf.truediv(intersect_area, true_box_area + pred_box_area - intersect_area)
    best_box = tf.equal(iou, tf.reduce_max(iou, [3], True)) 
    best_box = tf.to_float(best_box)
    true_box_conf = tf.expand_dims(best_box * y_true[:,:,:,:,4], -1)
    
    # adjust confidence
    true_box_prob = y_true[:,:,:,:,5:]
    
    y_true = tf.concat([true_box_xy, true_box_wh, true_box_conf, true_box_prob], 4)
    #y_true = tf.Print(y_true, [true_box_wh], message='DEBUG', summarize=30000)    
    
    ### Compute the weights
    weight_coor = tf.concat(4 * [true_box_conf], 4)
    weight_coor = SCALE_COOR * weight_coor
    
    weight_conf = SCALE_NOOB * (1. - true_box_conf) + SCALE_CONF * true_box_conf
    
    weight_prob = tf.concat(CLASS_NUM * [true_box_conf], 4) 
    weight_prob = SCALE_PROB * weight_prob 
    
    weight = tf.concat([weight_coor, weight_conf, weight_prob], 4)
    
    ### Finalize the loss
    loss = tf.pow(y_pred - y_true, 2)
    loss = loss * weight
    loss = tf.reshape(loss, [-1, GRID_W*GRID_H*BOX*(4 + 1 + CLASS_NUM)])
    loss = tf.reduce_sum(loss, 1)
    loss = .5 * tf.reduce_mean(loss)
    
    return loss
early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=3, mode='min', verbose=1)
checkpoint = ModelCheckpoint('weights.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min', period=1)
log_dir = os.path.join('.', 'yolo_logs')
os.makedirs(log_dir, exist_ok=True)
tb_counter  = max([int(num) for num in os.listdir(log_dir)] or [0]) + 1
tensorboard = TensorBoard(log_dir=log_dir + str(tb_counter), histogram_freq=0, write_graph=True, write_images=False)
finetune_rate = 0.00001
train_rate = finetune_rate * 1000
sgd = SGD(lr=train_rate, decay=0.0005, momentum=0.9)

model.compile(loss=custom_loss, optimizer=sgd)#'adagrad')
def data_gen(g_img_func, batch_size):
    while True:
        currt_inst = 0
        x_batch = np.zeros((batch_size, NORM_W, NORM_H, 3))
        y_batch = np.zeros((batch_size, GRID_W, GRID_H, BOX, 5+CLASS_NUM))
        
        for index in range(batch_size):
            
            # generate the image and bounding boxes
            img, all_obj, _ = g_img_func()
            
            
            # construct output from object's position and size
            for c_box in all_obj:    
                c_box.norm_dims(img)
                obj = dict(xmin = c_box.x-c_box.w/2,
                           xmax = c_box.x+c_box.w/2,
                           ymin = c_box.y-c_box.h/2,
                           ymax = c_box.y+c_box.h/2
                          )
                box = []
                center_x = c_box.x
                center_x = center_x / (float(NORM_W) / GRID_W)
                center_y = c_box.y
                center_y = center_y / (float(NORM_H) / GRID_H)
                
                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))
                
                if grid_x < GRID_W and grid_y < GRID_H:
                    obj_indx = c_box.c
                    box = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']]
                    
                    y_batch[currt_inst, grid_y, grid_x, :, 0:4]        = BOX * [box]
                    y_batch[currt_inst, grid_y, grid_x, :, 4  ]        = BOX * [1.]
                    y_batch[currt_inst, grid_y, grid_x, :, 5: ]        = BOX * [[0.]*CLASS_NUM]
                    y_batch[currt_inst, grid_y, grid_x, :, 5+obj_indx] = 1.0
                
            # concatenate batch input from the image
            x_batch[currt_inst] = img
            currt_inst += 1
            
            del img, all_obj
        
        yield x_batch, y_batch
print('Test Generator')
for _, (x, y) in zip(range(1), data_gen(generate_image, 2)):
    print('Generator Passed','x',x.shape, 'min,max,mean',x.min(), x.max(), x.mean(),'y', y.shape)
model.fit_generator(data_gen(generate_image, 16), 
                    50, 
                    epochs = 3, 
                    verbose = True,
                    callbacks = [early_stop, tensorboard], #+[checkpoint], 
                    )
show_single_prediction(.001, seed = 0)

