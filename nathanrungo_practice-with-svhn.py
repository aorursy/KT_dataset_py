from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
#SVG(model_to_dot(model).create(prog='dot', format='svg'))
def inv_sigmoid(s):
    return -np.log(1/s-1)
import json
import pandas as pd
import os
import h5py
import glob

import cv2
from IPython.display import Image  
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"]='0'
%matplotlib inline

num_classes = 11
#input_shape = (224, 224, 3)
#input_shape = (60, 120, 3)
input_shape = (416, 416, 3)


def imagePath(mode):       
    #imagePath = glob.glob(dir_path +'/*.png')
    SVHNPath = '../input/street-view-house-numbers'
    dirName = os.path.join(SVHNPath, mode, mode)
    imagePath = []
    #files = os.path.join(SVHNPath, mode, mode, '*.png')
    for filename in os.listdir(dirName):
        #print('process folder : %s' % mode)      
        filenameParts = os.path.splitext(filename)
        if filenameParts[1] != '.png':
           continue
        imagePath.append(int(filenameParts[0]))
    imagePath.sort()
    [f'{dirName}/{name}.png' for name in imagePath]
    return [f'{dirName}/{name}.png' for name in imagePath]

def mat_to_dataset(mat_path):
    f = h5py.File(mat_path, mode='r')
    datasets = {}
    files_count = len(f['digitStruct']['name'])
    for i in range(files_count):
        name_uint16 = f[f['digitStruct']['name'][i,0]][:]
        name = ''.join(chr(n) for n in name_uint16)
        
        bbox = {}
        box_i = f[f['digitStruct']['bbox'][i,0]]
        length = box_i['label'].shape[0]
        for key in ['height', 'label', 'left', 'top', 'width']:
            l = []
            if key=='label':
                l = [ int(str(int(f[box_i[key][index,0]][0][0]))[-1]) if length > 1 else int(box_i[key][0][0]) for index in range(length) ]
            else:
                l = [ int(f[box_i[key][index,0]][0][0]) if length > 1 else int(box_i[key][0][0]) for index in range(length) ]
            bbox[key] = l
        datasets[name] = bbox
        print(f'Loading {i} / {files_count}.\r', end='') 
    print() 
    print(f'{i+1} records loaded.') 
    return datasets

def save_dataset_from_mat(mode='train'):
    dirpath = '../input/street-view-house-numbers'
    mat_path = f'{dirpath}/{mode}_digitStruct.mat'
    dataset = mat_to_dataset(mat_path)
    
    filename = f'{mode}.json'
    with open(filename, 'w') as outfile:    
        json.dump(dataset, outfile)
        
def save_to_npy(d):
    d = {'x_train':x_train,'y_train':y_train,'x_test':x_test,'y_test':y_test}
    with open('data.npy', 'wb') as f:
        np.save(f,d)
def load_from_npy():
    d = np.load('data.npy', allow_pickle=True)
    x_train,y_train,x_test, y_test = d.item()['x_train'],d.item()['y_train'],d.item()['x_test'],d.item()['y_test']
    return (x_train,y_train),(x_test, y_test)

def pad_with_char(string, char):    
    return string + (6 - len(list(string))) * char

def label_padding(label, length=6):
    array = np.zeros((len(label), length),dtype=np.int32)
    array.fill(10)
    for i,l in enumerate(label):
        for j,c in enumerate(l):
            array[i,j] = c
    return array

def loadData(input_shape, num_classes):
    x_train, y_train = loadDataWithMode('train' ,input_shape, num_classes)
    x_test,  y_test  = loadDataWithMode('test' ,input_shape, num_classes)
    d = {'x_train':x_train,'y_train':y_train,'x_test':x_test,'y_test':y_test}
    #with open('data.npy', 'wb') as f:
        #np.save(f,d)
    return (x_train, y_train), (x_test,  y_test)

def loadDataWithMode(mode ,input_shape, num_classes):
    # load train_data
    train_path = imagePath(mode)
    length = 100#len(train_path)   
    if mode=='test':
        length=200
    h,w,c = input_shape
    x_train = np.zeros((length,h,w,c))
    for i in range(length):
        img = cv2.imread(train_path[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w,h)) 
        img = img / 255      
        x_train[i] = img
        print(f'Loading {mode} examples {i+1}/{length}.\r', end='')
    
    train_json = json.load(open(f'../input/svhn-label/{mode}.json'))
    train_label = [train_json[x]['label'] for x in train_json]
    train_label = label_padding(train_label)
    y_train = keras.utils.to_categorical(train_label, num_classes)
    y_train = y_train[:length].reshape(length,-1)
    
    print()
    #print(f'x_{mode}.shape:{x_train.shape}')
    #print(f'y_{mode}.shape:{y_train.shape}')
    return (x_train,y_train)

#train_label[i]
#img = cv2.resize(img, (28, 28)) 
#plt.imshow(img)
#(x_train,y_train),(x_test, y_test) = loadData(input_shape, num_classes)
#(x_train,y_train),(x_test, y_test) = load_from_npy()
#print(f'x_train.shape:{x_train.shape}')
#print(f'y_train.shape:{y_train.shape}')
#print(f'x_test.shape:{x_test.shape}')
#print(f'y_test.shape:{y_test.shape}')
# create a YOLOv3 Keras model and save it to file
# based on https://github.com/experiencor/keras-yolo3
import struct
import numpy as np
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import ZeroPadding2D
from keras.layers import UpSampling2D
from keras.layers.merge import add, concatenate
from keras.models import Model

def _conv_block(inp, convs, skip=True):
	x = inp
	count = 0
	for conv in convs:
		if count == (len(convs) - 2) and skip:
			skip_connection = x
		count += 1
		if conv['stride'] > 1: x = ZeroPadding2D(((1,0),(1,0)))(x) # peculiar padding as darknet prefer left and top
		x = Conv2D(conv['filter'],
				   conv['kernel'],
				   strides=conv['stride'],
				   padding='valid' if conv['stride'] > 1 else 'same', # peculiar padding as darknet prefer left and top
				   name='conv_' + str(conv['layer_idx']),
				   use_bias=False if conv['bnorm'] else True)(x)
		if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
		if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)
	return add([skip_connection, x]) if skip else x

def make_yolov3_model():
	input_image = Input(shape=(None, None, 3))
	# Layer  0 => 4
	x = _conv_block(input_image, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
								  {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
								  {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
								  {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])
	# Layer  5 => 8
	x = _conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
						{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
						{'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])
	# Layer  9 => 11
	x = _conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
						{'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])
	# Layer 12 => 15
	x = _conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
						{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
						{'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])
	# Layer 16 => 36
	for i in range(7):
		x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16+i*3},
							{'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17+i*3}])
	skip_36 = x
	# Layer 37 => 40
	x = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
						{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
						{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])
	# Layer 41 => 61
	for i in range(7):
		x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41+i*3},
							{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42+i*3}])
	skip_61 = x
	# Layer 62 => 65
	x = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
						{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
						{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])
	# Layer 66 => 74
	for i in range(3):
		x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66+i*3},
							{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67+i*3}])
	# Layer 75 => 79
	x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
						{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
						{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
						{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
						{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}], skip=False)
	# Layer 80 => 82
	yolo_82 = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 80},
							  {'filter':  48, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 81}], skip=False)
	# Layer 83 => 86
	x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}], skip=False)
	x = UpSampling2D(2)(x)
	x = concatenate([x, skip_61])
	# Layer 87 => 91
	x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
						{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
						{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
						{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
						{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}], skip=False)
	# Layer 92 => 94
	yolo_94 = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 92},
							  {'filter': 48, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 93}], skip=False)
	# Layer 95 => 98
	x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,   'layer_idx': 96}], skip=False)
	x = UpSampling2D(2)(x)
	x = concatenate([x, skip_36])
	# Layer 99 => 106
	yolo_106 = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 99},
							   {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 100},
							   {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 101},
							   {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 102},
							   {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 103},
							   {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 104},
							   {'filter': 48, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 105}], skip=False)
	model = Model(input_image, [yolo_82, yolo_94, yolo_106])
	return model

class WeightReader:
	def __init__(self, weight_file):
		with open(weight_file, 'rb') as w_f:
			major,	= struct.unpack('i', w_f.read(4))
			minor,	= struct.unpack('i', w_f.read(4))
			revision, = struct.unpack('i', w_f.read(4))
			if (major*10 + minor) >= 2 and major < 1000 and minor < 1000:
				w_f.read(8)
			else:
				w_f.read(4)
			transpose = (major > 1000) or (minor > 1000)
			binary = w_f.read()
		self.offset = 0
		self.all_weights = np.frombuffer(binary, dtype='float32')

	def read_bytes(self, size):
		self.offset = self.offset + size
		return self.all_weights[self.offset-size:self.offset]

	def load_weights(self, model):
		for i in range(106):
			try:
				conv_layer = model.get_layer('conv_' + str(i))
				print("loading weights of convolution #" + str(i))
				if i not in [81, 93, 105]:
					norm_layer = model.get_layer('bnorm_' + str(i))
					size = np.prod(norm_layer.get_weights()[0].shape)
					beta  = self.read_bytes(size) # bias
					gamma = self.read_bytes(size) # scale
					mean  = self.read_bytes(size) # mean
					var   = self.read_bytes(size) # variance
					weights = norm_layer.set_weights([gamma, beta, mean, var])
				if len(conv_layer.get_weights()) > 1:
					bias   = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
					kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
					kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
					kernel = kernel.transpose([2,3,1,0])
					conv_layer.set_weights([kernel, bias])
				else:
					kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
					kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
					kernel = kernel.transpose([2,3,1,0])
					conv_layer.set_weights([kernel])
			except ValueError:
				print("no convolution #" + str(i))

	def reset(self):
		self.offset = 0

# define the model
model = make_yolov3_model()
# load the model weights
#weight_reader = WeightReader('../input/yolov3/yolov3.weights')
# set the model weights into the model
#weight_reader.load_weights(model)
# save the model to file
model.save('model_yolo.h5')
#model.save_weights('weight_yolo.h5')
import numpy as np
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

class BoundBox:
	def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
		self.xmin = xmin
		self.ymin = ymin
		self.xmax = xmax
		self.ymax = ymax
		self.objness = objness
		self.classes = classes
		self.label = -1
		self.score = -1

	def get_label(self):
		if self.label == -1:
			self.label = np.argmax(self.classes)

		return self.label

	def get_score(self):
		if self.score == -1:
			self.score = self.classes[self.get_label()]

		return self.score

def _sigmoid(x):
	return 1. / (1. + np.exp(-x))

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
	grid_h, grid_w = netout.shape[:2]
	nb_box = 3
	netout = netout.reshape((grid_h, grid_w, nb_box, -1))
	nb_class = netout.shape[-1] - 5
	boxes = []
	netout[..., :2]  = _sigmoid(netout[..., :2])
	netout[..., 4:]  = _sigmoid(netout[..., 4:])
	netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
	netout[..., 5:] *= netout[..., 5:] > obj_thresh

	for i in range(grid_h*grid_w):
		row = int(i / grid_w)
		col = int(i % grid_w)
		for b in range(nb_box):
			# 4th element is objectness score
			objectness = netout[int(row)][int(col)][b][4]
			if(objectness.all() <= obj_thresh): continue
			# first 4 elements are x, y, w, and h
			x, y, w, h = netout[int(row)][int(col)][b][:4]
			x = (col + x) / grid_w # center position, unit: image width
			y = (row + y) / grid_h # center position, unit: image height
			w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
			h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
			# last elements are class probabilities
			classes = netout[int(row)][col][b][5:]
			box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
			boxes.append(box)
	return boxes

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
	new_w, new_h = net_w, net_h
	for i in range(len(boxes)):
		x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
		y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
		boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
		boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
		boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
		boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def _interval_overlap(interval_a, interval_b):
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

def bbox_iou(box1, box2):
	intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
	intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
	intersect = intersect_w * intersect_h
	w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
	w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
	union = w1*h1 + w2*h2 - intersect
	return float(intersect) / union

def do_nms(boxes, nms_thresh):
	if len(boxes) > 0:
		nb_class = len(boxes[0].classes)
	else:
		return
	for c in range(nb_class):
		sorted_indices = np.argsort([-box.classes[c] for box in boxes])
		for i in range(len(sorted_indices)):
			index_i = sorted_indices[i]
			if boxes[index_i].classes[c] == 0: continue
			for j in range(i+1, len(sorted_indices)):
				index_j = sorted_indices[j]
				if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
					boxes[index_j].classes[c] = 0

# load and prepare an image
def load_image_pixels(filename, shape):
	# load the image to get its shape
	image = load_img(filename)
	width, height = image.size
	# load the image with the required size
	image = load_img(filename, target_size=shape)
	# convert to numpy array
	image = img_to_array(image)
	# scale pixel values to [0, 1]
	image = image.astype('float32')
	image /= 255.0
	# add a dimension so that we have one sample
	image = expand_dims(image, 0)
	return image, width, height

# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
	v_boxes, v_labels, v_scores = list(), list(), list()
	# enumerate all boxes
	for box in boxes:
		# enumerate all possible labels
		for i in range(len(labels)):
			# check if the threshold for this label is high enough
			if box.classes[i] > thresh:
				v_boxes.append(box)
				v_labels.append(labels[i])
				v_scores.append(box.classes[i]*100)
				# don't break, many labels may trigger for one box
	return v_boxes, v_labels, v_scores

# draw all results
def draw_boxes(filename, v_boxes, v_labels, v_scores):
	# load the image
	data = plt.imread(filename)
	# plot the image
	plt.imshow(data)
	# get the context for drawing boxes
	ax = plt.gca()
	# plot each box
	for i in range(len(v_boxes)):
		box = v_boxes[i]
		# get coordinates
		y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
		# calculate width and height of the box
		width, height = x2 - x1, y2 - y1
		# create the shape
		rect = Rectangle((x1, y1), width, height, fill=False, color='white')
		# draw the box
		ax.add_patch(rect)
		# draw text and score in top left corner
		label = "%s (%.3f)" % (v_labels[i], v_scores[i])
		plt.text(x1, y1, label, color='white')
	# show the plot
	plt.show()
def encode(df,grid_list, num_box, num_classes, anchors_list, net_h, net_w):    
    yhat = []
    length = len(df)
    boxes = [[]] * len(df)  
    for index in range(len(df)):
        print(f'Loading examples {index+1}/{length}.\r', end='')
        r = df.iloc[index,:]
        filename, width, height, left, top, label = r['index'], r['width'], r['height'], r['left'], r['top'], r['label']
        origin_x, origin_y = r['image_w_h']
        bsize = 5 + num_classes
        grid_boxes = [] 
        for i in range(len(width)):             
            w, h, x, y, l = width[i], height[i], left[i], top[i], label[i]
            x = x + w / 2
            y = y + h / 2
            x = x / origin_x
            y = y / origin_y     
            w = w / origin_x * net_w
            h = h / origin_y * net_h
            grid_idx, anchor_idx = get_best_anchor(anchors_list, [w,h])
            grid = grid_list[grid_idx]
            grid_x , grid_y = grid, grid
            
            x = x * grid_x
            y = y * grid_y
            col = int(x)
            row = int(y)
            x = x - col
            y = y - row
            #x = inv_sigmoid(x)
            #y = inv_sigmoid(y)
            w = np.log(w / anchors_list[grid_idx][2 * anchor_idx + 0])
            h = np.log(h / anchors_list[grid_idx][2 * anchor_idx + 1])
            score = 1
            ybox = np.zeros((bsize,))   
            ybox[:5] = x,y,w,h,score
            ybox[5:] = keras.utils.to_categorical(l, num_classes)
            position = (row,col)
            bbox = grid_idx, row, col, anchor_idx, ybox.copy()        
            grid_boxes.append(bbox)
        boxes[index] = grid_boxes
    return boxes
def iou(box1, box2):
    w1, h1 = box1
    w2, h2 = box2
    inter_w = min(w1,w2)
    inter_h = min(h1,h2)
    intersect = inter_w * inter_h
    union = w1 * h1 + w2 * h2 - intersect
    return intersect / union

def get_best_anchor(anchors, box):
    l1,l2 = len(anchors),len(anchors[0])//2
    array_iou = np.zeros((l1,l2))
    for i in range(l1):
        anchor = anchors[i]
        for j in range(l2):
            array_iou[i,j] = iou(box,[anchor[j*2],anchor[j*2+1]])
    #print(array_iou)
    ind = np.unravel_index(np.argmax(array_iou, axis=None), array_iou.shape)
    return ind
def kmeans_xufive(ds, k):
    """k-means聚类算法

    k       - 指定分簇数量
    ds      - ndarray(m, n)，m个样本的数据集，每个样本n个属性值
    """

    m, n = ds.shape # m：样本数量，n：每个样本的属性值个数
    result = np.empty(m, dtype=np.int) # m个样本的聚类结果
    cores = np.empty((k, n)) # k个质心
    cores = ds[np.random.choice(np.arange(m), k, replace=False)] # 从m个数据样本中不重复地随机选择k个样本作为质心
    
    count=0
    while True: # 迭代计算
        #d = np.square(np.repeat(ds, k, axis=0).reshape(m, k, n) - cores)
        #distance = np.sqrt(np.sum(d, axis=2)) # ndarray(m, k)，每个样本距离k个质心的距离，共有m行
        count+=1
        print(f'\rcount: {count}',end='')
        distance = np.ones((m,k))
        for mm in range(m):
            for kk in range(k):
                distance[mm,kk] = 1 - iou(ds[mm,:],cores[kk,:])
        index_min = np.argmin(distance, axis=1) # 每个样本距离最近的质心索引序号

        if (index_min == result).all(): # 如果样本聚类没有改变
            return result, cores # 则返回聚类结果和质心数据

        result[:] = index_min # 重新分类
        for i in range(k): # 遍历质心集
            items = ds[result==i] # 找出对应当前质心的子样本集
            cores[i] = np.mean(items, axis=0) # 以子样本集的均值作为当前质心的位置
                                  
def test_cluster(result,cores):
    plt.scatter(ds[:,0], ds[:,1], s=1, c=result.astype(np.int))
    plt.scatter(cores[:,0], cores[:,1], marker='x', c=np.arange(k))
    plt.axis('equal')
    plt.show()

def get_box_shape(df, net_h, net_w):    
    yhat = []
    length = len(df)
    box_h = []
    box_w = []  
    for index in range(length):
        #print(f'Loading examples {index+1}/{length}.\r', end='')
        r = df.iloc[index,:]
        width, height = r['width'], r['height']
        origin_x, origin_y = r['image_w_h']
        for i in range(len(width)):             
            w, h = width[i], height[i]
            w = w / origin_x * net_w
            h = h / origin_y * net_h
            #box_shape.append((w, h))
            box_w.append(w)
            box_h.append(h)     
    return np.stack((box_w,box_h),axis=1)

def get_anchors(df):
    l = get_box_shape(df, 416, 416)
    #df = pd.DataFrame({'w':l[0],'h':l[1]})
    result, cores =  kmeans_xufive(l, 9)
    anchors = np.array(cores, dtype=int)
    size = [x[0]*x[1] for x in anchors]
    index = np.argsort(size)[::-1]
    return anchors[index,:].reshape(3,6).tolist()
#a = get_anchors(df_train) 
# [[99, 335, 75, 314, 59, 323], [62, 249, 45, 292, 46, 207], [32, 250, 34, 159, 21, 116]]
def image_dir(mode):
    return '../input/streetclassify/input/input/' + mode + '/'
def label_file(mode):
    return '../input/streetclassify/input/input/' + mode + '.json'

def get_image_w_h(df, mode):
    l = len(df)
    shape = [None] * l
    for index in range(l):
        filename = df.loc[index,'index']
        filename = image_dir(mode) + filename
        img = plt.imread(filename)
        shape[index] = img.shape[1], img.shape[0]
    df['image_w_h'] = shape
        
def get_dataset(mode):
    if os.path.exists():
        return pd.read_csv(f'../input/practice-with-svhn/{mode}.csv')
    df = pd.read_json(label_file(mode)).T.reset_index()
    df = df.iloc[:,:]
    get_image_w_h(df, mode)
    #df = df_train.head(4)
    grid_list = [13,26,52]
    anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
    anchors = [[99, 335, 75, 314, 59, 323], [62, 249, 45, 292, 46, 207], [32, 250, 34, 159, 21, 116]]

    boxes = encode(df, grid_list, 3, num_classes, anchors, 416, 416)
    df[f'grid'] = boxes
    #train_label = label_padding(df['label'])
    #digit = keras.utils.to_categorical(train_label, num_classes, dtype='int32')
    #df['digit'] = list(digit.reshape(digit.shape[0],-1))
    #df['dhex'] = [int(''.join([str(l[i]) if i<len(l) else 'a' for i in range(6)]),16) for l in df['label']]
    return df

def get_submit(file):
    df = pd.read_csv(file)#.reset_index()
    return df 

def json_to_dataset(file):
    df = pd.read_json(file).T.reset_index()
    label = df['label'].copy()#[train_json[x]['label'] for x in train_json])
    train_label = label_padding(label)
    df['digit_raw'] = [int(''.join([str(n) for n in l])) for l in label]
    digit = keras.utils.to_categorical(train_label, num_classes, dtype='int32')
    df['digit'] = list(digit.reshape(digit.shape[0],-1))
    #df[f'Digit'] = train_label.reshape(train_label.shape[0],-1).tolist()
    train_label = train_label.astype(str)
    df['Digit'] = train_label.tolist()
    t = train_label.transpose(1,0)
    df['count'] = [str(len(l)) for l in label]
    df['dhex'] = [int(''.join([str(l[i]) if i<len(l) else 'a' for i in range(6)]),16) for l in df['label']]
    for i in range(6):
        df[f'D{i}'] = t[i].tolist()
    return df    
df_train = get_dataset('train')
df_val = get_dataset('val')
df_test = get_submit('../input/streetclassify/input/input/test_A_sample_submit.csv')
#a = df_val.iloc[:,:]
#df_train.to_csv('train.csv')
#df_val.to_csv('val.csv')
batch_size = 8
train_dir = '../input/streetclassify/input/input/train'
#train_dir = '../input/street-view-house-numbers/train/train'
val_dir = '../input/streetclassify/input/input/val'
#val_dir = '../input/street-view-house-numbers/test/test'
test_dir = '../input/streetclassify/input/input/test_a'
#int_to_char = ['zero','one','two', 'three', 'four','five','six','seven','eight','nine','ten']
int_to_char = ['0','1','2', '3', '4','5','6','7','8','9','10']

trainGen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                        #rotation_range=5,
                                                        #width_shift_range=0.2,
                                                        #height_shift_range=0.2,
                                                        #channel_shift_range=0.2,
                                                        #zoom_range=0.2,
                                                        #validation_split=0.3
                                                       )
train_gen = trainGen.flow_from_dataframe(df_train.iloc[:],                 #dataframe
                                               #directory=train_folder,     #根目录（当前路径）
                                               directory=train_dir,     #根目录（当前路径）
                                               x_col='index',
                                               #y_col='dhex',#'D0',
                                               y_col='grid',#'D0',
                                               #y_col=['D0','D1','D2','D3','D4','D5'],
                                               #classes=int_to_char[:],
                                               target_size=input_shape[0:2],
                                               batch_size=batch_size,
                                               seed=3,
                                               shuffle=True,
                                               class_mode='raw',
                                              )
validGen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

valid_gen = validGen.flow_from_dataframe(df_val.iloc[:],                 #dataframe
                                               #directory=train_folder,     #根目录（当前路径） 
                                               directory=val_dir,     #根目录（当前路径） 
                                               x_col='index',
                                               #y_col='dhex',#'D0',
                                               y_col='grid',
                                               #y_col=['D0','D1','D2','D3','D4','D5'],
                                               #classes=int_to_char[:],
                                               target_size=input_shape[0:2],
                                               batch_size=batch_size,
                                               seed=3,
                                               shuffle=False,
                                               class_mode='raw',
                                              )
test_gen =  validGen.flow_from_dataframe(df_test.iloc[:],
                                               directory=test_dir,
                                               x_col='file_name',
                                               y_col=None,
                                               target_size=input_shape[0:2],
                                               batch_size=batch_size,
                                               seed=3,
                                               shuffle=False,
                                               class_mode=None,
                                              )
#'multi_output',)"binary", "categorical", "input", "multi_output", "raw", sparse" or None.
def MyGenerator(generator):
    grid_list = [13,26,52]
    anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
    
    num_box = len(anchors[0])//2
    while(True):
        x, y = next(generator)
        bz = y.shape[0]
        box_size = y[0][0][4].shape[0]
        y_true = [np.zeros((bz,g,g,num_box*box_size)) for g in grid_list]        
        for index in range(bz):
            for grid in y[index]:
                grid_index, row, col, b = grid[:4]
                y_true[grid_index][index, row, col, b*box_size:(b+1)*box_size] = grid[4]
        #b = dhex_to_sparse(y)
        #y = keras.utils.to_categorical(b,11).reshape(-1,66)
        yield x,y_true
        
train_generator = MyGenerator(train_gen)
valid_generator = MyGenerator(valid_gen)
test_generator = MyGenerator(test_gen)
def get_sample(df, **kw):
    i = np.random.randint(len(df)) 
    if 'array' in kw:
        i = np.random.randint(len(array))
        img = array[i]
        label = df["label"][i]
        print(f'From array:')
    elif 'generator' in kw:
        #img = next(generator)[0][0]
        #img = trainGen.random_transform(x_train[i],seed=None)
        generator = kw['generator']
        bz = generator.batch_size
        #i=0
        img = generator[i//bz][0][i%bz,...]
        label = generator[i//bz][1][i%bz]
        label = hex(label)
        print(i)
        print(f'From generator:')
    else:
        path = f'../input/street-view-house-numbers/train/train/{df["index"][i]}'
        path = f'../input/streetclassify/input/input/train/{df["index"][i]}'
        print(path)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = df["label"][i]
        print('From path:')
    plt.imshow(img)
    print(f'Example {i}:')
    print(f'shape: {img.shape}')
    print(f'label: {label}')
    return img

#img = get_sample(df_train.iloc[:1,:])#, generator=train_generator
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []        
        self.accuracy = []
        self.accuracy1 = []
        self.accuracy2 = []

        self.val_losses = []
        self.val_accuracy = []
        
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('my_metric_fn')) 
    
        #self.val_losses.append(logs.get('val_loss'))   
        #self.val_acc.append(logs.get('val_acc'))
        
def dhex_to_sparse(a):
    output_list = []
    for _ in range(6):
        output_list.append(a % 16)
        a = a//16    
    outputs = K.stack(output_list[::-1],1)
    y = outputs
    return y

def yolo_loss(y_true, y_pred):
    y_true = tf.reshape(y_true,(-1,6,11)) 
    y_pred = tf.reshape(y_pred,(-1,6,11))
    losses = keras.losses.categorical_crossentropy(y_true, y_pred)
    return K.sum(losses,-1)

def yolo_metric(y_true, y_pred):
    y_true = tf.reshape(y_true,(-1,6,11)) 
    y_pred = tf.reshape(y_pred,(-1,6,11))
    return K.cast(K.sum(K.cast(K.equal(K.argmax(y_true),
                          K.argmax(y_pred)), 
                               K.floatx()), 
                        axis=-1)==6,
                  K.floatx())
def simple():
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
        ]
    )
    return model

def fixed():
    pre_model = keras.applications.ResNet50(include_top=False,weights='imagenet',input_shape=input_shape)
    x = layers.Flatten()(pre_model.output)
    x = [layers.Dense(num_classes, activation="softmax")(x) for i in range(6)]
    outputs = layers.concatenate(x)
    #out = layers.Dense(num_classes, activation="softmax")(pre_model.output)
    model = keras.Model(inputs=pre_model.inputs,outputs=outputs)
    return model

def yolo():
    pre_model = keras.applications.ResNet50(include_top=False,weights='imagenet',input_shape=input_shape)
    outputs = layers.Conv2D(32, kernel_size=(1, 1), activation="relu")(pre_model.output)
    model = keras.Model(inputs=pre_model.inputs,outputs=outputs)
    return model

def create_model():
    model = yolo()    
    #model = keras.models.load_model(f'../input/practice-with-svhn/{model_file}', custom_objects=custom_objects)
    opt = keras.optimizers.Adam(lr=0.001)
    model.compile(opt, loss=yolo_loss, metrics=[yolo_metric])
    
    return model
model_version = 3
model_file = f'model_v{model_version}.h5'
custom_objects={                         
                }
#model = keras.models.load_model(f'../input/practice-with-svhn/model_v2.h5')
#model = keras.models.load_model(f'../input/practice-with-svhn/{model_file}', custom_objects=custom_objects)
#batch_size = 32
#H = model.fit(x_train, y_true, batch_size=batch_size,epochs=epochs,validation_split=0.1)
#model = create_model()
logdir = os.path.join('hourse_num')#'./hourse_num'
if not os.path.exists(logdir):
    os.mkdir(logdir)
    
output_model_file = os.path.join(logdir, model_file)

callbacks=[
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file,
                                    save_best_only = True,
                                    save_weights_only = False),
    keras.callbacks.EarlyStopping(patience=5,min_delta=1e-3),
    LossHistory()
]
opt = keras.optimizers.Adam(lr=0.001)
model.compile(opt, loss="mean_squared_error", metrics=["mean_squared_error"])
#from kaggle_secrets import UserSecretsClient
#user_secrets = UserSecretsClient()
#user_credential = user_secrets.get_gcloud_credential()
#user_secrets.set_tensorflow_credential(user_credential)
epochs = 1

#batch_size = 32
train_num = train_gen.samples
valid_num = valid_gen.samples
print(train_num, valid_num)
#history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1,shuffle=True)
h = LossHistory()
#H = model.fit_generator(train_generator)
H = model.fit_generator(train_generator,
                        steps_per_epoch=train_num//batch_size,
                        validation_data=MyGenerator(valid_generator),
                        validation_steps=valid_num//batch_size,                              
                        epochs=epochs,
                        callbacks=[callbacks])
model.save('model_v3.h5')
#model.save(model_file)
#model = keras.models.load_model(callbacks[1].filepath,custom_objects)
#model = keras.models.load_model('model_v4.h5')
# plot the training loss and accuracy
N = epochs
plt.style.use("ggplot")

plt.figure()
#plt.plot(H.history["loss"], label="train_loss")
h = callbacks[3]
plt.plot(h.losses, label="train_loss")

#plt.plot(H.history["val_loss"], label="val_loss")
plt.title("Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")

plt.savefig("loss.png")

#plt.figure()
#plt.plot(H.history["loss"], label="train_loss")
#plt.plot(H.history["val_loss"], label="val_loss")
#plt.plot(H.history["sparse_metric_fn"], label="train_accuracy")
#plt.plot(H.history["val_sparse_metric_fn"], label="val_accuracy")
#plt.title("Accuracy")
#plt.xlabel("Epoch #")
#plt.ylabel("Accuracy")
#plt.legend(loc="lower left")

#plt.savefig("accuracy.png")
%load_ext tensorboard

# Open an embedded TensorBoard viewer
%tensorboard --logdir {hourse_num}
NUM_TEST_IMAGES = valid_gen.samples
evaluater = model.evaluate_generator(valid_generator,
	steps=(NUM_TEST_IMAGES // batch_size) + 1, verbose=1)
# Get False Labeled
NUM_TEST_IMAGES = valid_gen.n
#predIdxs = model.predict_generator(val, steps=(NUM_TEST_IMAGES // batch_size) + 1, verbose=1)
train_gen[0][0][0:1]
predIdxs = model.predict(train_gen[0][0][0:1])
#print(predIdxs[0].reshape(-1,6,11).argmax(-1))
#predIdxs.shape
for x in predIdxs:
    print(x.shape)
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if(objectness.all() <= obj_thresh): continue
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
            boxes.append(box)
    return boxes

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

def _interval_overlap(interval_a, interval_b):
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

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    union = w1*h1 + w2*h2 - intersect
    return float(intersect) / union

def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0

# load and prepare an image
def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = load_img(filename)
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, target_size=shape)
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height

# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
                # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores

# draw all results
def draw_boxes(filename, v_boxes, v_labels, v_scores):
    # load the image
    data = pyplot.imread(filename)
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='white')
        # draw the box
        ax.add_patch(rect)
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        pyplot.text(x1, y1, label, color='white')
    # show the plot
    pyplot.show()

# load yolov3 model
#model = load_model('model.h5')
# define the expected input shape for the model
input_w, input_h = 416, 416
# define our new photo
photo_filename = image_dir('train') + '/000000.png'
# load and prepare image
image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))
# make prediction
yhat = model.predict(image)
# summarize the shape of the list of arrays
print([a.shape for a in yhat])
# define the anchors
anchors = [[99, 335, 75, 314, 59, 323], [62, 249, 45, 292, 46, 207], [32, 250, 34, 159, 21, 116]]
# define the probability threshold for detected objects
class_threshold = 0.6
boxes = list()
for i in range(len(yhat)):
    # decode the output of the network
    boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
# correct the sizes of the bounding boxes for the shape of the image
correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
# suppress non-maximal boxes
do_nms(boxes, 0.5)
# define the labels
labels = ['0','1','2', '3', '4','5','6','7','8','9','10']
# get the details of the detected objects
v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
# summarize what we found
for i in range(len(v_boxes)):
    print(v_labels[i], v_scores[i])
# draw what we found
draw_boxes(photo_filename, v_boxes, v_labels, v_scores)
def predict(predIdxs):
    p_list = [None] * predIdxs.shape[0]
    p = predIdxs.reshape(-1,6,11).argmax(-1)
    for i in range(p.shape[0]):
        l = ''
        for j in range(p.shape[1]):
            if p[i,j]==10:
                break        
            l += str(p[i,j])
        p_list[i] = l
    return p_list

df = pd.DataFrame({'x': df_val['index'],
                   'label': df_val['label'],
                   'prediction': predict(predIdxs),
                  })
df['acc'] = sparse_metric_fn(df_val['dhex'],predIdxs)
d = df.query('acc != 1')
i=50
#i = np.random.randint(len(df))
example = d.iloc[i,:]
path = f'../input/streetclassify/input/input/val/{example["x"]}'
print(path)
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
label1 = example["label"]
label2 = example["prediction"]

print('From path:')
plt.imshow(img)
#print(f'Example {i}:')
#print(f'shape: {img.shape}')
print(f'label: {label1}')
print(f'prediction: {label2}')
i=i+1
NUM_TEST_IMAGES = test_generator.samples
#predIdxs = model.predict_generator(test_generator, steps=(NUM_TEST_IMAGES // batch_size) + 1, verbose=1)
predIdxs = model.predict(x_train[:10])
print(predIdxs[0:10].reshape(-1,6,11).argmax(-1))

predIdxs.shape
submit_path = '../input/streetclassify/input/input/test_A_sample_submit.csv'
df_submit = pd.read_csv(submit_path)
df_submit['file_code'] = predict(predIdxs)
df_submit.to_csv('test_submit.csv')
df_submit
#predict(predIdxs)