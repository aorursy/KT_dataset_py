!nvidia-smi
!wget -O casia_webface.zip https://www.dropbox.com/s/wpx6tqjf0y5mf6r/faces_ms1m-refine-v2_112x112.zip?dl=1
!unzip casia_webface.zip
!rm casia_webface.zip

!pip install mxnet
!pip install -U efficientnet==1.1.0
!pip install bcolz
%tensorflow_version 2.x
import math
import mxnet as mx
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import efficientnet.tfkeras as efn 

from shutil import rmtree
# This function taken from https://github.com/auroua/InsightFace_TF/blob/master/data/mx2tfrecords.py

def mx2tfrecords(imgidx, imgrec):
    output_path = f"faces_emore/tran.tfrecords"
    writer = tf.compat.v1.python_io.TFRecordWriter(output_path)
    for i in imgidx:
        img_info = imgrec.read_idx(i)
        header, img = mx.recordio.unpack(img_info)
        label = int(header.label)
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())  # Serialize To String
        if i % 10000 == 0:
            print('%d num image processed' % i)
    writer.close()

imgrec = mx.recordio.MXIndexedRecordIO(f"faces_emore/train.idx", f"faces_emore/train.rec", 'r')
s = imgrec.read_idx(0)
header, _ = mx.recordio.unpack(s)
print(header.label)
imgidx = list(range(1, int(header.label[0])))

mx2tfrecords(imgidx, imgrec)

!rm faces_emore/train.idx
!rm faces_emore/train.rec
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

file_obj = drive.CreateFile({'id': "1WO5Meh_yAau00Gm2Rz2Pc0SRldLQYigT"})
file_obj.GetContentFile('lfw_align_112.zip')
!unzip -q lfw_align_112.zip
!rm lfw_align_112.zip
class ArcFaceLayer(tf.keras.layers.Layer):
	def __init__(self, num_classes, arc_m=0.5, arc_s=64., regularizer_l: float = 5e-4, **kwargs):  # has been set to it's defaults according to arcface paper
		super(ArcFaceLayer, self).__init__(**kwargs)
		self.num_classes = num_classes
		self.regularizer_l = regularizer_l
		self.arc_m = arc_m
		self.arc_s = arc_s

		self.cos_m = tf.identity(math.cos(self.arc_m))
		self.sin_m = tf.identity(math.sin(self.arc_m))
		self.th = tf.identity(math.cos(math.pi - self.arc_m))
		self.mm = tf.multiply(self.sin_m, self.arc_m)

	def build(self, input_shape):
		self.kernel = self.add_weight(name="kernel", shape=[512, self.num_classes], initializer=tf.keras.initializers.glorot_normal(),
		                              trainable=True, regularizer=tf.keras.regularizers.l2(self.regularizer_l))

		super(ArcFaceLayer, self).build(input_shape)

	def call(self, features, labels):
		embedding_norm = tf.norm(features, axis=1, keepdims=True)
		embedding = tf.divide(features, embedding_norm, name='norm_embedding')
		weights_norm = tf.norm(self.kernel, axis=0, keepdims=True)
		weights = tf.divide(self.kernel, weights_norm, name='norm_weights')

		cos_t = tf.matmul(embedding, weights, name='cos_t')
		cos_t2 = tf.square(cos_t, name='cos_2')
		sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
		sin_t = tf.sqrt(sin_t2, name='sin_t')
		cos_mt = self.arc_s * tf.subtract(tf.multiply(cos_t, self.cos_m), tf.multiply(sin_t, self.sin_m), name='cos_mt')

		cond_v = cos_t - self.th
		cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

		keep_val = self.arc_s*(cos_t - self.mm)
		cos_mt_temp = tf.where(cond, cos_mt, keep_val)

		mask = tf.one_hot(labels, depth=self.num_classes, name='one_hot_mask')
		inv_mask = tf.subtract(1., mask, name='inverse_mask')

		s_cos_t = tf.multiply(self.arc_s, cos_t, name='scalar_cos_t')

		output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')

		return output
class TensorBoardCallback:
	def delete_graphs(self):
		if tf.io.gfile.exists(self.logdir):
			rmtree(self.logdir)
			print(f"[*] {self.logdir} has deleted with shutil's rmtree")

	def initialize(self, delete_if_exists: bool = False):
		if delete_if_exists:
			self.delete_graphs()

		self.file_writer = tf.summary.create_file_writer(logdir=self.logdir)

	def __init__(self, logdir: str = "graphs/"):
		self.logdir = logdir
		self.file_writer = None

		self.initial_step = 0

	def __call__(self, data_json: dict, description: str = None, **kwargs):
		with self.file_writer.as_default():
			for key in data_json:
				tf.summary.scalar(key, data_json[key], step=self.initial_step, description=description)

		self.initial_step += 1

	def add_with_step(self, data_json: dict, description: str = None, step: int = 0):
		with self.file_writer.as_default():
			for key in data_json:
				tf.summary.scalar(key, data_json[key], step=step, description=description)

	def add_text(self, name: str, data: str, step: int, **kwargs):
		with self.file_writer.as_default():
			tf.summary.text(name, data, step=step)

	def add_images(self, name: str, data, step: int, max_outputs: int = None, **kwargs):
		if max_outputs is None:
			max_outputs = data.shape[0]

		with self.file_writer.as_default():
			tf.summary.image(name, data, max_outputs=max_outputs, step=step)

import sys
import os
import mxnet as mx
import tensorflow as tf


def Conv(data, **kwargs):
	# name = kwargs.get('name')
	# _weight = mx.symbol.Variable(name+'_weight')
	# _bias = mx.symbol.Variable(name+'_bias', lr_mult=2.0, wd_mult=0.0)
	# body = mx.sym.Convolution(weight = _weight, bias = _bias, **kwargs)
	kwargs["kernel_size"] = kwargs["kernel"]
	kwargs["filters"] = kwargs["num_filter"]
	kwargs["strides"] = kwargs["stride"]
	padding="valid"
	try:
		# data = tf.keras.layers.ZeroPadding2D(kwargs["pad"])(data)
		padding="same"
		del kwargs["pad"]
	except KeyError:
		pass

	del kwargs["kernel"]
	del kwargs["num_filter"]
	del kwargs["stride"]
	body = tf.keras.layers.Conv2D(padding=padding, **kwargs)(data)
	return body


def Act(data, act_type, name):
	if act_type == 'prelu':
		body = tf.keras.layers.PReLU(name=name)(data)
	else:
		body = tf.keras.layers.Activation(act_type, name=name)(data)
	return body


def residual_unit_v1(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
	"""Return ResNet Unit symbol for building ResNet
	Parameters
	----------
	data : str
		Input data
	num_filter : int
		Number of output channels
	bnf : int
		Bottle neck channels factor with regard to num_filter
	stride : tuple
		Stride used in convolution
	dim_match : Boolean
		True means channel number between input and output is the same, otherwise means differ
	name : str
		Base name of the operators
	workspace : int
		Workspace used in convolution operator
	"""
	use_se = kwargs.get('version_se', 1)
	bn_mom = kwargs.get('bn_mom', 0.9)
	workspace = kwargs.get('workspace', 256)
	memonger = kwargs.get('memonger', False)
	act_type = kwargs.get('version_act', 'prelu')
	# print('in unit1')
	if bottle_neck:
		conv1 = Conv(data=data, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=stride, pad=(0, 0),
					 use_bias=False, name=name + '_conv1')
		bn1 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn1')(conv1)
		act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
		conv2 = Conv(data=act1, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=(1, 1), pad=(1, 1),
					 use_bias=False, name=name + '_conv2')
		bn2 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn2')(conv2)
		act2 = Act(data=bn2, act_type=act_type, name=name + '_relu2')
		conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), use_bias=False, name=name + '_conv3')
		bn3 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn3')(conv3)

		if use_se:
			# se begin
			body = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), name=name + '_se_pool1')(bn3)
			body = Conv(data=body, num_filter=num_filter // 16, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv1")
			body = Act(data=body, act_type=act_type, name=name + '_se_relu1')
			body = Conv(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv2")
			body = tf.keras.layers.Activation('sigmoid', name=name + "_se_sigmoid")(body)
			bn3 = tf.keras.layers.Multiply()([bn3, body])
		# se end

		if dim_match:
			shortcut = data
			x = tf.keras.layers.Add()([bn3, shortcut])
		else:
			conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, use_bias=False, name=name + '_conv1sc')
			shortcut = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_sc')(conv1sc)
			x = tf.keras.layers.Add()([bn3, shortcut])
		return Act(data=x, act_type=act_type, name=name + '_relu3')
	else:
		conv1 = Conv(data=data, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
					 use_bias=False, name=name + '_conv1')
		bn1 = tf.keras.layers.BatchNormalization(momentum=bn_mom, epsilon=2e-5, name=name + '_bn1')(conv1)
		act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
		conv2 = Conv(data=act1, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
					 use_bias=False, name=name + '_conv2')
		bn2 = tf.keras.layers.BatchNormalization(momentum=bn_mom, epsilon=2e-5, name=name + '_bn2')(conv2)
		if use_se:
			# se begin
			body = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), name=name + '_se_pool2')(bn2)
			body = Conv(data=body, num_filter=int(num_filter // 16), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv1")
			body = Act(data=body, act_type=act_type, name=name + '_se_relu1')
			body = Conv(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv2")
			body = tf.keras.layers.Activation('sigmoid', name=name + "_se_sigmoid")(body)
			bn2 = tf.keras.layers.Multiply()([bn2, body])
		# se end

		if dim_match:
			shortcut = data
			x = tf.keras.layers.Add()([bn2 + shortcut])
		else:
			conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, use_bias=False,
						   name=name + '_conv1sc')
			shortcut = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_sc')(conv1sc)
			x = tf.keras.layers.Add()([bn2 + shortcut])
		return Act(data=x, act_type=act_type, name=name + '_relu3')


def residual_unit_v1_L(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
	"""Return ResNet Unit symbol for building ResNet
	Parameters
	----------
	data : str
		Input data
	num_filter : int
		Number of output channels
	bnf : int
		Bottle neck channels factor with regard to num_filter
	stride : tuple
		Stride used in convolution
	dim_match : Boolean
		True means channel number between input and output is the same, otherwise means differ
	name : str
		Base name of the operators
	workspace : int
		Workspace used in convolution operator
	"""
	use_se = kwargs.get('version_se', 1)
	bn_mom = kwargs.get('bn_mom', 0.9)
	workspace = kwargs.get('workspace', 256)
	memonger = kwargs.get('memonger', False)
	act_type = kwargs.get('version_act', 'prelu')
	# print('in unit1')
	if bottle_neck:
		conv1 = Conv(data=data, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
					 use_bias=False, name=name + '_conv1')
		bn1 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn1')(conv1)
		act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
		conv2 = Conv(data=act1, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=(1, 1), pad=(1, 1),
					 use_bias=False, name=name + '_conv2')
		bn2 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn2')(conv2)
		act2 = Act(data=bn2, act_type=act_type, name=name + '_relu2')
		conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1, 1), stride=stride, pad=(0, 0), use_bias=False, name=name + '_conv3')
		bn3 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn3')(conv3)

		if use_se:
			# se begin
			body = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), name=name + '_se_pool1')(bn3)
			body = Conv(data=body, num_filter=int(num_filter // 16), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv1")
			body = Act(data=body, act_type=act_type, name=name + '_se_relu1')
			body = Conv(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv2")
			body = tf.keras.layers.Activation('sigmoid', name=name + "_se_sigmoid")(body)
			bn3 = tf.keras.layers.Multiply()([bn3, body])
		# se end

		if dim_match:
			shortcut = data
			x = tf.keras.layers.Add()([bn3, shortcut])
		else:
			conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, use_bias=False, name=name + '_conv1sc')
			shortcut = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_sc')(conv1sc)
			x = tf.keras.layers.Add()([bn3, shortcut])
		return Act(data=x, act_type=act_type, name=name + '_relu3')
	else:
		conv1 = Conv(data=data, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
					 use_bias=False, name=name + '_conv1')
		bn1 = tf.keras.layers.BatchNormalization(momentum=bn_mom, epsilon=2e-5, name=name + '_bn1')(conv1)
		act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
		conv2 = Conv(data=act1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
					 use_bias=False, name=name + '_conv2')
		bn2 = tf.keras.layers.BatchNormalization(momentum=bn_mom, epsilon=2e-5, name=name + '_bn2')(conv2)
		if use_se:
			# se begin
			body = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), name=name + '_se_pool1')(bn2)
			body = Conv(data=body, num_filter=num_filter // 16, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv1")
			body = Act(data=body, act_type=act_type, name=name + '_se_relu1')
			body = Conv(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv2")
			body = tf.keras.layers.Activation('sigmoid', name=name + "_se_sigmoid")(body)
			bn2 = tf.keras.layers.Multiply()([bn2, body])
		# se end

		if dim_match:
			shortcut = data
			x = tf.keras.layers.Add()([bn2 + shortcut])
		else:
			conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, use_bias=False,
							name=name + '_conv1sc')
			shortcut = tf.keras.layers.BatchNormalization(momentum=bn_mom, epsilon=2e-5, name=name + '_sc')(conv1sc)
			x = tf.keras.layers.Add()([bn2 + shortcut])
		return Act(data=x, act_type=act_type, name=name + '_relu3')


def residual_unit_v2(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
	"""Return ResNet Unit symbol for building ResNet
	Parameters
	----------
	data : str
		Input data
	num_filter : int
		Number of output channels
	bnf : int
		Bottle neck channels factor with regard to num_filter
	stride : tuple
		Stride used in convolution
	dim_match : Boolean
		True means channel number between input and output is the same, otherwise means differ
	name : str
		Base name of the operators
	workspace : int
		Workspace used in convolution operator
	"""
	use_se = kwargs.get('version_se', 1)
	bn_mom = kwargs.get('bn_mom', 0.9)
	workspace = kwargs.get('workspace', 256)
	memonger = kwargs.get('memonger', False)
	act_type = kwargs.get('version_act', 'prelu')
	# print('in unit2')
	if bottle_neck:
		# the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
		bn1 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn1')(data)
		act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
		conv1 = Conv(data=act1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
					 use_bias=False, name=name + '_conv1')
		bn2 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn2')(conv1)
		act2 = Act(data=bn2, act_type=act_type, name=name + '_relu2')
		conv2 = Conv(data=act2, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=stride, pad=(1, 1),
					 use_bias=False, name=name + '_conv2')
		bn3 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn3')(conv2)
		act3 = Act(data=bn3, act_type=act_type, name=name + '_relu3')
		conv3 = Conv(data=act3, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), use_bias=False, name=name + '_conv3')
		if use_se:
			# se begin
			body = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), name=name + '_se_pool1')(conv3)
			body = Conv(data=body, num_filter=num_filter // 16, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv1")
			body = Act(data=body, act_type=act_type, name=name + '_se_relu1')
			body = Conv(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv2")
			body = tf.keras.layers.Activation('sigmoid', name=name + "_se_sigmoid")(body)
			conv3 = tf.keras.layers.Multiply()([conv3, body])
		if dim_match:
			shortcut = data
			x = tf.keras.layers.Add()([conv3 + shortcut])
		else:
			shortcut = Conv(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, use_bias=False,
							name=name + '_sc')
			x = tf.keras.layers.Add()([conv3 + shortcut])
		return x
	else:
		bn1 = tf.keras.layers.BatchNormalization( momentum=bn_mom, epsilon=2e-5, name=name + '_bn1')(data)
		act1 = Act(data=bn1, act_type=act_type, name=name + '_relu1')
		conv1 = Conv(data=act1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
					 use_bias=False, name=name + '_conv1')
		bn2 = tf.keras.layers.BatchNormalization(momentum=bn_mom, epsilon=2e-5, name=name + '_bn2')(conv1)
		act2 = Act(data=bn2, act_type=act_type, name=name + '_relu2')
		conv2 = Conv(data=act2, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
					 use_bias=False, name=name + '_conv2')
		if use_se:
			# se begin
			body = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), name=name + '_se_pool1')(conv2)
			body = Conv(data=body, num_filter=num_filter // 16, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv1")
			body = Act(data=body, act_type=act_type, name=name + '_se_relu1')
			body = Conv(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv2")
			body = tf.keras.layers.Activation('sigmoid', name=name + "_se_sigmoid")(body)
			conv2 = tf.keras.layers.Multiply()([conv2, body])
		if dim_match:
			shortcut = data
		else:
			shortcut = Conv(data=act1, num_filter=num_filter, kernel=(1, 1), stride=stride, use_bias=False,
							name=name + '_sc')
		x = tf.keras.layers.Add()([conv2 + shortcut])
		return x


def residual_unit_v3(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
	"""Return ResNet Unit symbol for building ResNet
	Parameters
	----------
	data : str
		Input data
	num_filter : int
		Number of output channels
	bnf : int
		Bottle neck channels factor with regard to num_filter
	stride : tuple
		Stride used in convolution
	dim_match : Boolean
		True means channel number between input and output is the same, otherwise means differ
	name : str
		Base name of the operators
	workspace : int
		Workspace used in convolution operator
	"""
	use_se = kwargs.get('version_se', 1)
	bn_mom = kwargs.get('bn_mom', 0.9)
	workspace = kwargs.get('workspace', 256)
	memonger = kwargs.get('memonger', False)
	act_type = kwargs.get('version_act', 'prelu')
	# print('in unit3')
	if bottle_neck:
		bn1 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn1')(data)
		conv1 = Conv(data=bn1, num_filter=int(num_filter * 0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0),
					 use_bias=False, name=name + '_conv1')
		bn2 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn2')(conv1)
		act1 = Act(data=bn2, act_type=act_type, name=name + '_relu1')
		conv2 = Conv(data=act1, num_filter=int(num_filter * 0.25), kernel=(3, 3), stride=(1, 1), pad=(1, 1),
					 use_bias=False, name=name + '_conv2')
		bn3 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn3')(conv2)
		act2 = Act(data=bn3, act_type=act_type, name=name + '_relu2')
		conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1, 1), stride=stride, pad=(0, 0), use_bias=False, name=name + '_conv3')
		bn4 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn4')(conv3)

		if use_se:
			# se begin
			body = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), name=name + '_se_pool1')(bn4)
			body = Conv(data=body, num_filter=num_filter // 16, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv1")
			body = Act(data=body, act_type=act_type, name=name + '_se_relu1')
			body = Conv(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv2")

			body = tf.keras.layers.Activation('sigmoid', name=name + "_se_sigmoid")(body)
			bn4 = tf.keras.layers.Multiply()([bn4, body])
		# se end

		if dim_match:
			shortcut = data
		else:
			conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, use_bias=False,
						   name=name + '_conv1sc')
			shortcut = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_sc')(conv1sc)

		x = tf.keras.layers.Add()([bn4 + shortcut])
		return x
	else:
		bn1 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn1')(data)
		conv1 = Conv(data=bn1, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1),
					 use_bias=False, name=name + '_conv1')
		bn2 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn2')(conv1)
		act1 = Act(data=bn2, act_type=act_type, name=name + '_relu1')
		conv2 = Conv(data=act1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1),
					 use_bias=False, name=name + '_conv2')
		bn3 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn3')(conv2)
		if use_se:
			# se begin
			body = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), name=name + '_se_pool1')(bn3)
			body = Conv(data=body, num_filter=num_filter // 16, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv1")
			body = Act(data=body, act_type=act_type, name=name + '_se_relu1')
			body = Conv(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
						name=name + "_se_conv2")
			body = tf.keras.layers.Activation('sigmoid', name=name + "_se_sigmoid")(body)
			bn3 = tf.keras.layers.Multiply()([bn3, body])
		# se end

		if dim_match:
			shortcut = data
		else:
			conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, use_bias=False,
						   name=name + '_conv1sc')
			shortcut = tf.keras.layers.BatchNormalization(momentum=bn_mom, epsilon=2e-5, name=name + '_sc')(conv1sc)

		x = tf.keras.layers.Add()([bn3, shortcut])
		return x


def residual_unit_v3_x(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
	"""Return ResNeXt Unit symbol for building ResNeXt
	Parameters
	----------
	data : str
		Input data
	num_filter : int
		Number of output channels
	bnf : int
		Bottle neck channels factor with regard to num_filter
	stride : tuple
		Stride used in convolution
	dim_match : Boolean
		True means channel number between input and output is the same, otherwise means differ
	name : str
		Base name of the operators
	workspace : int
		Workspace used in convolution operator
	"""
	assert (bottle_neck)
	use_se = kwargs.get('version_se', 1)
	bn_mom = kwargs.get('bn_mom', 0.9)
	workspace = kwargs.get('workspace', 256)
	memonger = kwargs.get('memonger', False)
	act_type = kwargs.get('version_act', 'prelu')
	num_group = 32
	# print('in unit3')
	bn1 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn1')(data)
	conv1 = Conv(data=bn1, num_group=num_group, num_filter=int(num_filter * 0.5), kernel=(1, 1), stride=(1, 1),
				 pad=(0, 0),
				 use_bias=False, name=name + '_conv1')
	bn2 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn2')(conv1)
	act1 = Act(data=bn2, act_type=act_type, name=name + '_relu1')
	conv2 = Conv(data=act1, num_group=num_group, num_filter=int(num_filter * 0.5), kernel=(3, 3), stride=(1, 1),
				 pad=(1, 1),
				 use_bias=False, name=name + '_conv2')
	bn3 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn3')(conv2)
	act2 = Act(data=bn3, act_type=act_type, name=name + '_relu2')
	conv3 = Conv(data=act2, num_filter=num_filter, kernel=(1, 1), stride=stride, pad=(0, 0), use_bias=False, name=name + '_conv3')
	bn4 = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_bn4')(conv3)

	if use_se:
		# se begin
		body = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), name=name + '_se_pool1')(bn4)
		body = Conv(data=body, num_filter=num_filter // 16, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
					name=name + "_se_conv1")
		body = Act(data=body, act_type=act_type, name=name + '_se_relu1')
		body = Conv(data=body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
					name=name + "_se_conv2")
		body = tf.keras.layers.Activation('sigmoid', name=name + "_se_sigmoid")(body)
		bn4 = tf.keras.layers.Multiply()([bn4, body])
	# se end

	if dim_match:
		shortcut = data
	else:
		conv1sc = Conv(data=data, num_filter=num_filter, kernel=(1, 1), stride=stride, use_bias=False, name=name + '_conv1sc')
		shortcut = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name=name + '_sc')(conv1sc)

	x = tf.keras.layers.Add()([bn4 + shortcut])
	return x


def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
	uv = kwargs.get('version_unit', 3)
	version_input = kwargs.get('version_input', 1)
	if uv == 1:
		if version_input == 0:
			return residual_unit_v1(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
		else:
			return residual_unit_v1_L(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
	elif uv == 2:
		return residual_unit_v2(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
	elif uv == 4:
		return residual_unit_v4(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)
	else:
		return residual_unit_v3(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs)


def get_fc1(last_conv, num_classes, fc_type, input_channel=512):
	bn_mom = 0.9
	body = last_conv

	return body


def resnet(units, num_stages, filter_list, num_classes, bottle_neck):
	bn_mom = 0.9
	kwargs = {'version_se': 0,
			  'version_input': 1,
			  'version_output': "E",
			  'version_unit': 3,
			  'version_act': "prelu",
			  'bn_mom': bn_mom,
			  }
	"""Return ResNet symbol of
	Parameters
	----------
	units : list
		Number of units in each stage
	num_stages : int
		Number of stage
	filter_list : list
		Channel size of each stage
	num_classes : int
		Ouput size of symbol
	dataset : str
		Dataset type, only cifar10 and imagenet supports
	workspace : int
		Workspace used in convolution operator
	"""
	version_se = kwargs.get('version_se', 1)
	version_input = kwargs.get('version_input', 1)
	assert version_input >= 0
	version_output = kwargs.get('version_output', 'E')
	fc_type = version_output
	version_unit = kwargs.get('version_unit', 3)
	act_type = kwargs.get('version_act', 'prelu')
	memonger = kwargs.get('memonger', False)
	print(version_se, version_input, version_output, version_unit, act_type, memonger)
	num_unit = len(units)
	assert (num_unit == num_stages)
	data = tf.keras.layers.Input((112, 112, 3))
	body = Conv(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1),
				use_bias=False, name="conv0")
	body = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name='bn0')(body)
	body = Act(data=body, act_type=act_type, name='relu0')

	for i in range(num_stages):
		# if version_input==0:
		#  body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
		#                       name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, **kwargs)
		# else:
		#  body = residual_unit(body, filter_list[i+1], (2, 2), False,
		#    name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, **kwargs)
		body = residual_unit(body, filter_list[i + 1], (2, 2), False,
							 name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, **kwargs)
		for j in range(units[i] - 1):
			body = residual_unit(body, filter_list[i + 1], (1, 1), True, name='stage%d_unit%d' % (i + 1, j + 2),
								 bottle_neck=bottle_neck, **kwargs)

	if bottle_neck:
		body = Conv(data=body, num_filter=512, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
					use_bias=False, name="convd")
		body = tf.keras.layers.BatchNormalization(epsilon=2e-5, momentum=bn_mom, name='bnd')(body)
		body = Act(data=body, act_type=act_type, name='relu')  # relud?

	body = get_fc1(body, num_classes, fc_type)
	return data, body


def get_symbol(num_layers: int = 100):
	"""
	Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
	Original author Wei Wu
	"""
	num_classes = 85000
	if num_layers >= 500:
		filter_list = [64, 256, 512, 1024, 2048]
		bottle_neck = True
	else:
		filter_list = [64, 64, 128, 256, 512]
		bottle_neck = False
	num_stages = 4
	if num_layers == 18:
		units = [2, 2, 2, 2]
	elif num_layers == 34:
		units = [3, 4, 6, 3]
	elif num_layers == 49:
		units = [3, 4, 14, 3]
	elif num_layers == 50:
		units = [3, 4, 14, 3]
	elif num_layers == 74:
		units = [3, 6, 24, 3]
	elif num_layers == 90:
		units = [3, 8, 30, 3]
	elif num_layers == 98:
		units = [3, 4, 38, 3]
	elif num_layers == 99:
		units = [3, 8, 35, 3]
	elif num_layers == 100:
		units = [3, 13, 30, 3]
	elif num_layers == 134:
		units = [3, 10, 50, 3]
	elif num_layers == 136:
		units = [3, 13, 48, 3]
	elif num_layers == 140:
		units = [3, 15, 48, 3]
	elif num_layers == 124:
		units = [3, 13, 40, 5]
	elif num_layers == 160:
		units = [3, 24, 49, 3]
	elif num_layers == 101:
		units = [3, 4, 23, 3]
	elif num_layers == 152:
		units = [3, 8, 36, 3]
	elif num_layers == 200:
		units = [3, 24, 36, 3]
	elif num_layers == 269:
		units = [3, 30, 48, 8]
	else:
		raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

	input_layer, body = resnet(units=units,
				 num_stages=num_stages,
				 filter_list=filter_list,
				 num_classes=num_classes,
				 bottle_neck=bottle_neck)

	model = tf.keras.models.Model(input_layer, body)
	model.summary()

	return model
class BatchNormalization(tf.keras.layers.BatchNormalization):
	"""Make trainable=False freeze BN for real (the og version is sad).
	   ref: https://github.com/zzh8829/yolov3-tf2
	"""

	def call(self, x, training=False):
		if training is None:
			training = tf.constant(False)
		training = tf.logical_and(training, self.trainable)
		return super().call(x, training)


class MainModel:
	@tf.function
	def test_step_reg(self, x, y):
		logits, features = self.model([x, y], training=False)
		loss = self.loss_function(y, logits)

		reg_loss = tf.add_n(self.model.losses)

		return logits, features, loss, reg_loss

	@tf.function
	def train_step_reg(self, x, y):
		with tf.GradientTape() as tape:
			logits, features = self.model([x, y], training=True)

			loss = self.loss_function(y, logits)
			reg_loss = tf.add_n(self.model.losses)

			loss_all = tf.add(loss, reg_loss)

		gradients = tape.gradient(loss_all, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

		return logits, features, loss, reg_loss

	def change_learning_rate_of_optimizer(self, new_lr: float):
		self.optimizer.learning_rate = new_lr
		self.last_lr = new_lr

		assert self.optimizer.learning_rate == self.optimizer.lr

		return True

	def __init__(self):
		self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
		self.last_lr = None

	@tf.function
	def train_step(self, x, y):
		with tf.GradientTape() as tape:
			logits, features = self.model([x, y], training=True)
			loss = self.loss_function(y, logits)

		gradients = tape.gradient(loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

		return logits, features, loss

	@tf.function
	def test_step(self, x, y):
		logits, features = self.model([x, y], training=False)
		loss = self.loss_function(y, logits)

		return logits, features, loss

	def turn_softmax_into_arcface(self, num_classes: int):
		label_input_layer = tf.keras.layers.Input((None,), dtype=tf.int64)

		x = ArcFaceLayer(num_classes=num_classes, name="arcfaceLayer")(self.model.layers[-3].output, label_input_layer)

		self.model = tf.keras.models.Model([self.model.layers[0].input, label_input_layer], [x, self.model.layers[-3].output])
		self.model.summary()

	def change_regularizer_l(self, new_value: float = 5e-4):
		for layer in self.model.layers:
			if "Conv" in str(layer):
				layer.kernel_regularizer = tf.keras.regularizers.l2(new_value)

			elif "BatchNorm" in str(layer):
				layer.gamma_regularizer = tf.keras.regularizers.l2(new_value)
				layer.momentum = 0.9
				layer.epsilon = 2e-5

			elif "PReLU" in str(layer):
				layer.alpha_regularizer = tf.keras.regularizers.l2(new_value)

			elif "Dense" in str(layer):
				layer.kernel_regularizer = tf.keras.regularizers.l2(new_value)

			elif "arcfaceLayer" in str(layer):
				layer.kernel_regularizer = tf.keras.regularizers.l2(new_value)

		self.model = tf.keras.models.model_from_json(self.model.to_json())  # To apply regularizers
		print(f"[*] Kernel regularizer value set to --> {new_value}")

	def __call__(self, input_shape, weights: str = None, num_classes: int = 10, learning_rate: float = 0.1,
	             regularizer_l: float = 5e-4, weight_path: str = None,
	             pooling_layer: tf.keras.layers.Layer = tf.keras.layers.GlobalAveragePooling2D,
	             create_model: bool = True, use_arcface: bool = True,
	             optimizer="ADAM"):

		self.last_lr = learning_rate

		if optimizer == "ADAM":
			self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=0.1)
			print("[*] ADAM chosen as optimizer")
		elif optimizer == "SGD":
			self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
			print("[*] SGD chosen as optimizer")
		elif optimizer == "MOMENTUM":
			self.optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
			# MomentumOptimizer is not recommended, it is from TF 1.x makes problem at learning rate change, i will update if TF 2.x version comes out
			print("[*] MomentumOptimizer chosen as optimizer")
		else:
			raise Exception(f"{optimizer} is not a valid name! Go with either ADAM, SGD or MOMENTUM")

		if create_model:
			label_input_layer = tf.keras.layers.Input((None,), dtype=tf.int64)
			self.model = self.get_model(input_shape=input_shape, weights=weights)
			self.model.trainable = True

			self.change_regularizer_l(regularizer_l)
			# ACCORDING TO ARCFACE PAPER
			x = pooling_layer()(self.model.layers[-1].output)
			x = BatchNormalization(momentum=0.9, epsilon=2e-5)(x)
			x = tf.keras.layers.Dropout(0.4)(x)
			x1 = tf.keras.layers.Dense(512, activation=None, name="features_without_bn", use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(regularizer_l))(x)
			x = BatchNormalization(momentum=0.9, scale=False, epsilon=2e-5)(x1)

			if use_arcface:
				x = ArcFaceLayer(num_classes=num_classes, arc_m=0.5, arc_s=64., regularizer_l=regularizer_l, name="arcfaceLayer")(x, label_input_layer)
			else:
				x = tf.keras.layers.Dense(num_classes, activation=None, name="classificationLayer", kernel_regularizer=tf.keras.regularizers.l2(regularizer_l))(x)

			self.model = tf.keras.models.Model([self.model.layers[0].input, label_input_layer], [x, x1], name=f"{self.__name__}-ArcFace")
			self.model.summary()

			try:
				self.model.load_weights(weight_path)
				print("[*] WEIGHTS FOUND FOR MODEL, LOADING...")
			except Exception as e:
				print(e)
				print("[*] THERE IS NO WEIGHT FILE FOR MODEL, INITIALIZING...")


class ResNet50(MainModel):
	@property
	def __name__(self):
		return "ResNet50"

	def __init__(self, **kwargs):
		super(ResNet50, self).__init__(**kwargs)

	def get_model(self, input_shape, weights: str = None, **kwargs):
		return get_symbol(50)


class ResNet101(MainModel):
	@property
	def __name__(self):
		return "ResNet101"

	def __init__(self, **kwargs):
		super(ResNet101, self).__init__(**kwargs)

	def get_model(self, input_shape, weights: str = None, **kwargs):
		return get_symbol(100)


class ResNet152(MainModel):
	@property
	def __name__(self):
		return "ResNet101"

	def __init__(self, **kwargs):
		super(ResNet152, self).__init__(**kwargs)

	def get_model(self, input_shape, weights: str = None, **kwargs):
		return get_symbol(152)


class EfficientNetFamily(MainModel):
	all_models = [
		efn.EfficientNetB0,
		efn.EfficientNetB1,
		efn.EfficientNetB2,
		efn.EfficientNetB3,
		efn.EfficientNetB4,
		efn.EfficientNetB5,
		efn.EfficientNetB6,
		efn.EfficientNetB7,
	]

	@property
	def __name__(self):
		return f"EfficientNetB{self.model_id}"

	def __init__(self, model_id: int, **kwargs):
		self.model_id = model_id
		if not 0 <= self.model_id <= 7:
			raise ValueError(f"model_id must be \"0 <= model_id <=7\", yours({self.model_id}) is not valid!")

		super(EfficientNetFamily, self).__init__(**kwargs)

	def get_model(self, input_shape, weights: str = None, **kwargs):
		return self.all_models[self.model_id](input_shape=input_shape, weights=weights, include_top=False)


class Xception(MainModel):
	@property
	def __name__(self):
		return "Xception"

	def __init__(self, **kwargs):
		super(Xception, self).__init__(**kwargs)

	def get_model(self, input_shape, weights: str = None, **kwargs):
		return tf.keras.applications.Xception(input_shape=input_shape, weights=weights, include_top=False)

class DataEngineTypical:
	def make_label_map(self):
		self.label_map = {}

		for i, class_name in enumerate(tf.io.gfile.listdir(self.main_path)):
			self.label_map[class_name] = i

		self.reverse_label_map = {v: k for k, v in self.label_map.items()}

	def path_yielder(self):
		for class_name in tf.io.gfile.listdir(self.main_path):
			if not "tfrecords" in class_name:
				for path_only in tf.io.gfile.listdir(self.main_path + class_name):
					yield (self.main_path + class_name + "/" + path_only, self.label_map[class_name])

	def image_loader(self, image):
		image = tf.io.read_file(image)
		image = tf.io.decode_jpeg(image, channels=3)
		image = tf.image.resize(image, (112, 112), method="nearest")
		image = tf.image.random_flip_left_right(image)

		return (tf.cast(image, tf.float32) - 127.5) / 128.

	def mapper(self, path, label):
		return (self.image_loader(path), label)

	def __init__(self, main_path: str, batch_size: int = 16, buffer_size: int = 10000, epochs: int = 1,
	             reshuffle_each_iteration: bool = False, test_batch=64,
	             map_to: bool = True):
		self.main_path = main_path.rstrip("/") + "/"
		self.make_label_map()

		self.dataset_test = None
		if test_batch > 0:
			reshuffle_each_iteration = False
			print(f"[*] reshuffle_each_iteration set to False to create a appropriate test set, this may cancelled if tf.data will fixed.")

		self.dataset = tf.data.Dataset.from_generator(self.path_yielder, (tf.string, tf.int64))
		if buffer_size > 0:
			self.dataset = self.dataset.shuffle(buffer_size, reshuffle_each_iteration=reshuffle_each_iteration, seed=42)

		if map_to:
			self.dataset = self.dataset.map(self.mapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
		self.dataset = self.dataset.batch(batch_size, drop_remainder=True)

		if test_batch > 0:
			self.dataset_test = self.dataset.take(int(test_batch))
			self.dataset = self.dataset.skip(int(test_batch))

		self.dataset = self.dataset.repeat(epochs)


class DataEngineTFRecord:
	def image_loader(self, image_raw):
		image = tf.image.decode_jpeg(image_raw, channels=3)
		image = tf.image.resize(image, (112, 112), method="nearest")
		image = tf.image.random_flip_left_right(image)

		return (tf.cast(image, tf.float32) - 127.5) / 128.

	def mapper(self, tfrecord_data):
		features = {'image_raw': tf.io.FixedLenFeature([], tf.string), 'label': tf.io.FixedLenFeature([], tf.int64)}
		features = tf.io.parse_single_example(tfrecord_data, features)

		return self.image_loader(features['image_raw']), tf.cast(features['label'], tf.int64)

	def __init__(self, tf_record_path: str, batch_size: int = 16, epochs: int = 10, buffer_size: int = 50000,
	             reshuffle_each_iteration: bool = True,
	             test_batch=64, map_to: bool = True):
		self.dataset_test = None
		if test_batch > 0:
			reshuffle_each_iteration = False
			print(
				f"[*] reshuffle_each_iteration set to False to create a appropriate test set, this may cancelled if tf.data will fixed.")
		self.tf_record_path = tf_record_path

		self.dataset = tf.data.TFRecordDataset(self.tf_record_path)
		if buffer_size > 0:
			self.dataset = self.dataset.shuffle(buffer_size, reshuffle_each_iteration=reshuffle_each_iteration, seed=42)

		if map_to:
			self.dataset = self.dataset.map(self.mapper, num_parallel_calls=tf.data.experimental.AUTOTUNE)
		self.dataset = self.dataset.batch(batch_size, drop_remainder=True)

		if test_batch > 0:
			self.dataset_test = self.dataset.take(int(test_batch))
			self.dataset = self.dataset.skip(int(test_batch))

		self.dataset = self.dataset.repeat(epochs)
import os
import cv2
import bcolz
import numpy as np
import tensorflow as tf
import tqdm
from sklearn.model_selection import KFold


def l2_norm(x, axis=1):
    """l2 norm"""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    output = x / norm

    return output


def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=os.path.join(path, name), mode='r')
    issame = np.load('{}/{}_list.npy'.format(path, name))

    return carray, issame


def get_lfw_data(data_path):
    """get validation data"""
    _lfw, _lfw_issame = get_val_pair(data_path, 'lfw_align_112/lfw')

    return _lfw, _lfw_issame


def get_val_data(data_path):
    """get validation data"""
    _lfw, _lfw_issame = get_val_pair(data_path, 'lfw_align_112/lfw')
    _agedb_30, _agedb_30_issame = get_val_pair(data_path, 'AgeDB/agedb_30')
    _cfp_fp, _cfp_fp_issame = get_val_pair(data_path, 'cfp_align_112/cfp_fp')

    return _lfw, _agedb_30, _cfp_fp, _lfw_issame, _agedb_30_issame, _cfp_fp_issame


def hflip_batch(imgs):
    return imgs[:, :, ::-1, :]


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame),
                               np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    _acc = float(tp + tn) / dist.size
    return tpr, fpr, _acc


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame,
                  nrof_folds=10):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds,))
    best_thresholds = np.zeros((nrof_folds,))
    indices = np.arange(nrof_pairs)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds,))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)

        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = \
                calculate_accuracy(threshold,
                                   dist[test_set],
                                   actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index],
            dist[test_set],
            actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds


def evaluate(embeddings, actual_issame, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy, best_thresholds = calculate_roc(
        thresholds, embeddings1, embeddings2, np.asarray(actual_issame),
        nrof_folds=nrof_folds)

    return tpr, fpr, accuracy, best_thresholds


def perform_val_arcface(embedding_size, batch_size, model,
                        carray, issame, nrof_folds=10, is_ccrop=False, is_flip=True):
    """perform val"""
    embeddings = np.zeros([len(carray), embedding_size])

    for idx in tqdm.tqdm(range(0, len(carray), batch_size)):
        batch = carray[idx:idx + batch_size]
        batch = np.transpose(batch, [0, 2, 3, 1])
        b, g, r = tf.split(batch, 3, axis=-1)
        batch = tf.concat([r, g, b], -1)
        if is_flip:
            flipped = hflip_batch(batch)
            emb_batch = model([batch, tf.ones((batch.shape[0],), dtype=tf.int64)], training=False)[-1] + model([flipped, tf.ones((batch.shape[0],), dtype=tf.int64)], training=False)[-1]
            embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
        else:
            emb_batch = model([batch, tf.ones((batch.shape[0],), dtype=tf.int64)], training=False)[-1]
            embeddings[idx:idx + batch_size] = l2_norm(emb_batch)

    tpr, fpr, accuracy, best_thresholds = evaluate(
        embeddings, issame, nrof_folds)

    return accuracy.mean(), best_thresholds.mean()


def perform_val(embedding_size, batch_size, model,
                carray, issame, nrof_folds=10, is_ccrop=False, is_flip=True):
    """perform val"""
    embeddings = np.zeros([len(carray), embedding_size])

    for idx in tqdm.tqdm(range(0, len(carray), batch_size)):
        batch = carray[idx:idx + batch_size]
        batch = np.transpose(batch, [0, 2, 3, 1])
        b, g, r = tf.split(batch, 3, axis=-1)
        batch = tf.concat([r, g, b], -1)
        if is_flip:
            flipped = hflip_batch(batch)
            emb_batch = model(batch, training=False) + model(flipped, training=False)
            embeddings[idx:idx + batch_size] = l2_norm(emb_batch)
        else:
            emb_batch = model(batch, training=False)
            embeddings[idx:idx + batch_size] = l2_norm(emb_batch)

    tpr, fpr, accuracy, best_thresholds = evaluate(
        embeddings, issame, nrof_folds)

    return accuracy.mean(), best_thresholds.mean()
class Trainer:
	@staticmethod
	def get_wrong(y_real, y_pred):
		return tf.where(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(y_pred), -1), y_real), tf.float32) == 00)

	@staticmethod
	def calculate_accuracy(y_real, y_pred):
		return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(y_pred), axis=1), y_real), dtype=tf.float32))

	def only_test(self, dataset_test=None, display_wrong_images: bool = False):
		if dataset_test is None:
			if self.dataset_engine.dataset_test is None:
				raise Exception("there is no defined test dataset")

			dataset_test = self.dataset_engine.dataset_test

		acc_mean = tf.keras.metrics.Mean()
		loss_mean = tf.keras.metrics.Mean()
		wrong_images = []

		for i, (x, y) in enumerate(dataset_test):
			logits, features, loss, reg_loss = self.model_engine.test_step_reg(x, y)
			accuracy = self.calculate_accuracy(y, logits)
			if accuracy < 1.0:
				images = x.numpy()[self.get_wrong(y, logits).numpy()][0]
				[wrong_images.append(image) for image in images]

			acc_mean(accuracy)
			loss_mean(loss)

			print(f"[*] Step {i}, Accuracy --> %{accuracy} || Loss --> {loss} || Reg Loss --> {reg_loss}")

		if display_wrong_images and len(wrong_images) > 0:
			self.tensorboard_engine.initialize(delete_if_exists=False)
			print(f"[*] TensorBoard initialized on {self.tensorboard_engine.logdir}")

			self.tensorboard_engine.add_images(f"wrong images from 'only_test' function", tf.convert_to_tensor(wrong_images), 0)
			print(f"[*] Wrong images({len(wrong_images)}) added to TensorBoard")

		print(f"\n\n[*] Accuracy Mean --> %{acc_mean.result().numpy()} || Loss Mean --> {loss_mean.result().numpy()}")

		return acc_mean, loss_mean, wrong_images

	def __init__(self, model_engine, dataset_engine, tensorboard_engine, use_arcface: bool,
	             learning_rate: float = 0.01,
	             model_path: str = "classifier_model.tf",
	             pooling_layer: tf.keras.layers.Layer = tf.keras.layers.GlobalAveragePooling2D,
	             lr_step_dict: dict = None,
	             optimizer: str = "ADAM", test_only_lfw: bool = True):
		self.model_path = model_path
		self.model_engine = model_engine
		self.dataset_engine = dataset_engine
		self.tensorboard_engine = tensorboard_engine
		self.use_arcface = use_arcface
		self.lr_step_dict = lr_step_dict

		self.num_classes = 85742  # 85742 for MS1MV2, 10575 for Casia, 105 for MINE
		tf.io.gfile.makedirs("/".join(self.model_path.split("/")[:-1]))

		self.tb_delete_if_exists = True
		if self.use_arcface:
			if not test_only_lfw:
				self.lfw, self.agedb_30, self.cfp_fp, self.lfw_issame, self.agedb_30_issame, self.cfp_fp_issame = get_val_data("../datasets/")
			else:
				self.lfw, self.lfw_issame = get_lfw_data("/content/")

		if self.lr_step_dict is not None:
			print("[*] LEARNING RATE WILL BE CHECKED WHEN step\\alfa_divided_ten == 0")
			learning_rate = list(self.lr_step_dict.values())[0]

		self.model_engine(
			input_shape=(112, 112, 3),
			weights=None,  # "imagenet" or None, not available for InceptionResNetV1
			num_classes=self.num_classes,  # 85742 for MS1MV2, 10575 for Casia, 105 for MINE
			learning_rate=learning_rate,
			regularizer_l=5e-4,  # weight decay, train once with 5e-4 and then try something lower such 1e-5
			pooling_layer=pooling_layer,  # Recommended: Flatten
			create_model=True,  # if you have a H5 file with config set this to zero and load model to self.model_engine.model
			use_arcface=self.use_arcface,  # set False if you want to train it as regular classification
			weight_path=self.model_path,  # paths of weights file(h5 or tf), it is okay if doesn't exists
			optimizer=optimizer  # Recommended: SGD
		)

	def test_on_val_data(self, is_ccrop: bool = False, step_i: int = 1, alfa_multiplied_ten: int = 1):
		step = int(alfa_multiplied_ten / step_i)

		print("-----------------------------------")
		acc_lfw, best_th = perform_val_arcface(512, 64, self.model_engine.model, self.lfw, self.lfw_issame, is_ccrop=is_ccrop)
		print(f"[*] Results on LFW, Accuracy --> {acc_lfw} || Best Threshold --> {best_th}")
		print("-----------------------------------")
		self.tensorboard_engine.add_with_step({"LFW": acc_lfw}, step=step)

	def __call__(self, max_iteration: int = None, alfa_step=1000, qin: int = 10):
		if max_iteration is not None and max_iteration <= 0:
			max_iteration = None

		alfa_divided_ten = int(alfa_step / 10)
		alfa_multiplied_qin = int(alfa_step * qin)

		print(f"[*] Possible maximum step: {tf.data.experimental.cardinality(self.dataset_engine.dataset)}\n")

		acc_mean = tf.keras.metrics.Mean()
		loss_mean = tf.keras.metrics.Mean()

		self.tensorboard_engine.initialize(
			delete_if_exists=self.tb_delete_if_exists
		)
		print(f"[*] TensorBoard initialized on {self.tensorboard_engine.logdir}")

		for i, (x, y) in enumerate(self.dataset_engine.dataset):
			logits, features, loss, reg_loss = self.model_engine.train_step_reg(x, y)
			accuracy = self.calculate_accuracy(y, logits)
			acc_mean(accuracy)
			loss_mean(loss)

			self.tensorboard_engine({"loss": loss, "reg_loss": reg_loss, "accuracy": accuracy})

			if i % alfa_divided_ten == 0:
				if i % alfa_step == 0 and i > 10:
					self.model_engine.model.save_weights(self.model_path)
					print(f"[{i}] Model saved to {self.model_path}")

				print(f"[{i}] Loss: {loss_mean.result().numpy()} || Reg Loss: {reg_loss.numpy()} || Accuracy: %{acc_mean.result().numpy()} || LR: {self.model_engine.optimizer.learning_rate.numpy()}")
				acc_mean.reset_states()
				loss_mean.reset_states()
				if self.lr_step_dict is not None:
					lower_found = False
					for key in self.lr_step_dict:
						if i < int(key):
							lower_found = True
							lr_should_be = self.lr_step_dict[key]
							if lr_should_be != self.model_engine.last_lr:
								self.model_engine.change_learning_rate_of_optimizer(lr_should_be)
								print(f"[{i}] Learning Rate set to --> {lr_should_be}")

							break

					if not lower_found:
						print(f"[{i}] Reached to given maximum steps in 'lr_step_dict'({list(self.lr_step_dict.keys())[-1]})")
						self.model_engine.model.save_weights(self.model_path)
						print(f"[{i}] Model saved to {self.model_path}, end of training.")
						break

				if i % alfa_multiplied_qin == 0 and self.dataset_engine.dataset_test is not None and i > 10:
					for x_test, y_test in self.dataset_engine.dataset_test:
						logits, features, loss, reg_loss = self.model_engine.test_step_reg(x_test, y_test)
						accuracy = self.calculate_accuracy(y, logits)

						self.tensorboard_engine({"val. loss": loss, "val. accuracy": accuracy})

						acc_mean(accuracy)
						loss_mean(loss)

					print(f"[{i}] Val. Loss --> {loss_mean.result().numpy()} || Val. Accuracy --> %{acc_mean.result().numpy()}")
					acc_mean.reset_states()
					loss_mean.reset_states()

				if i % alfa_multiplied_qin == 0 and self.use_arcface and i > 10:
					self.test_on_val_data(False, i, alfa_multiplied_qin)
					self.save_final_model()
					print("[*] Final model saved")

				if max_iteration is not None and i >= max_iteration:
					print(f"[{i}] Reached to given maximum iteration({max_iteration})")
					self.model_engine.model.save_weights(self.model_path)
					print(f"[{i}] Model saved to {self.model_path}, end of training.")
					break

		if max_iteration is None:
			print(f"[*] Reached to end of dataset")
			self.model_engine.model.save_weights(self.model_path)
			print(f"[*] Model saved to {self.model_path}, end of training.")

	def save_final_model(self, path: str = "arcface_final.h5", n: int = -4):
		m = tf.keras.models.Model(self.model_engine.model.layers[0].input, self.model_engine.model.layers[n].output)

		m.save(path)
		print(f"[*] Final feature extractor saved to {path}")
%load_ext tensorboard.notebook
%tensorboard --logdir classifier_tensorboard --port 8008
TDOM = DataEngineTFRecord(
  "faces_emore/tran.tfrecords",
  batch_size=128,
  epochs=-1,  # set to -1 so it can stream forever
  buffer_size=30000,
  reshuffle_each_iteration=True,
  test_batch=0  # set to 0 if you are using ArcFace
)  # TDOM for "Tensorflow Dataset Object Manager"
TBE = TensorBoardCallback(
  logdir="classifier_tensorboard"
)  # TBE for TensorBoard Engine
ME = ResNet50()  # Model architecture, ResNet50 Recommended
k_value: float = 4.  # recommended --> (512 / TDOM.batch_size)
trainer = Trainer(
  model_engine=ME,
  dataset_engine=TDOM,
  tensorboard_engine=TBE,
  use_arcface=True,
  learning_rate=0.004,
  model_path="my_ms1m_arcResnetIR50/model.tf",
  optimizer="SGD",
  lr_step_dict={
    int(60000 * k_value): 0.004,
    int(80000 * k_value): 0.0005,
    int(100000 * k_value): 0.0003,
    int(110000 * k_value): 0.0001,
  },
  pooling_layer=tf.keras.layers.Flatten
)

trainer(max_iteration=-1, alfa_step=5000, qin=2)
trainer.save_final_model(path="my_ms1m_arcResnetIR50_arcface_final.h5")
