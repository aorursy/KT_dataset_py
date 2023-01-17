!pip install onnx2keras
import torch
import torch.nn as nn
from torchvision.models import resnet18
import onnx
from onnx2keras import onnx_to_keras
import numpy as np

import os
import cv2
import glob

import tensorflow as tf
import keras
from keras.models import load_model, save_model
from keras.layers import Input, GlobalAveragePooling2D, GlobalMaxPooling2D
import keras.backend as K
from keras.models import Model, load_model
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class BaseNet(nn.Module):   
    def __init__(self, features):
        super(BaseNet, self).__init__()
        self.output_dim = 512
        self.features = nn.Sequential(*features)
        self.pool = nn.AvgPool2d(kernel_size = 1, stride = (4, 4))
        self.flatten = Flatten()
        self.fc1 = nn.Linear(512, self.output_dim, bias = True)
    
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x
resnet = resnet18(pretrained = True)
features = list(resnet.children())[:-2]
model = BaseNet(features)
model.to(device)

print(model)
dummy_input = torch.randn(1, 3, 128, 128, device='cpu')
input_names = ['input_image']
output_names = ['global_descriptor']

torch.onnx.export(model, dummy_input, "resnet18.onnx", verbose=True, input_names=input_names, output_names=output_names)
onnx_model = onnx.load('resnet18.onnx')
k_model = onnx_to_keras(onnx_model, ['input_image'], change_ordering = True)
k_model.summary()
input_image = Input((128,128,3))
output = k_model(input_image)

model = Model(inputs=[input_image], outputs=[output])
model.summary()
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = model
    
    @tf.function(input_signature=[
      tf.TensorSpec(shape=[None, None, 3], dtype=tf.uint8, name='input_image')
    ])
    def call(self, input_image):
        output_tensors = {}
        
        # resizing
        im = tf.image.resize(input_image, (128,128))
        
        # preprocessing
        im = preprocess_input(im)
        
        extracted_features = self.model(tf.convert_to_tensor([im], dtype=tf.uint8))[0]
        output_tensors['global_descriptor'] = tf.identity(extracted_features, name='global_descriptor')
        return output_tensors
m = MyModel()
served_function = m.call
tf.saved_model.save(m, export_dir="./my_model", signatures={'serving_default': served_function})
from zipfile import ZipFile

with ZipFile('submission.zip','w') as zip:           
    zip.write('./my_model/saved_model.pb', arcname='saved_model.pb') 
    zip.write('./my_model/variables/variables.data-00000-of-00001', arcname='variables/variables.data-00000-of-00001')
    zip.write('./my_model/variables/variables.data-00000-of-00001', arcname='variables/variables.index') 