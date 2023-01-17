from keras.applications.resnet50 import ResNet50
import os
import shutil
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
conv_base = ResNet50(weights='imagenet',
                    include_top=False)
conv_base.summary()
