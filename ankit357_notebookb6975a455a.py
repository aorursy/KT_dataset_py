# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)'
import matplotlib.pyplot as plt
from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.applications.resnet import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from tensorflow.python.keras import optimizers


import cv2
import math
from IPython.display import clear_output
%matplotlib inline

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
IMG_SIZE = 224
TRAIN_BATCH_SIZE = 77
TEST_BATCH_SIZE = 1
from keras.models import load_model
trained_model_l = load_model('../input/model1/model.h5')
from keras.models import load_model
trained_model_s = load_model('../input/model2/model2.h5')
import pickle
a_file = open("../input/labels/data.pkl", "rb")

output = pickle.load(a_file)
label_dict_l = output.copy()
import pickle
a_file = open("../input/labels/data2.pkl", "rb")

output = pickle.load(a_file)
label_dict_s = output.copy()
def draw_prediction( frame, class_string ):
    x_start = frame.shape[1] -600
    cv2.putText(frame, class_string, (x_start, 75), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0), 2, cv2.LINE_AA)
    return frame
def prepare_image_for_prediction( img):
   
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    # The below function inserts an additional dimension at the axis position provided
    img = np.expand_dims(img, axis=0)
    # perform pre-processing that was done when resnet model was trained.
    return preprocess_input(img)
def get_display_string(pred_class, label_dict):
    txt = ""
    for c, confidence in pred_class:
        txt += label_dict[c]
        if c :
            txt += '['+ str(confidence) +']'
    #print("count="+str(len(pred_class)) + " txt:" + txt)
    return txt
def predict(  model, video_path, filename, label_dict ):
    
    vs = cv2.VideoCapture(video_path)
    fps = math.floor(vs.get(cv2.CAP_PROP_FPS))
    ret_val = True
    writer = 0
    
    while True:
        ret_val, frame = vs.read()
        if not ret_val:
            break
       
        resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame_for_pred = prepare_image_for_prediction( resized_frame )
        pred_vec = model.predict(frame_for_pred)
        #print(pred_vec)
        pred_class =[]
        confidence = np.round(pred_vec.max(),2) 
        
        if confidence > 0.4:
            pc = pred_vec.argmax()
            pred_class.append( (pc, confidence) )
        else:
            pred_class.append( (0, 0) )
        if pred_class:
            txt = get_display_string(pred_class, label_dict)       
            frame = draw_prediction( frame, txt )
        #print(pred_class)
        #plt.axis('off')
        #plt.imshow(frame)
        #plt.show()
        #clear_output(wait = True)
        if not writer:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(filename, fourcc, fps,(frame.shape[1], frame.shape[0]), True)
            
        # write the out
        writer.write(frame)
        
    vs.release()
    writer.release()

def predict(  model, video_path, filename, label_dict, model2, label_dict2 ):
    
    vs = cv2.VideoCapture(video_path)
    fps = math.floor(vs.get(cv2.CAP_PROP_FPS))
    ret_val = True
    writer = 0
    
    while True:
        ret_val, frame = vs.read()
        if not ret_val:
            break
       
        resized_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame_for_pred = prepare_image_for_prediction( resized_frame )
        pred_vec = model.predict(frame_for_pred)
        #print(pred_vec)
        pred_class =[]
        confidence = np.round(pred_vec.max(),2) 
        if confidence > 0.4:
            pc = pred_vec.argmax()
            pred_class.append( (pc, confidence) )
        else:
            pred_class.append( (0, 0) )
        
        print(pred_class)        
        if pred_class[0][0] == 3:
            pred_vec = model2.predict(frame_for_pred)
            #print(pred_vec)
            pred_class =[]
            confidence = np.round(pred_vec.max(),2) 
            if confidence > 0.4:
                pc = pred_vec.argmax()
                pred_class.append( (pc, confidence) )
            else:
                pred_class.append( (0, 0) )
            
            print(pred_class)
            if pred_class:
                txt = get_display_string(pred_class, label_dict2)       
                frame = draw_prediction( frame, txt )
            #print(pred_class)
            #plt.axis('off')
            #plt.imshow(frame)
            #plt.show()
            #clear_output(wait = True)
            if not writer:
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(filename, fourcc, fps,(frame.shape[1], frame.shape[0]), True)
            
            # write the out
            writer.write(frame)
                              
        else:
            if pred_class:
                txt = get_display_string(pred_class, label_dict)       
                frame = draw_prediction( frame, txt )
            #print(pred_class)
            #plt.axis('off')
            #plt.imshow(frame)
            #plt.show()
            #clear_output(wait = True)
            if not writer:
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(filename, fourcc, fps,(frame.shape[1], frame.shape[0]), True)
            
            # write the out
            writer.write(frame)
        
    vs.release()
    writer.release()
                                  
video_path = '../input/test-video/test.mp4'
predict ( trained_model_l, video_path, 'testres.avi',  label_dict_l )
video_path = '../input/test-video/test.mp4'
predict ( trained_model_l, video_path, 'testres2.avi',  label_dict_l, trained_model_s, label_dict_s)
