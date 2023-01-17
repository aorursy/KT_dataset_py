import tensorflow as tf
from tensorflow.keras.models import model_from_json
#from tensorflow.python.keras.backend import set_session
import numpy as np
import json
import torch

#config = tf.compat.v1.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.15
#session = tf.compat.v1.Session(config=config)
#set_session(session)
from collections import OrderedDict 

class SkinLesionTypeDetectionModel(object):

    SKIN_LESION_TYPE_LIST = ['Actinic Keratoses','Basal Cell Carcinoma','Benign Keratosis',
  'Dermatofibroma','Melanoma','Melanocytic Nevi','Vascular skin lesion']

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        #self.loaded_model.compile()
        #self.loaded_model.predict()
        
    def predict_skin_lesion_type(self, img):
        #global session
        #set_session(session)
        self.preds = self.loaded_model.predict(img)
        
        print(self.preds)
        dict_ = {k:v for k,v in zip (SkinLesionTypeDetectionModel.SKIN_LESION_TYPE_LIST,np.squeeze(self.preds))}
        dict1 = {k: v for k, v in sorted(dict_.items(), key=lambda item: item[1])}
        
        pred1 = dict1.popitem()
        print("Highest Probability:  ",pred1)
        pred2 = dict1.popitem()
        print("Second Highest Probability: ",pred2)
        pred3 = dict1.popitem()
        print("Third Highest Probability: ",pred3)
        print(np.argmax(self.preds))
        #return SkinLesionTypeDetectionModel.SKIN_LESION_TYPE_LIST[np.argmax(self.preds)]
        
        return pred1, pred2, pred3
import cv2
path = "../input/skin-cancer-trained-model-and-weights/"
model = SkinLesionTypeDetectionModel(path+"model.json", path+"model.h5")
img = cv2.imread("../input/image4/4.jpg",3)
roi = cv2.resize(img, (224,224))
#pred = model.predict_skin_lesion_type(roi)


image = tf.keras.preprocessing.image.load_img("../input/image4/4.jpg", 
                                              target_size=(224, 224))
image = tf.keras.preprocessing.image.img_to_array(image)
image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
image = np.expand_dims(image, axis=0)
pred1, pred2, pred3 = model.predict_skin_lesion_type(image)

print(pred1, pred2, pred3)
from flask import Flask, request, render_template, Response
#from model import SkinLesionTypeDetectionModel

import numpy as np
import tensorflow as tf
import os
import string
import random

PATH = "../input/skin-cancer-trained-model-and-weights/"
OUTPUT_DIR = 'static'

model = SkinLesionTypeDetectionModel(PATH+"model.json", PATH+"model.h5")

app = Flask(__name__)

def generate_filename():
    return ''.join(random.choices(string.ascii_lowercase, k=20)) + '.jpg'

def get_prediction(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    pred1, pred2, pred3 = model.predict_skin_lesion_type(image)
    
    print(pred1, pred2, pred3)
    return pred1, pred2, pred3

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            if uploaded_file.filename[-3:] in ['jpg', 'png']:
                image_path = os.path.join(OUTPUT_DIR, generate_filename())
                uploaded_file.save(image_path)
                
                """"""
                pred1 = ('Melanoma', 0.5374107) 
                pred2 = ('Basal Cell Carcinoma', 0.21953377) 
                pred3 = ('Dermatofibroma', 0.1933243)
                """"""
                #pred1, pred2, pred3 = get_prediction(image_path)
                result = {
                    'highest_class_name': pred1[0],
                    'highest_prob':pred1[1],
                    
                    'second_highest_class_name': pred2[0],
                    'second_highest_prob':pred2[1],
                    
                    'third_highest_class_name': pred3[0],
                    'third_highest_prob':pred3[1],
                    'path_to_image': image_path
                }
                return render_template('show.html', result=result)
    return render_template('index.html')

'''@app.route('/skin_lesion_image/<img>',methods=['POST'])
def skin_lesion_image(img):
    print("here")
    return Response(gen(img))'''

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True,use_reloader=False)
