# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from fastai import *

from fastai.vision import *
classes = ['teddys','grizzly','black']
folder = 'black'

file = 'urls_black.txt'
path = Path('data/bears')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)
!cp ../input/* {path}/
download_images(path/file, dest, max_pics=300)
folder = 'teddys'

file = 'urls_teddys.txt'
path = Path('data/bears')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)
download_images(path/file, dest, max_pics=300)
folder = 'grizzly'

file = 'urls_grizzly.txt'
path = Path('data/bears')

dest = path/folder

dest.mkdir(parents=True, exist_ok=True)
download_images(path/file, dest, max_pics=300)
#path.ls()
for c in classes:

    print(c)

    verify_images(path/c, delete=True, max_size=500)
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

         ds_tfms=get_transforms(), size=224, num_workers=0).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(3,4))

#data.show_batch(rows=3,fig_size(7,8),num_workers=4)
data.classes, data.c, len(data.valid_ds), len(data.train_ds)
learn = cnn_learner(data, models.resnet50, metrics=error_rate)
learn.fit_one_cycle(5)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4,max_lr=slice(1e-5, 1e-3))
learn.save('stage-2')
interp=ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
from fastai.widgets import *
ds, idxs = DatasetFormatter().from_toplosses(learn, n_imgs=15)
interp.plot_top_losses(9, figsize=(15,11))

ImageCleaner(ds, idxs, path)
ds, idxs = DatasetFormatter().from_similars(learn)
#ImageCleaner(ds, idxs,path,duplicates=True)
#img = open_image('00000087.jpg')

img = open_image(path/'grizzly'/'00000087.jpg')

img

#pred_class,pred_idx,outputs = learn.predict(img)

#pred_class
data.classes
classes = ['black', 'grizzly', 'teddys']
tfms = get_transforms()
data2 = ImageDataBunch.single_from_classes(path, classes, tfms, size=224)

data2.normalize(imagenet_stats)



learn = cnn_learner(data2, models.resnet50).load('stage-2')
pred_class,pred_idx,outputs = learn.predict(img)

pred_class
learn.export()
from flask import Flask

from flask import request

import pickle

import flask
#app = flask.Flask(__name__)



#getting our trained model from a file we created earlier

#model = pickle.load(open("../Teddys-lesson2.py","r"))



#@app.route("/classify-url", methods=["GET"])

#async def classify_url(request):

    #bytes = await get_bytes(request.query_params["url"])

    #img = open_image(BytesIO(bytes))

    #_,_,losses = learner.predict(img)

    #return JSONResponse({

        #"predictions": sorted(

            #zip(cat_learner.data.classes, map(float, losses)),

            #key=lambda p: p[1],

            #reverse=True

        #)

    #})
import pickle

import flask



#app = flask.Flask(__name__)



#getting our trained model from a file we created earlier

#model = pickle.load(open("Teddys-lesson2.p","rb"))



#@app.route('/predict', methods=['POST'])

#def predict():

    #grabbing a set of wine features from the request's body

    #feature_array = request.get_json()['feature_array']

    

    #our model rates the wine based on the input array

   # prediction = model.predict([feature_array]).tolist()

    

    #preparing a response object and storing the model's predictions

   # response = {}

   # response['predictions'] = prediction

    

    #sending our response object back as json

  #  return flask.jsonify(response)





shutil.rmtree("../models",ignore_errors=True)
final_model_directory = os.getcwd()+ "/../models"

#final_model_name='Teddys-lesson2.pkl'

final_model_name='model.pkl'


learn.export(final_model_directory+f"/{final_model_name}")
!pwd