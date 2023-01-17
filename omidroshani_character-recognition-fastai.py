!pip install fastai==1.0.61
from fastai import *

from fastai.vision import *

bs = 64
data = ImageDataBunch.from_folder("../input/characterrecognitionfromnumberplate/Training Data", ds_tfms=get_transforms(), size=224, valid_pct = 0.20)

data.classes
learn = cnn_learner(data, models.resnet50,pretrained=True, metrics=accuracy)
learn.fit_one_cycle(10)
learn.export( file = '/kaggle/working/export.pkl' )
learn2 = load_learner( '/kaggle/working' )
il = ImageList.from_folder("../input/characterrecognitionfromnumberplate/Testing Data/0")

import json



with open('/kaggle/working/labels.json', 'w') as outfile:

    labels = [ lbl.replace("class_","") for lbl in data.classes ]

    json.dump(labels, outfile)
str(learn2.predict(il[0])[0]).replace("class_","")