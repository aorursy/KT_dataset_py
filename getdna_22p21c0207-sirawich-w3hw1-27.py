import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import display # print stuff beautiflly

import matplotlib.pyplot as plt # plot and show images

import time # Pause

import os # Files

import cv2; from keras.applications.imagenet_utils import preprocess_input; # Image Processing

from tqdm import tqdm # progress bar



import gc # Clear Memory



Seed = 0 # Fixed randomization



import warnings

warnings.filterwarnings("ignore") # Ignore All Warnings
def ClearMemory():

    print(f"[ClearMemory] {gc.collect()} objects cleared")

ClearMemory()
Train = pd.read_csv("../input/super-ai-image-classification/train/train/train.csv",index_col="id")

Train.columns = ["Class"]

Train.index = [f"../input/super-ai-image-classification/train/train/images/{x}" for x in Train.index]

display(Train)
DirectoryName, _, Filenames = list(os.walk('../input/super-ai-image-classification/val/val/images'))[0]

Test = pd.DataFrame(np.zeros(len(Filenames),dtype=int),index=[os.path.join(DirectoryName,x) for x in Filenames], columns=["Class"])

del(DirectoryName); del(_); del(Filenames)

display(Test)
from keras.applications.imagenet_utils import preprocess_input

from keras.preprocessing import image

from PIL import Image, ImageOps

from sklearn.preprocessing import normalize
from tensorflow.keras.applications.vgg16 import VGG16 as FeatureExtractor

from tensorflow.keras import Model



FeatModel = FeatureExtractor(include_top=False, weights='imagenet', classes=1000) # Define Model for Extracting Feature



def ExtractFeatureFromModel(imgpath):

    img = image.load_img(imgpath)

    img = ImageOps.fit(img, (224*2, 224*2), Image.ANTIALIAS)

    

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    

    features = FeatModel.predict(x, batch_size=1,verbose=0)

    features = np.ndarray.flatten(features).astype('float64')

    

    return normalize([features])[0]
DefaultSize = (400, 400);

def ExtractImageFeatures(ImagePaths):

    ImageFeatures = []; # You can use list comprehension, but I want the progress bar also

    for ImagePath in tqdm(ImagePaths):

        ImageFeatures.append(ExtractFeatureFromModel(ImagePath))

    time.sleep(0.5)

    ClearMemory()

    time.sleep(0.5)

    return ImageFeatures
Train["Feature"] = ExtractImageFeatures(Train.index)

Test["Feature"] = ExtractImageFeatures(Test.index)
Data0 = Train[Train.Class==0].sample(frac=1,random_state=Seed) # Shuffle Class 0 Data

Data1 = Train[Train.Class==1].sample(frac=1,random_state=Seed) # Shuffle Class 1 Data



Class0Count = len(Data0)

Class1Count = len(Data1)



del(Train)



DataX0 = np.array(Data0.Feature.tolist())

DataX1 = np.array(Data1.Feature.tolist())

DataY0 = Data0.Class # ToCat(Data0.Class,num_classes = 2)

DataY1 = Data1.Class # ToCat(Data1.Class,num_classes = 2)



del(Data0); del(Data1);
print(f"""There are {Class0Count} Class 0 Pictures.

There are {Class1Count} Class 1 Pictures.""")
TrainSize = 0.8



TrainX = (np.concatenate((DataX0[:round(len(DataX0)*TrainSize)], DataX1[:round(len(DataX1)*TrainSize)])))

ValX   = (np.concatenate((DataX0[round(len(DataX0)*TrainSize):], DataX1[round(len(DataX1)*TrainSize):])))

TrainY = (np.concatenate((DataY0[:round(len(DataY0)*TrainSize)], DataY1[:round(len(DataY1)*TrainSize)])))

ValY   = (np.concatenate((DataY0[round(len(DataY0)*TrainSize):], DataY1[round(len(DataY1)*TrainSize):])))



TrainY = TrainY.astype(float)

ValY = ValY.astype(float)
ClearMemory()
print(f"""Shape

TrainX      = {TrainX.shape}

TrainY      = {TrainY.shape}

ValX        = {ValX.shape}

ValY        = {ValY.shape}



Class

Train      0: {len(TrainY)-sum(TrainY)}

Train      1: {sum(TrainY)}

Validation 0: {len(ValY)-sum(ValY)}

Validation 1: {sum(ValY)}

""");
ClearMemory()
from sklearn.svm import LinearSVC as PredictionModel

from sklearn.calibration import CalibratedClassifierCV



Model = PredictionModel(random_state=Seed)

Model = CalibratedClassifierCV(Model) 



Model.fit(np.vstack(TrainX),TrainY)



Pred = Model.predict(np.vstack(ValX))



Acc = sum(Pred == ValY)/len(Pred)



Conf = Model.predict_proba(np.vstack(ValX))

display(Conf)
ClearMemory()
ValPred = Model.predict(np.vstack(ValX)).astype(int)
from sklearn.metrics import f1_score

from sklearn.metrics import classification_report



Score = np.mean(f1_score(ValY, ValPred, average=None));



print(f"""F1 Score = {Score:.4f}

Prediction Class 1: {sum(ValPred)}

Prediction Class 0: {len(ValPred)-sum(ValPred)}



There are total of {Class1Count} Class 1 Pictures.

There are total of {Class0Count} Class 0 Pictures.



Classification Report:

{classification_report(ValY, Pred)}""")
TestPred = Model.predict(np.vstack(np.array(Test.Feature.tolist())))

Test.Class = TestPred.astype(int);
Summission = Test[["Class"]].reset_index().rename(columns={"index":"id","Class":"category"})

Summission.id = [x.split("/")[-1] for x in Summission.id]

Summission.to_csv("Submission.csv", index=False)