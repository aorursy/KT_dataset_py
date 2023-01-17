InitialVariable = dir()
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import display # print stuff beautiflly

import matplotlib.pyplot as plt # plot and show images

import time # Pause

import os # Files

import cv2; from keras.applications.imagenet_utils import preprocess_input; # Image Processing

from tqdm import tqdm # progress bar



Seed = 0 # Fixed randomization



import warnings

warnings.filterwarnings("ignore") # Ignore All Warnings
import gc # Clear Memory

def ClearMemory(showlog=True):

    if showlog:

        print(f"[ClearMemory] {gc.collect()} objects cleared")

    else:

        gc.collect();

InitialVariable.append("gc")

InitialVariable.append("ClearMemory")



ClearMemory()
Train = pd.read_csv("../input/thai-mnist-classification/mnist.train.map.csv",index_col="id")

Train.columns = ["Class"]

Train.index = [f"../input/thai-mnist-classification/train/{x}" for x in Train.index]

display(Train)
DirectoryName, _, Filenames = list(os.walk('../input/thai-mnist-classification/test'))[0]

Test = pd.DataFrame(np.zeros(len(Filenames),dtype=int),index=[os.path.join(DirectoryName,x) for x in Filenames], columns=["Number"])

del(DirectoryName); del(_); del(Filenames)

display(Test)
from keras.applications.imagenet_utils import preprocess_input

from keras.preprocessing import image

from PIL import Image, ImageOps

from sklearn.preprocessing import normalize
from skimage.morphology import convex_hull_image

def invert(imagem):

    return 255 - imagem

def convex_resize(img):

    img = invert(img)

    img = convex_crop(img,pad=20)

    img = cv2.resize(img,(32,32))

    return img

def convex_crop(img,pad=20):

    convex = convex_hull_image(img)

    r,c = np.where(convex)

    while (min(r)-pad < 0) or (max(r)+pad > img.shape[0]) or (min(c)-pad < 0) or (max(c)+pad > img.shape[1]):

        pad = pad - 1

    return img[min(r)-pad:max(r)+pad,min(c)-pad:max(c)+pad]



def thes_resize(img,thes=40):

    img = invert(img)

    img = convex_crop(img,pad=20)

    img = cv2.resize(img,(224,224))

    return img



from tensorflow.keras.applications.vgg16 import VGG16 as FeatureExtractor

from tensorflow.keras import Model



FeatModel = FeatureExtractor(include_top=False, weights='imagenet', classes=1000) # Define Model for Extracting Feature



def ExtractFeatureFromModel(imgpath):

    

    img = cv2.cvtColor(thes_resize(cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE)),cv2.COLOR_GRAY2RGB)

    

    

    x = np.expand_dims(img, axis=0)

    x = preprocess_input(x)

    

    features = FeatModel.predict(x, batch_size=1,verbose=0)

    features = np.ndarray.flatten(features).astype(float)

    

    return normalize([features])[0]
def ExtractImageFeatures(ImagePaths):

    ImageFeatures = []; # You can use list comprehension, but I want the progress bar also

    for ImagePath in tqdm(ImagePaths):

        ImageFeatures.append(ExtractFeatureFromModel(ImagePath))

    time.sleep(0.5);

    ClearMemory(showlog=True)

    time.sleep(0.5)

    return ImageFeatures
Test["Feature"] = ExtractImageFeatures(Test.index)

Test.to_pickle("./TestSet.pkl")

del(Test)



Train["Feature"] = ExtractImageFeatures(Train.index)
Datas = [Train[Train.Class==i].sample(frac=1,random_state=Seed) for i in range(10)] # Shuffle Every Class Data and save as array



ClassCounts = [len(x) for x in Datas]



del(Train)



DataXs = [np.array(x.Feature.tolist()) for x in Datas]



DataYs = [x.Class for x in Datas]



while len(Datas) > 0: del(Datas[0])
for (i, x) in enumerate(ClassCounts):

    print(f"Class {i}: {x} Pictures.")
TrainSize = 0.8



TrainX = np.concatenate([x[:round(len(x)*TrainSize)] for x in DataXs])

ValX   = np.concatenate([x[round(len(x)*TrainSize):] for x in DataXs])

TrainY = np.concatenate([y[:round(len(y)*TrainSize)] for y in DataYs])

ValY   = np.concatenate([y[round(len(y)*TrainSize):] for y in DataYs])



TrainY = TrainY.astype(float)

ValY = ValY.astype(float)
ClearMemory()
print(f"""Shape

TrainX = {TrainX.shape}

TrainY = {TrainY.shape}

ValX   = {ValX.shape}

ValY   = {ValY.shape}""");
ClearMemory()
from xgboost import XGBClassifier as PredictionModel

# from sklearn.svm import LinearSVC as PredictionModel



Model = PredictionModel(random_state=Seed)



Model.fit(np.vstack(TrainX),TrainY)
ClearMemory()
if (TrainSize != 1): ValPred = Model.predict(np.vstack(ValX)).astype(int)
if (TrainSize != 1):

    from sklearn.metrics import accuracy_score

    from sklearn.metrics import classification_report



    Score = accuracy_score(ValY, ValPred, normalize=False);



    print(f"""Accuracy = {Score}/{len(ValPred)} ({Score/len(ValPred)*100:.2f})



    Classification Report:

    {classification_report(ValY, ValPred)}""")



    del(ValY); del(ValPred); del(ClassCounts);



    ClearMemory()
from joblib import dump

dump(Model,"Model.joblib")
# Reset Environment

import sys

this = sys.modules[__name__];

for n in dir():

    if n not in (InitialVariable + ["this","InitialVariable"]): delattr(this, n)

del(this); del(n);
ClearMemory()
import pandas as pd

import numpy as np



from joblib import load



Test = pd.read_pickle("./TestSet.pkl")



Model = load("Model.joblib")



TestPred = Model.predict(np.vstack(np.array(Test.Feature.tolist())))

Test.Number = TestPred.astype(int);



Test.drop("Feature", axis=1, inplace=True)

Test.index = [x.split("/")[-1] for x in Test.index]



Test.to_pickle("./TestSet.pkl")
# Reset Environment

import sys

this = sys.modules[__name__];

for n in dir():

    if n not in (InitialVariable + ["this","InitialVariable"]): delattr(this, n)

del(this); del(n); del(InitialVariable);
ClearMemory()
# Reset Environment

import sys

this = sys.modules[__name__];

for n in dir():

    if n not in (InitialVariable + ["this","InitialVariable"]): delattr(this, n)

del(this); del(n); del(InitialVariable);
ClearMemory()
import pandas as pd

import numpy as np



from joblib import load



Test = pd.read_pickle("./TestSet.pkl").Number.append(pd.Series([np.nan], index=[np.nan]))
def GetValue(F1, F2, F3):

    Answer = 0

    if pd.isna(F1):

        Answer = F2 + F3

    elif F1 == 0:

        Answer = F2 * F3

    elif F1 == 1:

        Answer = abs(F2 - F3)

    elif F1 == 2:

        Answer = (F2 + F3) * abs(F2 - F3)

    elif F1 == 3:

        Answer = abs((F3 * (F3 +1) - F2 * (F2-1)) / 2)

    elif F1 == 4:

        Answer = 50 + (F2 - F3)

    elif F1 == 5:

        Answer = min(F2, F3)

    elif F1 == 6:

        Answer = max(F2, F3)

    elif F1 == 7:

        Answer = ((F2 * F3) % 9 ) * 11

    elif F1 == 8:

        Answer = (((F2 ** 2) + 1) * F2) + (F3 * (F3 + 1))

        while Answer > 99:

            Answer = Answer - 99

    elif F1 == 9:

        Answer = 50 + F2

    else:

        print(F1)

    return Answer
TestRules = pd.read_csv("../input/thai-mnist-classification/test.rules.csv",index_col="id")
TestRules.feature1 = [Test[x] for x in TestRules.feature1]

TestRules.feature2 = [Test[x] for x in TestRules.feature2]

TestRules.feature3 = [Test[x] for x in TestRules.feature3]
TestRules.predict = [GetValue(F1, F2, F3) for F1, F2, F3 in TestRules[["feature1", "feature2", "feature3"]].iloc]
TestRules
ClearMemory()
Summission = TestRules[["predict"]].reset_index().rename(columns={"index":"id"})

Summission.to_csv("Submission.csv", index=False)