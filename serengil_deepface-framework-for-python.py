import os
import numpy as np
import pandas as pd
import zipfile
import cv2
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
TRAIN_ZIP = '../input/recognizing-faces-in-the-wild/train.zip'
TEST_ZIP='../input/recognizing-faces-in-the-wild/test.zip'
print("unzipping train set")
with zipfile.ZipFile(TRAIN_ZIP, 'r') as zip_ref:
    zip_ref.extractall("../output/kaggle/working/train")

print("unzipping test set")
with zipfile.ZipFile(TEST_ZIP, 'r') as zip_ref:
    zip_ref.extractall("../output/kaggle/working/test")
df = pd.read_csv("../input/recognizing-faces-in-the-wild/train_relationships.csv")
df.head()
def findCustomImages(path):
    images = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            images.append(os.path.join(dirname, filename))
    
    return images
root = "../output/kaggle/working/train/"

samples = []
for index, instance in df.iterrows():
    person1 = root+instance.p1
    person2 = root+instance.p2
    
    person1_images = findCustomImages(person1)
    person2_images = findCustomImages(person2)
    
    for i in person1_images:
        for j in person2_images:
            sample = []
            sample.append(i)
            sample.append(j)
            samples.append(sample)
df = pd.DataFrame(samples, columns = ["p1", "p2"])
df.head()
print("There are ",df.shape[0]," image pairs existing as a relative")
!pip install gdown==3.10.1 --no-deps
!pip install deepface==0.0.22 --no-deps
from deepface import DeepFace
trainset_df = df.sample(5).reset_index(drop = True)
trainset = trainset_df.values.tolist()
models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace"]
metrics = ["cosine", "euclidean", "euclidean_l2"]
for model in models:
    for metric in metrics:
        
        #print("Analyzing ", model, " and ", metric)
        resp_obj = DeepFace.verify(trainset, model_name = model, distance_metric = metric, enforce_detection = False)
        
        distances = []
        for key, value in resp_obj.items():
            distances.append(value['distance'])
        
        trainset_df['%s_%s' % (model, metric)] = distances
trainset_df.head()
#trainset_df.DeepFace_cosine.plot.kde()

for model in models:
    for metric in metrics:
        print("Distribution for ",model," and ", metric," pair")
        trainset_df['%s_%s' % (model, metric)].plot.kde()
        plt.show()
        print("-----------------------------------------")
sets = ["p1", "p2"]

for item in sets:
    resp_obj = DeepFace.analyze(trainset_df[item].values.tolist(), enforce_detection = False)

    attributes = []
    for key, value in resp_obj.items():
        attribute = []
        attribute.append(value["dominant_emotion"])
        attribute.append(value["age"])
        attribute.append(value["gender"])
        attribute.append(value["dominant_race"])
        attributes.append(attribute)

    attributes = pd.DataFrame(attributes, columns = ["%s_emotion" % (item), "%s_age" % (item), "%s_gender" % (item), "%s_race" % (item)])

    trainset_df = pd.concat([trainset_df, attributes], axis=1)
trainset_df