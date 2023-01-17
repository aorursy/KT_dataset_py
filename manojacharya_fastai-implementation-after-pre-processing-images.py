# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

#for dirname, _, filenames in os.walk('/kaggle/input'):
#    i=0
    #print(filenames)
    #print(os.path.join(dirname, filenames))
    #for filename in filenames:
    #    if(i>10):
    #        break
    #    print(os.path.join(dirname, filename))
    #    print(filename)
    #    i+=1

# Any results you write to the current directory are saved as output.
bs = 250
# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart
def readJSONFile(path):
    import json
    with open(path) as f:
        data = json.load(f)
    return data
from fastai.vision import *
from fastai.metrics import error_rate
dataPath=Path('/kaggle/input/iwildcam-2020-fgvc7')
#dataPath=Path('c:/Users/manoj/PycharmProjects/data/iwildcam-2020/')
dataPath.ls()
jsonFilePath=dataPath/'iwildcam2020_train_annotations.json'
data = readJSONFile(jsonFilePath)

annotations = data["annotations"]
images=data["images"]
categories = data["categories"]
info = data["info"]

# Convert to Data frame

annotations = pd.DataFrame.from_dict(annotations)
images = pd.DataFrame.from_dict(images)
categories = pd.DataFrame.from_dict(categories)


#Remove data from memory
del data

#Create column image_id to use for merging the two data frames
images["image_id"]  = images["id"]

# Merge annotations and images on image_id

trainDf1 = (pd.merge(annotations, images, on='image_id'))
#Remove Unnecessary fields
trainDf1.drop(["id_y","id_x"], axis = 1, inplace=True)

#print(trainDf1.columns)

trainDf1 = pd.merge(trainDf1, categories.rename(columns={"id":"category_id"}), on="category_id" )
# Unset annotations and images dataframe as they are no longer needed
del annotations
del images

categories[categories["id"]==115 ]
categories
trainDf1[["name","category_id"]]
df=trainDf1[["file_name","category_id"]]
df=df.rename(columns={"file_name":"name","category_id":"label"})

#df=trainDf1[["file_name","name"]]
#df=df.rename(columns={"file_name":"name","name":"label"})
df
minSamples=1000
duplicateDf=pd.DataFrame()

for label in df["label"].unique():
    length=0
    
    x=min
    y=len(df[df["label"]==label])
    multiplier=1
    if(y<minSamples):
        multiplier=int(minSamples/y)
        y=y*multiplier
    #length=y
    #print("{} {} {}".format(y, multiplier, length))
    duplicateDf=duplicateDf.append([df[df["label"]==label]]*multiplier,ignore_index=True)

df=duplicateDf
df
tfms = get_transforms(do_flip=False)

filePath=str(dataPath/"train")

import os
print(os.getcwd())
print(filePath)
df
## Just to make sure that the Image data bunch selected is proper
'''
i=0
while 1:
    try:
        np.random.seed(i)
        data=ImageDataBunch.from_df(filePath, df, ds_tfms=tfms, size=224, bs=100)
    except:
        i+=1
        if(i%100==0):
            print(str(i) + " did not work")
        continue
    else: 
        print('Seed '+str(i)+' works')
        break
    break
'''    

np.random.seed(25)
#data = ImageDataBunch.from_df("/home/manoj/Documents/data/data/iwildcam-2020/train/28X28",df, 
data = ImageDataBunch.from_df(filePath
                              , pd.DataFrame(df)
                              , ds_tfms=tfms
                              , size=200
                              , valid_pct=.2
                              , bs=bs)
categories[categories["id"].isin([257, 229, 420, 306, 296, 402, 408, 420, 412])]
data.show_batch(rows=3, figsize=(15,15))
print(data.classes)
len(data.classes),data.c
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
#learn = cnn_learner(data, models.resnet50, metrics=error_rate)
learn.model
learn.fit_one_cycle(1)
learn.recorder.plot_losses()
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
learn.unfreeze()
learn.fit_one_cycle(10)
#jsonTestFilePath='/home/manoj/Documents/data/data/iwildcam-2020/iwildcam2020_test_information.json'
jsonTestFilePath=dataPath/'iwildcam2020_test_information.json'
testData = readJSONFile(jsonTestFilePath)

testImages=testData["images"]
testCategories = testData["categories"]
testInfo = testData["info"]

# Convert to Data frame

testImages = pd.DataFrame.from_dict(testImages)
testCategories = pd.DataFrame.from_dict(testCategories)

#Remove data from memory
del testData, testInfo

# Remove Unnecessary fields from images
testDf1 = pd.DataFrame(testImages.file_name)
testImages
#testPath=Path("/home/manoj/Documents/data/data/iwildcam-2020/test/100X100")
testPath=dataPath/'test'
#df=[ {"file_name":str(file).replace(str(testPath)+'/',''), "name": learn.predict(open_image(file))[0] }
df=[ {"file_name":str(file).replace(str(testPath)+'/',''), "Id": learn.predict(open_image(file))[0] }
    for file in testPath.ls()[:]
]
df=pd.DataFrame(df)
df["file_name"]=list(map(lambda x: os.path.basename(x), df["file_name"]))

jsonSubmissionFilePath=dataPath/'sample_submission.csv'
submission=pd.read_csv(jsonSubmissionFilePath)
submission.drop(columns=["Category"], inplace=True)
submission
testXref=testImages[["file_name","id"]]
len(testXref)
df.merge(testXref, on='file_name')
#df1=df.merge(testImages, on='file_name')[["id","name"]]
df1=df.merge(testXref, on='file_name')[["id","Id"]]
#df1=df1.rename(columns={"id":"Id"})
df1=df1.rename(columns={"Id":"Category", "id":"Id"})
df1
#df2=submission.merge(df1, on="Id")[["Id","Category"]]
df2=submission.merge(df1, on="Id")
df2
df2.to_csv("submission.2020040217.csv", index=False)

