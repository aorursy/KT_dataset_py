# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from pathlib import Path

from fastai.vision import *

from fastai.metrics import error_rate
def dataPath(path, pathToDataset):

     path = Path(path)

     datasetPath = path/pathToDataset

     return path, datasetPath, datasetPath.ls()
path = '../input'

path, datasetPath, subdirectories = dataPath(path, 'waffles-or-icecream')

(path, datasetPath, subdirectories)
def listifyFileNames(subdirectoryPath):

    

    fileNames = [fileName for i in range(len(subdirectoryPath)) for fileName in subdirectoryPath[i].ls()]

    validExtensions = ['.jpg', '.jpeg', '.png', '.JPG', '.jpeg', '.PNG']

    validFileNames = list(filter(lambda fileName: fileName.suffix in validExtensions, fileNames))

    

    return validFileNames
def listifyLabels(fileNamesList):

    return ['ice-cream' if '/ice_cream/' in str(fileName) else 'waffles' for fileName in fileNamesList]
def createDataBunch(path, filePathList, labelFunc, percentOfDataToSplit, imageSize):

     return ImageDataBunch.from_lists(path,

                                        filePathList,

                                        labels = labelFunc,

                                        ds_tfms=get_transforms(),

                                        valid_pct=percentOfDataToSplit,

                                        size=imageSize).normalize(imagenet_stats)
fileNamesList = listifyFileNames(subdirectories)

data = createDataBunch(datasetPath, fileNamesList, listifyLabels(fileNamesList),0.2, 224)

data.classes
#TODO: need to modify the above fucntion as it is returning only icecreams

data.show_batch(rows=3, figsize=(7,6))
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
preTrainedModelPath = 'pretrained-model-for-classifying-types-of-trash'

preTrainedModelPath
learn.path = path

learn.model_dir = preTrainedModelPath

learn.model_dir
learn.load('final')
learn.fit_one_cycle(4)
image = learn.data.valid_ds[3][0]

image
learn.predict(image)
def saveModel(learnerObject, model_dir, modelName= None, export: bool= True,return_path: bool= True):

    learnerObject.model_dir = Path(model_dir)

    

    if export is True:

        learnerObject.path = Path(model_dir)

        learnerObject.export()

    else:

        learnerObject.save(modelName, return_path=return_path)
saveModel(learn, "/kaggle/working", 'stage-1', False, False)
def plotTopLossesAndConfusionMatrix(learnerObject, data, numOfRows, figureSize: tuple, confMatrixSize: tuple, dpi, plotConfusionMatrix: bool = True):

    interp = ClassificationInterpretation.from_learner(learnerObject)

    losses,idxs = interp.top_losses()

    len(data.valid_ds)==len(losses)==len(idxs)

    interp.plot_top_losses(numOfRows, figsize=figureSize)

    

    if plotConfusionMatrix:

        interp.plot_confusion_matrix(figsize=confMatrixSize, dpi=dpi)

        

    interp.most_confused(min_val=2)
plotTopLossesAndConfusionMatrix(learn, data, 9, (15,11), (11,11), 60)
learn.unfreeze()

learn.fit_one_cycle(1)
learn.load('stage-1')

learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-5,1e-3))
plotTopLossesAndConfusionMatrix(learn, data, 9, (15,11), (11,11), 60)
saveModel(learn, "/kaggle/working", False)