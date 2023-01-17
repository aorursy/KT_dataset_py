%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *
bs = 64

# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart
#help(URLs)

#help(untar_data)
path = untar_data(URLs.PLANET_SAMPLE); path
#path.ls()
path_anno = path

path_img = path/'train'
#import csv

#with open(path/'labels.csv') as csvDataFile:

#    csvReader = csv.reader(csvDataFile)

#    for row in csvReader:

#        print(row)
#help(csv)
fileName = []

labels = []



with open(path/'labels.csv') as csvDataFile:

    csvReader = csv.reader(csvDataFile)

    for row in csvReader:

        fileName.append(row[0])

        labels.append(row[1])

#print(fileName)

#print(labels)
clearFileNames = []

with open(path/'labels.csv') as csvDataFile:

    csvReader = csv.reader(csvDataFile)

    for row in csvReader:

        if 'clear' in row[1]:

            clearFileNames.append(row[0])

#print(clearFileNames)
nonClearFileNames = []

with open(path/'labels.csv') as csvDataFile:

    csvReader = csv.reader(csvDataFile)

    for row in csvReader:

        if 'clear' not in row[1] and row[0] != 'image_name':

            nonClearFileNames.append(row[0])

#print(nonClearFileNames)

nonClearFileNames[:5]
#doc(DataFrame)

clearFileNames = []

nonClearFileNames = []

with open(path/'labels.csv') as csvDataFile:

    csvReader = csv.reader(csvDataFile)

    for row in csvReader:

        if 'clear' in row[1] and row[0] != 'image_name':

            clearFileNames.append(row[0])

        if 'clear' not in row[1] and row[0] != 'image_name':

            nonClearFileNames.append(row[0])

clearDF = pd.DataFrame({'name':clearFileNames,'label':'clear'})

nonClearDF = pd.DataFrame({'name':nonClearFileNames,'label':'non clear'})

labelledData = pd.concat([clearDF, nonClearDF], ignore_index=True)
#doc(ImageDataBunch.from_df)

tfms = get_transforms()

data = ImageDataBunch.from_df(path/'train',labelledData, size=224, suffix='.jpg', ds_tfms=tfms, bs=bs).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(9,9))
learn = create_cnn(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.fit_one_cycle(4)
learn.fit_one_cycle(4)
learn.save('Stage 1')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(9,9),heatmap=True)
interp.plot_top_losses(9, figsize=(9,9),heatmap=False)
#doc(interp.plot_top_losses)
learn.unfreeze()
learn.fit_one_cycle(1)
learn.load('Stage 1');
learn.lr_find()
learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(4, max_lr=slice(1e-6,1e-3))
interp.plot_top_losses(9, figsize=(9,9),heatmap=True)
learn.save('Stage_2')
tfms = get_transforms()

data = ImageDataBunch.from_df(path/'train',labelledData, size=299, suffix='.jpg', ds_tfms=tfms, bs=bs//2, num_workers=0).normalize(imagenet_stats)
learn = create_cnn(data, models.resnet50, metrics=error_rate)
learn.fit_one_cycle(8)
learn.lr_find()

learn.recorder.plot()
learn.save('stage-1-50')
learn.unfreeze()

learn.fit_one_cycle(8, max_lr=slice(5e-6,5e-2))
learn.save('stage-1-50-v2')

learn.load('stage-1-50')
interp = ClassificationInterpretation.from_learner(learn)
interp.most_confused(min_val=2)

interp.plot_top_losses(9, figsize=(9,9),heatmap=True)