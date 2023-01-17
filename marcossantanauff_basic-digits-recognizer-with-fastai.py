# the following three lines are suggested by the fast.ai course

%reload_ext autoreload

%autoreload 2

%matplotlib inline
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from fastai.vision import *

from PIL import Image
# Train set

train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
print(train.shape)

print(test.shape)
train.head(5)
test.head(5)
img = [np.reshape(train.iloc[idx,1:].values,(28,28)) for idx in range(5)]
len(img)
for f in img:   

    plt.imshow(f,cmap='gray')

    plt.show()
#TRAIN = Path("../train")

#TEST = Path("../test")

PATH = Path('../')

TRAIN = Path('../train')

TRAIN.mkdir(parents=True, exist_ok=True)

TRAIN





TEST = Path('../test')

TEST.mkdir(parents=True, exist_ok=True)

TEST



PATH.ls()

os.listdir('/kaggle/working/')
train_img = train.iloc[:,1:785]

test_img = test.iloc[:,:]
#Source: [https://www.kaggle.com/christianwallenwein/beginners-guide-to-mnist-with-fast-ai]

def save_img(data,fpath,isTest=False):

    if isTest == False:

        for index, row in data.iterrows():

    

            label,digit = row[0], row[1:]

    

            filepath = fpath

            filename = "train_{}.jpg".format(index)

            digit = digit.values

            digit = digit.reshape(28,28)

            digit = digit.astype(np.uint8)

    

            img = Image.fromarray(digit)

            img.save(filepath/filename)

            

    else:

        for index, row in data.iterrows():

    

            digit = row[:]

    

            filepath = fpath

            filename = "test_{}.jpg".format(index)

            digit = digit.values

            digit = digit.reshape(28,28)

            digit = digit.astype(np.uint8)

    

            img = Image.fromarray(digit)

            img.save(filepath/filename)
save_img(train,TRAIN,False)
sorted(os.listdir(TRAIN))[0]
save_img(test,TEST,True)
sorted(os.listdir(TEST))[0]
train['filename'] = ['train/train_{}.jpg'.format(x) for x,_ in train.iterrows()]
train.head()
test['filename'] = ['test/test_{}.jpg'.format(x) for x,_ in test.iterrows()]
test.head()
# Sanity check

img = open_image(TRAIN/'train_0.jpg')

img
tfms = ([*rand_pad(padding=3,size=28,mode='reflection'),zoom(scale=1.005),],[])
src = ImageList.from_df(train,PATH,cols='filename').split_by_rand_pct(0.2).label_from_df(cols='label').add_test_folder(PATH/'test')
src
data =  src.transform(tfms, size=28).databunch().normalize(imagenet_stats)
data.show_batch(2,2)
data.classes,data.c
data.train_ds[0][0]
learn = cnn_learner(data,models.resnet50,metrics=accuracy,model_dir='/kaggle/working/')
learn.lr_find(num_it=600)
learn.recorder.plot(skip_end=25)
lr=4e-3
learn.fit_one_cycle(10,slice(lr))
learn.recorder.plot_losses()
learn.save('stage1-resnet50-mnist')
learn.load('stage1-resnet50-mnist')
learn.unfreeze()
learn.lr_find(num_it=600)


learn.recorder.plot()
learn.fit_one_cycle(10,slice(1e-5,1e-4/5))
learn.save('stage2-resnet50-mnist')
learn.export()
PATH.ls()
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(9, figsize=(7, 7))

# Maybe feed the network with top losses to improve score?

test = ImageList.from_folder(TEST)

test
learn = load_learner(PATH, test=test)
preds, _ = learn.get_preds(ds_type=DatasetType.Test)
probabilities = preds[0].tolist()

[f"{index}: {probabilities[index]}" for index in range(len(probabilities))]
class_score = np.argmax(preds, axis=1)
class_score[0].item()
sample_submission =  pd.read_csv("../input/digit-recognizer/sample_submission.csv")

sample_submission.head()



# remove file extension from filename

ImageId = [os.path.splitext(p)[0].split('test_')[1] for p in os.listdir(TEST)]

# typecast to int so that file can be sorted by ImageId

ImageId = [int(path) for path in ImageId]

# +1 because index starts at 1 in the submission file

ImageId = [ID+1 for ID in ImageId]
sorted(ImageId)[-1]
submission  = pd.DataFrame({

    "ImageId": ImageId,

    "Label": class_score

})

# submission.sort_values(by=["ImageId"], inplace = True)

submission.to_csv("nona-submission.csv", index=False)