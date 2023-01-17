%reload_ext autoreload

%autoreload 2

%matplotlib inline
import numpy as np

import pandas as pd

from fastai.vision import *
path = Path("../input/")

path.ls()
# look at the image-label file

df = pd.read_csv(path/"train_v2.csv")

df.head()
# image transformation

tfms = get_transforms(flip_vert = True, max_lighting = 0.1, max_zoom = 1.05, max_warp = 0.)
# create ImageList

np.random.seed(7)

source = (ImageList.from_csv(path, "train_v2.csv", folder = "train-jpg", suffix = ".jpg")

         .split_by_rand_pct(0.2)

         .label_from_df(cols = "tags", label_delim = " "))

# label_delim to separate the words in "tags" column so as to generate multiple labels 
# data with size 128 (default 256)

data = (source.transform(tfms, size = 128)

       .databunch()

       .normalize(imagenet_stats))
# show the data

data.show_batch(rows = 4, figsize = (15,15))
# metrics

acc_thresh = partial(accuracy_thresh, thresh = 0.2) # choose threshold = 0.2

f2_score = partial(fbeta, beta = 2, thresh = 0.2) # fbeta where beta = 2 (F2) and threshold = 0.2
# download model

learn = cnn_learner(data, models.resnet50, metrics = [acc_thresh, f2_score], model_dir = "/tmp/models")
# find good learning rate

learn.lr_find()

learn.recorder.plot()
# baseline model with image size=128

learn.fit_one_cycle(cyc_len = 5, max_lr = slice(1e-2))
# save this model

learn.save("baseline-rn50-128")
# create 2nd data with original size

np.random.seed(7)

data2 = (source.transform(tfms, size = 256)

        .databunch()

        .normalize(imagenet_stats))
# create another CNN model

learn2 = cnn_learner(data2, models.resnet50, metrics = [acc_thresh, f2_score], model_dir = "/tmp/models")
# plot the learning rate of this model

learn2.lr_find()

learn2.recorder.plot()
# baseline model with image size = 256

lr = 3e-2

learn2.fit_one_cycle(cyc_len = 5, max_lr = slice(lr))
# save baseline model for size=256

learn2.save("baseline-rn50-256")
# plot the learning rate

learn2.unfreeze()

learn2.lr_find()

learn2.recorder.plot()
# model 2 for image size = 256

learn2.fit_one_cycle(cyc_len = 10, max_lr = slice(1e-5, lr/10))
# plot the training and validation loss of the model

learn2.recorder.plot_losses()
# save the unfreezed model

learn2.save("stage-2-rn50-256")
# export the model

learn2.export(file = "../working/export.pkl")
#test = ImageList.from_folder(path/"test-jpg").add(ImageList.from_folder(path/"test-jpg-additional"))

test = ImageList.from_folder(path/"test-jpg-v2")

len(test)
load_path = Path("../working/")



learn = load_learner(load_path, test=test)

predicts, _ = learn.get_preds(ds_type = DatasetType.Test)
# pick the labels as long as the probability is more than 0.2

labels_pred = [" ".join([learn.data.classes[i] for i,p in enumerate(pred) if p > 0.2]) for pred in predicts]



labels_pred[:5]
for img in learn.data.test_ds.items[:10]:

    print(img.name)
# pick up the images' names

image_names = [img.name[:-4] for img in learn.data.test_ds.items] # img.name[:-4] because I want to remove '.jpg' from the name
# create the dataframe of images' names and their tags (the format we have seen in train_v2.csv)

df2 = pd.DataFrame({"image_name":image_names, "tags":labels_pred})

df2.head()
# create the csv file for submission

df2.to_csv("submission.csv", index = False)