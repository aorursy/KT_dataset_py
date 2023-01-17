import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
!pip install git+https://github.com/fastai/fastai2 
from fastai2.vision.all import *
# train and test csv
train = pd.read_csv("../input/shopee-vgg16/train_data.csv")
test = pd.read_csv("../input/shopee-product-detection-student/test.csv")
# paths leading to images
train_path = Path("/kaggle/input/shopee-product-detection-student/train/train/train/")
test_path = Path("/kaggle/input/shopee-product-detection-student/test/test/test/")
# add the category to filename for easier usage with fastai API
train['filename'] = train.apply(lambda x: str(x.category).zfill(2) + '/' + x.filename, axis=1)
train
# train in a 10% subset of the data
# to speed up experimentation
# comment these lines out to increase accuracy (but necessitates longer training time)
from sklearn.model_selection import train_test_split
_, train = train_test_split(train, test_size=0.1, stratify=train.category)
item_tfms = [RandomResizedCrop(224, min_scale=0.75)]
batch_tfms = [*aug_transforms(), Normalize.from_stats(*imagenet_stats)]
def get_dls_from_df(df):
    df = df.copy()
    options = {
        "item_tfms": item_tfms,
        "batch_tfms": batch_tfms,
        "bs": 32,
    }
    dls = ImageDataLoaders.from_df(df, train_path, **options)
    return dls
dls = get_dls_from_df(train)
dls.show_batch()
learn = cnn_learner(dls, resnet34, metrics=accuracy)
learn.fine_tune(30)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(15,15), dpi=60)
# get the most confused labels with at least 10 incorrect predictions
interp.most_confused(10)
test_images = test.filename.apply(lambda fn: test_path/fn)
test_dl = dls.test_dl(test_images)
preds = learn.get_preds(dl=test_dl, with_decoded=True)
preds
# save raw predictions
torch.save(preds, "rawpreds")
submission  = test[["filename"]]
submission["category"] = preds[2]
# zero-pad the submissions
submission["category"] = submission.category.apply(lambda c: str(c).zfill(2))
# preview
submission
# save the submissions as CSV
submission.to_csv("submission.csv", index = False)
