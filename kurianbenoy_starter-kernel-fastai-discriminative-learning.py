from fastai.vision import *

from fastai.metrics import error_rate
path = "../input"

SEED = 42

sz = 124

bs = 64

test_df = pd.read_csv(f"{path}/sample_submission.csv")

sub_df = pd.read_csv(f"{path}/sample_submission.csv")
data = ImageDataBunch.from_folder(path, valid_pct=0.1, ds_tfms=get_transforms(), size=sz, bs=bs, seed=SEED).normalize(imagenet_stats)

data.add_test(ImageList.from_df(test_df, path, folder="test/test"))
#Showing Images

data.show_batch(rows=3, figsize=(5,5))
learn = cnn_learner(data, models.resnet34, metrics=[accuracy],model_dir="/tmp/model/")
# Plotting to find the ideal learning rate

learn.lr_find()

learn.recorder.plot()
lr=1e-1/2
learn.fit_one_cycle(3, slice(lr))
# Saving the model

learn.save('stage1')
data = ImageDataBunch.from_folder(path, train=".", 

                                  valid_pct=0.2,

                                  ds_tfms=get_transforms(),

                                  size=224,bs=64, 

                                  num_workers=0).normalize(imagenet_stats)
learn.lr_find()

learn.recorder.plot()
lr = 1e-3/3
learn.fit_one_cycle(2, slice(lr))
test_probs, _ = learn.get_preds(ds_type=DatasetType.Test)

test_preds = [data.classes[pred] for pred in np.argmax(test_probs.numpy(), axis=-1)]
sub_df = pd.read_csv(f"{path}/sample_submission.csv")

sub_df.predicted_class = test_preds

sub_df.to_csv("submission.csv", index=False)
sub_df.head()