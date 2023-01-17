path = "../input/scene_classification/scene_classification/train/"
from fastai import *

from fastai.vision import *
bs = 256
df = pd.read_csv('../input/scene_classification/scene_classification/train.csv')

df.head()
tfms = get_transforms(flip_vert=False,max_zoom=1.0,max_warp=0)
data_small = (ImageList.from_csv(path, csv_name = '../train.csv') 

        .split_by_rand_pct()              

        .label_from_df()            

        .add_test_folder(test_folder = '../test')              

        .transform(tfms, size=128)

        .databunch(num_workers=0))
data_large = (ImageList.from_csv(path, csv_name = '../train.csv') 

        .split_by_rand_pct()              

        .label_from_df()            

        .add_test_folder(test_folder = '../test')              

        .transform(tfms, size=256)

        .databunch(num_workers=0))
data_small.show_batch(rows=3, figsize=(8,10))
data_large.show_batch(rows=3, figsize=(8,10))
print(data_small.classes)

print(data_large.classes)
learn_34 = cnn_learner(data_small, models.resnet34, metrics=[error_rate, accuracy], model_dir="/tmp/model/")

learn_50 = cnn_learner(data_small, models.resnet50, metrics=[error_rate, accuracy], model_dir="/tmp/model/")

learn_101 = cnn_learner(data_small, models.resnet101, metrics=[error_rate, accuracy], model_dir="/tmp/model/")
learn_34.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn_34)



losses,idxs = interp.top_losses()



len(data_small.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(7,7), dpi=60)
learn_34.save('/kaggle/working/resnet34-size128-stage1')
learn_34.data = data_large
learn_34.unfreeze()
learn_34.lr_find()
learn_34.recorder.plot()
learn_34.fit_one_cycle(3, max_lr=slice(1e-6, 1e-4))
learn_34.save('/kaggle/working/resnet34-size256-stage1')
learn_50.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn_50)



losses,idxs = interp.top_losses()



len(data_small.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(7,7), dpi=60)
learn_50.save('/kaggle/working/resnet50-size128-stage1')
learn_50.data = data_large
learn_50.unfreeze()
learn_50.lr_find()
learn_50.recorder.plot()
learn_50.fit_one_cycle(3, max_lr=slice(1e-6, 1e-4))
learn_50.save('/kaggle/working/resnet50-size256-stage1')
learn_101.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn_101)



losses,idxs = interp.top_losses()



len(data_small.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(7,7), dpi=60)
learn_101.save('/kaggle/working/resnet101-size128-stage1')
learn_101.data = data_large
learn_101.unfreeze()
learn_101.lr_find()
learn_101.recorder.plot()
learn_101.fit_one_cycle(3, max_lr=slice(1e-6, 1e-4))
learn_101.save('/kaggle/working/resnet101-size256-stage1')