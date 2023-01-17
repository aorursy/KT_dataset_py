path = "../input/scene_classification/scene_classification/train/"
from fastai import *

from fastai.vision import *
bs = 256
df = pd.read_csv('../input/scene_classification/scene_classification/train.csv')

df.head()
tfms = get_transforms(flip_vert=False,max_zoom=1.0,max_warp=0)

data = (ImageList.from_csv(path, csv_name = '../train.csv') 

        .split_by_rand_pct()              

        .label_from_df()            

        .add_test_folder(test_folder = '../test')              

        .transform(tfms, size=256)

        .databunch(num_workers=0))
data.show_batch(rows=3, figsize=(8,10))
print(data.classes)
learn_34 = cnn_learner(data, models.resnet34, metrics=[error_rate, accuracy], model_dir="/tmp/model/")

learn_50 = cnn_learner(data, models.resnet50, metrics=[error_rate, accuracy], model_dir="/tmp/model/")

learn_101 = cnn_learner(data, models.resnet101, metrics=[error_rate, accuracy], model_dir="/tmp/model/")
learn_34.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn_34)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(7,7), dpi=60)
learn_34.save('/kaggle/working/resnet34-stage1')
learn_34.unfreeze()
learn_34.lr_find()
learn_34.recorder.plot()
learn_34.fit_one_cycle(1, max_lr=slice(1e-6, 1e-4))
learn_34.save('/kaggle/working/resnet34-stage2')
learn_50.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn_50)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(7,7), dpi=60)
learn_50.save('/kaggle/working/resnet50-stage1')
learn_50.unfreeze()
learn_50.lr_find()
learn_50.recorder.plot()
learn_50.fit_one_cycle(1, max_lr=slice(1e-6, 1e-4))
learn_50.save('/kaggle/working/resnet50-stage2')
learn_101.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn_101)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(7,7), dpi=60)
learn_101.save('/kaggle/working/resnet101-stage1')
learn_101.unfreeze()
learn_101.lr_find()
learn_101.recorder.plot()
learn_101.fit_one_cycle(1, max_lr=slice(1e-6, 1e-4))
learn_101.save('/kaggle/working/resnet101-stage2')