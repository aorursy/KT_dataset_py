path = "../input/scene_classification/scene_classification/train/"
from fastai import *

from fastai.vision import *
bs = 256
df = pd.read_csv('../input/scene_classification/scene_classification/train.csv')

df.head()
df_sub = df[(df['label'] != 2) & (df['label'] != 5)]

df_sub['label'].value_counts()
tfms = get_transforms(flip_vert=False,max_zoom=1.0,max_warp=0)
allClasses = [0,1,2,3,4,5]
data_small_sub = (ImageList.from_df(df_sub,path) 

        .split_by_rand_pct()              

        .label_from_df(classes=allClasses)            

        .add_test_folder(test_folder = '../test')              

        .transform(tfms, size=128)

        .databunch(num_workers=0))
data_small_sub.show_batch(rows=3, figsize=(8,10))
data_large_full = (ImageList.from_df(df,path) 

        .split_by_rand_pct()              

        .label_from_df(classes=allClasses)            

        .add_test_folder(test_folder = '../test')              

        .transform(tfms, size=256)

        .databunch(num_workers=0))
data_large_full.show_batch(rows=3, figsize=(8,10))
print(data_small_sub.classes)

print(data_large_full.classes)
learn_101 = cnn_learner(data_small_sub, models.resnet101, metrics=[error_rate, accuracy], model_dir="/tmp/model/")
learn_101.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn_101)



losses,idxs = interp.top_losses()



len(data_small_sub.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(7,7), dpi=60)
learn_101.save('/kaggle/working/resnet101-size128-fewclasses-stage1')
learn_101.data = data_large_full
learn_101.unfreeze()
learn_101.lr_find()
learn_101.recorder.plot()
learn_101.fit_one_cycle(2, max_lr=slice(1e-5, 1e-4))
learn_101.save('/kaggle/working/resnet101-size256-allclasses-stage1')
interp = ClassificationInterpretation.from_learner(learn_101)



losses,idxs = interp.top_losses()



len(data_large_full.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(7,7), dpi=60)