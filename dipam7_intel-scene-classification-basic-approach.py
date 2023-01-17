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
learn = create_cnn(data, models.resnet34, metrics=[error_rate, accuracy], model_dir="/tmp/model/")
learn.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix(figsize=(7,7), dpi=60)
learn.save('/kaggle/working/stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(1, max_lr=slice(1e-6, 1e-4))
learn.save('/kaggle/working/stage-2')
# preds,_ = learn.get_preds(ds_type=DatasetType.Test)
# labelled_preds = []

# for pred in preds:

#     labelled_preds.append(int(np.argmax(pred)))

    

# # labelled_preds[0:10]

# len(labelled_preds)
# import os

# filenames = os.listdir('../input/scene_classification/scene_classification/test/')
# len(filenames) == len(labelled_preds)
# submission = pd.DataFrame(

#     {'image_name': filenames,

#      'label': labelled_preds,

#     })
# submission.to_csv('first_submission.csv')
# download the notebook before committing