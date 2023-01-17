!curl -s https://course.fast.ai/setup/colab | bash

from fastai.vision import *

from fastai.tabular import *

from pathlib import Path
cp -R /kaggle/input/platesv2/plates/plates /kaggle/input/plate
ls /kaggle/input/plate
path = Path('/kaggle/input/plate')
np.random.seed(42)

data = ImageDataBunch.from_folder(path/"train", train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), bs=16, size=224, num_workers=4).normalize(imagenet_stats)
data.classes
data.show_batch(rows=5, figsize=(7,8))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
learn = cnn_learner(data, models.resnet18, metrics=error_rate)
learn.fit_one_cycle(100)
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(200, max_lr=slice(3e-6,2e-6))
learn.save("/kaggle/working/stage")
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
img = open_image(path/'test/0000.jpg')

img
pred_class,pred_idx,outputs = learn.predict(img)

pred_class
from os import listdir

sample_list = [f[:4] for f in listdir(path/"test")]

sample_list.sort()
pred_list_cor = []

for f in sample_list :

    file = f+".jpg"

    p,_,_ = learn.predict(open_image(path/"test"/file))

    pred_list_cor.append(p.obj)
final_df = pd.DataFrame({'id':sample_list,'label':pred_list_cor})

final_df.to_csv('/kaggle/working/plate_submission.csv', header=True, index=False)
final_df.head()