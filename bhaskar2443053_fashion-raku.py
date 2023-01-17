%reload_ext autoreload

%autoreload 2

%matplotlib inline

import os

import pandas as pd

from fastai import *

from fastai.vision import *
!pwd
path = "/kaggle/input/fashion_small/fashion_small"

print(os.listdir(path))
df = pd.read_csv("/kaggle/input/fashion_small/fashion_small/styles.csv", error_bad_lines=False);
df.head()
l=[]

for i in df['id']:

    if not os.path.exists('/kaggle/input/fashion_small/fashion_small/resized_images/'+str(i) +".jpg"):

        l.append(i)

        df.drop(df[df.id == i].index, inplace=True)
bs=64
!nvidia-smi
src=(ImageList.from_df(df, path=path, folder='resized_images', suffix='.jpg', cols=0)

                .split_by_rand_pct(0.2)

                .label_from_df( cols=3)

                .transform(get_transforms(), size=224)

                .databunch(bs=bs,num_workers=0)).normalize(imagenet_stats)
src.show_batch()
learn = create_cnn(

    src,

    models.resnet34,

    path='.',    

    metrics=accuracy, 

    ps=0.5

)
learn.lr_find()
learn.recorder.plot(skip_end=5)
learn.fit_one_cycle(5, 1e-2)
learn.save('freeze_1')
learn.recorder.plot_losses()

learn.unfreeze()

learn.fit_one_cycle(6, max_lr=slice(1e-4,1e-3))
learn.recorder.plot_losses()
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(src.valid_ds)==len(losses)==len(idxs)
len(src.classes)
doc(interp.plot_top_losses)
interp.plot_top_losses(9, figsize=(15,11),heatmap=False)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.most_confused(min_val=2)
learn.save('/kaggle/working/unfreeze')