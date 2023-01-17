!pip install fastai2 > /dev/null
import fastai2

from fastai2.vision.all import *
path = Path('../input/prostate-cancer-grade-assessment')

path.ls()
df = pd.read_csv(path/'train.csv')

img_path = Path('../input/panda-train-png-images/train/')
df.head(3)
# add .png to filenames

df['image_id'] = df['image_id'].apply(lambda x: str(x)+'.png')

df.head(3)
prostates = DataBlock(blocks=(ImageBlock, CategoryBlock),

                   splitter=RandomSplitter(),

                   get_x=ColReader(0, pref=img_path),

                   get_y=ColReader(1),

                   item_tfms=Resize(224),

                   batch_tfms=aug_transforms()

                     )
dls = prostates.dataloaders(df, bs=16)

dls.show_batch()
dls.show_batch()
learn = cnn_learner(dls, resnet50, metrics=accuracy)

learn.fit_one_cycle(1)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(k = 9)