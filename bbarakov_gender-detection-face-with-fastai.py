from fastai.vision import *
path = Path('../input/dataset1/dataset1')

path.ls()
classes = ['man', 'woman']
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
data.show_batch(rows=2, figsize=(7,8))
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir='/tmp/.torch/models/')
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-5))
learn.save('stage-2')
learn.load('stage-2')
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
from fastai.widgets import *
db = (ImageList.from_folder(path)

                   .split_none()

                   .label_from_folder()

                   .transform(get_transforms(), size=224)

                   .databunch()

     )
learn_cln = cnn_learner(db, models.resnet34, metrics=error_rate, model_dir='/tmp/.torch/models/')



learn_cln.load('stage-2');
ds, idxs = DatasetFormatter().from_toplosses(learn_cln)
ImageCleaner(ds, idxs, Path('/tmp'))
ds, idxs = DatasetFormatter().from_similars(learn_cln)
ImageCleaner(ds, idxs, Path('/tmp'), duplicates=True)
db = (ImageList.from_csv(Path('/tmp'), 'cleaned.csv', folder='')

                   .split_none()

                   .label_from_df()

                   .transform(get_transforms(), size=224)

                   .databunch()

     )
learn_cln = cnn_learner(db, models.resnet34, metrics=error_rate, model_dir='/tmp/.torch/models/')



learn_cln.load('stage-2');
learn.fit_one_cycle(4)
img = open_image(path/'test'/'woman'/'face_8.jpg')

img
pred_class,pred_idx,outputs = learn.predict(img)

pred_class