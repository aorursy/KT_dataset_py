from fastai.vision import *
dirs = [

    ('black', 'urls_black.csv'),

    ('grizzly', 'urls_grizzly.csv'), 

    ('teddys', 'urls_teddy.csv'),

]

inputPath = Path('../input/bears/datasets')
for folder, file in dirs:

    print(folder, file)

    path = Path('data/outputs')

    dest = path/folder

    dest.mkdir(parents=True, exist_ok=True)

    try:

        download_images(inputPath/folder/file, dest)

    except:

        pass
for c, _ in dirs:

    verify_images(path/c, delete=True, max_size=500)
np.random.seed(1)

data = ImageDataBunch.from_folder(path, train='.', valid_pct=0.2, ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
data.show_batch(rows=3, figsize=(7,8))
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
learn.save('stage-1')
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(2, max_lr=slice(1e-5, 1e-4))
learn.save('stage-2')
learn.load('stage-2')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(6)
learn.export()
defaults.device = torch.device('cpu')
img = open_image(path/'black'/'00000005.jpg')

img
learn = load_learner(path)
pred_class, pred_idx, outputs = learn.predict(img)

pred_class
!cp ./data/outputs/export.pkl .
from IPython.display import FileLink

FileLink('export.pkl')



!rm -r ./data/
!ls .