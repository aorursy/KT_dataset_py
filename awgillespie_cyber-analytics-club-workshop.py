from fastai import *

from fastai.vision import *
# Exclude first gen commodores, they are giving me headaches



classes = ['moustache', 'beard', 'fumanchu']
!ls -la
# Explain your data set in one word

data_name = "facial_hair"
for model in classes:

    folder = model

    file = model + '.csv'

    path = Path("data/" + data_name)

    print(file)

    dest = path/folder

    dest.mkdir(parents=True, exist_ok=True)

    !cp ../input/* {path}/

    download_images(path/file, dest, max_pics=299)
for c in classes:

    print(c)

    verify_images(path/c, delete=True, max_size=500)
np.random.seed(42)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

        ds_tfms=get_transforms(), size=224, num_workers=0).normalize(imagenet_stats)


# If you already cleaned your data, run this cell instead of the one before

# np.random.seed(42)

# data = ImageDataBunch.from_csv(path, folder=".", valid_pct=0.2, csv_labels='cleaned.csv',

#         ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
data.classes
data.show_batch(rows=3, figsize=(9,12))
learner = cnn_learner(data, models.resnet34, metrics=error_rate)
learner.metrics=[accuracy]

learner.fit_one_cycle(6)
learner.save('stage-1')
learner.unfreeze()
learner.lr_find()
learner.recorder.plot()
learner.fit_one_cycle(2, max_lr=slice(1e-4,1e-3))
learner.save('stage-2')
learner.load('stage-2');
interp = ClassificationInterpretation.from_learner(learner)
interp.most_confused(min_val=2)[:10]
interp.plot_confusion_matrix()
from fastai.widgets import *
ds, idxs = DatasetFormatter().from_toplosses(learner, n_imgs=100)
ImageCleaner(ds, idxs, path)
ds, idxs = DatasetFormatter().from_similars(learner)
#import fastai

#fastai.defaults.device = torch.device('cpu')
#img = open_image(path/'black'/'00000021.jpg')

#img
#classes = ['black', 'grizzly', 'teddys']
#data2 = ImageDataBunch.single_from_classes(path, classes, tfms=get_transforms(), size=224).normalize(imagenet_stats)
#learn = create_cnn(data2, models.resnet34).load('stage-2')
#pred_class,pred_idx,outputs = learn.predict(img)

#pred_class
#learn = create_cnn(data, models.resnet34, metrics=error_rate)
#learn.fit_one_cycle(5, max_lr=1e-5)
#learn.recorder.plot_losses()
#learn = create_cnn(data, models.resnet34, metrics=error_rate, pretrained=False)
#learn.fit_one_cycle(1)
# np.random.seed(42)

# data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.9, bs=32, 

#        ds_tfms=get_transforms(do_flip=False, max_rotate=0, max_zoom=1, max_lighting=0, max_warp=0

#                              ),size=224, num_workers=4).normalize(imagenet_stats)
# learn = create_cnn(data, models.resnet50, metrics=error_rate, ps=0, wd=0)

# learn.unfreeze()
# learn.fit_one_cycle(40, slice(1e-6,1e-4))