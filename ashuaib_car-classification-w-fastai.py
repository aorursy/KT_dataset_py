%reload_ext autoreload

%autoreload 2

%matplotlib inline



from fastai import *

from fastai.vision import *



bs = 64
cars = ['Accord','Altima','Charger', 'Corolla','Tesla_s','Civic']
path = Path('../data/car')



for car in cars:

    (path/car).mkdir(parents=True, exist_ok=True)



path.ls()
for car in cars:

    download_images(Path('../input')/f'{car.lower()}', path/car, max_pics=500)
for car in cars:

    print(car)

    verify_images(path/car,delete =True,max_size=500)
np.random.seed(123)

data = ImageDataBunch.from_folder(path,train=".",valid_pct=0.2,ds_tfms=get_transforms(),

                                  size=224,bs=bs,num_workers=0).normalize(imagenet_stats)
print((len(data.classes),data.c))

data.show_batch(rows=3, figsize=(7,6))
learn = create_cnn(data, models.resnet50, metrics=error_rate)
learn.fit_one_cycle(8)

learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_top_losses(9, figsize=(15,11))
interp.plot_confusion_matrix()
interp.most_confused(min_val=2)
learn.unfreeze()

for i in range(8):

    learn.fit_one_cycle(2)

    learn.save('stage-'+str(i+2))

#learn.load('stage-1')
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4, max_lr=slice(6e-5,4e-4))
learn.save('lr-optimized')
interp=ClassificationInterpretation.from_learner(learn)

interp.most_confused(min_val=2)

interp.plot_confusion_matrix()
from fastai.widgets import *
ds, idxs = DatasetFormatter().from_toplosses(learn, ds_type=DatasetType.Valid)

ImageCleaner(ds, idxs, path)
#After cleaning data, run this cell

np.random.seed(42)

csvpath=Path(path/'cleaned.csv')

data = ImageDataBunch.from_csv(".", folder=".", valid_pct=0.2, csv_labels=csvpath,

    ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
#After cleaning data, run this cell

np.random.seed(42)

csvpath=Path(path/'cleaned.csv')

data = ImageDataBunch.from_csv(".", folder=".", valid_pct=0.2, csv_labels=csvpath,

    ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)