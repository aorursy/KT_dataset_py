



# lets import our necessary magic libs

%reload_ext autoreload

%autoreload 2

%matplotlib inline

from fastai.vision import *
classes = ['off_white_AF1','off_white_airmax90','off_white_airmax97','off_white_blazer',

           'off_white_chicago','off_white_converse','off_white_hyperdunk',

           'off_white_presto','off_white_vapormax', 'off_white_zoomfly']

path = Path('/kaggle/working')

write_path = Path('../input')

def download_and_verify():

    for c in classes: 

        print( '***** Currently Downloading: {} *****'.format(c))

        file = c[10:] + '.csv'

        dest = path/c

        download_images(write_path/file,dest, max_pics=200)

        verify_images(path/c, delete=True, max_workers=8)

        

download_and_verify()

    
np.random.seed(7)

data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,

                                 ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
('training DS size:', len(data.train_ds), 'validation DS size:' ,len(data.valid_ds))
# show me what we got 

data.show_batch(rows=3, figsize=(7,9))
learn = cnn_learner(data, models.resnet34, metrics=[error_rate, accuracy])
# train one cycle

learn.fit_one_cycle(10)
# lets see and interpretation of how well our model did 

interpretation = ClassificationInterpretation.from_learner(learn)
losses, indxs = interpretation.top_losses()

# show me the top losses of our model 

# where the items displayed are instances of where them odel was most confiden, yet incorrect

interpretation.plot_top_losses(9, figsize=(20,15))
from fastai.widgets import *



# keep in mind all fastai databunches have x&y vars

# where x = filenames 

# where y = labels

# valid_top_loss_paths = data.valid_ds.x[indxs]

# train_top_loss_paths = data.train_ds.x[indxs]



# we can also pass in from_similars or from_most_unsure if we want those 

ds, idxs = DatasetFormatter().from_toplosses(learn)
ImageCleaner(ds, idxs, path)
df = pd.read_csv(path/'cleaned.csv', header='infer')

df.head(20)
# ImageDataBunch.from_folder(path, train=".", valid_pct=0.2, ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

clean_data = ImageDataBunch.from_csv(".", folder=".", csv_labels="cleaned.csv", valid_pct=0.2, ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
('training DS size:', len(clean_data.train_ds), 'validation DS size:' ,len(clean_data.valid_ds))
clean_learn = cnn_learner(clean_data, models.resnet34, metrics=[error_rate, accuracy])
clean_learn.fit_one_cycle(10)
clean_learn.unfreeze() #unfreeze the remaining layers

clean_learn.fit_one_cycle(2) #retrain 

clean_learn.save('stage1-no-tuning')
clean_learn.lr_find()

clean_learn.recorder.plot(suggestion=True)
clean_learn.fit_one_cycle(20,max_lr=slice(1e-4, 1e-6))
clean_learn.recorder.plot_losses()
learn.save('stage2-90')