%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *
import jovian
#!pip install jovian
#! {sys.executable} -m pip install kaggle --upgrade
#! mkdir -p ~/.kaggle

#! mv kaggle.json ~/.kaggle
path = Config.data_path()/'planet'

path.mkdir(parents=True, exist_ok=True)

path
#planet_tiny = untar_data(URLs.PLANET_TINY)

planet = untar_data(URLs.PLANET_SAMPLE)

planet_tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
planet_tiny.ls()
df = pd.read_csv(planet/'labels.csv');df.head()
len(df)
np.random.seed(42)

#Creating a databunch using Factory method approach

#data = (ImageDataBunch.from_csv(planet_tiny, folder='train', size=128, 

#                         suffix='jpg', label_delim=' ', ds_tfms=planet_tiny_tfms))

# #### Above is not working #####                        



#Datablock api

#

source = (ImageList.from_df(df, path=planet, folder='train', suffix='.jpg')

       .split_by_rand_pct(0.2)

       .label_from_df(label_delim=' '))
data = (source.transform(planet_tfms) #Reduce the size if needed

       .databunch().normalize(imagenet_stats))
data.show_batch(rows=3, figsize=(12,12))
arch = models.resnet50
#doc(cnn_learner)
# Do not understand the below 2 lines

acc_02 = partial(accuracy_thresh, thresh=0.2)

f_score = partial(fbeta, thresh=0.2)

learner = cnn_learner(data, arch, pretrained=True, metrics=[acc_02, f_score])
learner.lr_find()
learner.recorder.plot()
lr = 0.01
#doc(learner.fit_one_cycle)
learner.fit_one_cycle(cyc_len=5, max_lr=slice(lr))
learner.save('stage-1-rn50')
learner.unfreeze()
learner.lr_find()
learner.recorder.plot()
learner.fit_one_cycle(cyc_len=5, max_lr=slice(1e-5, lr/5))
learner.save('stage-2-rn50')
data = (src.transform(tfms=planet_tfms, size=256)

        .databunch()

        .normalize(imagenet_stats))
learner.data = data

data.train_ds[0][0].shape
#doc(learner.freeze) 
#learner.unfreeze()
#learner.freeze() #https://docs.fast.ai/basic_train.html#Learner.freeze
learner.model
learner.freeze()
learner.model
learner.lr_find()
jovian.commit()