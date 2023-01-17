%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *
bs = 64

# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart
path = untar_data(URLs.PETS); path
path_img = path/'images'

fnames = get_image_files(path_img)

fnames[:5]
fnames = sorted(fname for fname in fnames if fname.name[0].lower() in 'abc')

len(fnames)
np.random.seed(2)

pat = re.compile(r'/([^/]+)_\d+.jpg$')

data = ImageDataBunch.from_name_re(path_img, fnames, pat,

                                   ds_tfms=get_transforms(),

                                   size=224, bs=bs, num_workers=0

                                  ).normalize(imagenet_stats)

data
data.show_batch(rows=3, figsize=(7,6))
print(data.classes)

len(data.classes),data.c
def badLearner() :    

    model = nn.Sequential(

        Flatten(),

        nn.Linear(224*224*3,data.c)

    )

    return Learner(data,model,metrics=accuracy)
def randomSeedForTraining(seed) :

    "This is to make the training demos below repeatable."

    random.seed(seed)

    torch.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)



randomSeedForTraining(3141);

learn = badLearner()
print(learn.summary())
learn.fit(3,1.0)
randomSeedForTraining(314);

learn = badLearner()

learn.lr_find(); learn.recorder.plot()
learn.fit(5,3e-5)
randomSeedForTraining(3141);

learn = create_cnn(data, models.resnet34, metrics=accuracy, pretrained=False)
learn.lr_find(); learn.recorder.plot()
learn.fit(8,5e-3)
randomSeedForTraining(3141);

learn = create_cnn(data, models.resnet34, metrics=accuracy)
print(learn.summary())
learn.fit_one_cycle(3,5e-3)
learn.save('stage-1')
learn.unfreeze()

learn.fit_one_cycle(1,5e-3)
learn.load('stage-1')  # forget about the training above and revert to stage 1

learn.unfreeze()

learn.lr_find(); learn.recorder.plot()
learn.fit_one_cycle(2,slice(1e-5,5e-4))
randomSeedForTraining(3141);

learn = create_cnn(data, models.resnet34, metrics=accuracy, wd=0.1)

learn.fit_one_cycle(3,5e-3)
randomSeedForTraining(3141);

learn = create_cnn(data, models.resnet34, metrics=accuracy, ps=[0.1,0.2])

learn.fit_one_cycle(3,5e-3)
data = ImageDataBunch.from_name_re(path_img, fnames, pat,

                                   ds_tfms=get_transforms(flip_vert=True,max_warp=0.1),

                                   size=224, bs=bs, num_workers=0

                                  ).normalize(imagenet_stats)