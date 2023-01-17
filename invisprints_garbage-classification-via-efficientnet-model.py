%reload_ext autoreload

%autoreload 2

%matplotlib inline
!pip install pytorchcv
from torchvision.models import *

from fastai import *

from fastai.vision import *

from fastai.vision.models import *

from pytorchcv.model_provider import get_model as ptcv_get_model
path = Path('/kaggle/input/garbage-classify-v2/garbage_classify_v2');path.ls()

data_path = path/'train_data_v2'
fnames = get_image_files(data_path)

fnames[:5]
def get_labels(x): return np.loadtxt(data_path/f'{x.stem}.txt', dtype=str)[1]

# data = ImageDataBunch.from_name_func(path, fnames, get_labels,

#                                      ds_tfms=get_transforms(),size=125,bs=32).normalize(imagenet_stats)

np.random.seed(42)

src = (ImageList.from_folder(path)

        .split_by_rand_pct(0.2)

        .label_from_func(get_labels))

data = (src.transform(get_transforms(flip_vert=True), size=224)

        .databunch(bs=32)

        .normalize(imagenet_stats))

data
model_name = 'efficientnet_b3b'

def getModel(pretrained=True):

    return ptcv_get_model(model_name, pretrained=pretrained).features
learn = cnn_learner(data, getModel, pretrained=True,

                    cut=noop, split_on=lambda m: (m[0][4], m[1]),

                    metrics=accuracy, model_dir = '/kaggle/working/',

                    bn_wd=False, true_wd=True,

                    #loss_func=LabelSmoothingCrossEntropy(),

                    callback_fns=[ShowGraph])
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10, slice(1e-2))
learn.save('efn-1')
learn.unfreeze()

learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(5, slice(1e-5,1e-4))

learn.save('efn-2')
learn.export('/kaggle/working/export.pkl')
learn.model
learn.load('./efn-2')

learn.validate()
learn.data.classes
test_img = fnames[5]

img = open_image(test_img)

img
pred_class,idx,pro = learn.predict(img)

pred_class,idx,pro
pred_class.obj