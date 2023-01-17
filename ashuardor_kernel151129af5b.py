%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai import *;
from fastai.vision import *;
#path=Path("../input/the-oxfordiiit-pet-dataset/images")
path=Path("../input/asl-alphabet/asl_alphabet_train/asl_alphabet_train")
fnames=get_image_files(path)
fnames[:5]
np.random.seed(2)
pat=r'/([^/]+)\d+.jpg$'
#data=ImageDataBunch.from_name_re(path,fnames,pat,ds_tfms=get_transforms(),size=224)
data = (ImageList.from_folder(path).random_split_by_pct().label_from_folder().transform(get_transforms(), size=224).databunch())
data.normalize(imagenet_stats)
data.show_batch(rows=3,figsize=(7,6))
print(data.classes)
len(data.classes), data.c
path = Path("/kaggle/working")

learn = cnn_learner(data, models.resnet34, metrics=error_rate,path= path)
learn.fit_one_cycle(2)
#img=open_image("../input/the-oxfordiiit-pet-dataset/images/Abyssinian_106.jpg")
img=open_image("../input/asl-alphabet/asl_alphabet_test/asl_alphabet_test/D_test.jpg")
img
learn.predict(img)
learn.export()