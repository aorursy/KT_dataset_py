import warnings
warnings.filterwarnings("ignore")
%reload_ext autoreload
%autoreload 2
%matplotlib inline
from fastai import *
from fastai.vision import *
import os, os.path
#fastai.defaults.device = torch.device('cpu')
path = Path('/kaggle/input/birds-uk-8classes-subset/Aves_subset_8classes_distributionSimilarToFullSet')
path.ls()
img = open_image(path/'mergustest.jpg')
img
#classes = ['Carduelis cannabina', 'Larus canus', 'Larus delawarensis', 'Leptoptilos crumenifer', 'Mergus serrator', 'Paroaria coronata', 'Saxicola rubetra', 'Tadorna ferruginea']
#data2 = ImageDataBunch.single_from_classes(path, classes, ds_tfms=get_transforms(), size=256).normalize(imagenet_stats)
#came from the docs
path2 = Path('/kaggle/input/predict-test')
learn = load_learner(path2, 'model_birds_8classes_BM_50eps.pkl')
classes = learn.data.classes
#used in L2, only loads a .pth checkpoint?
#learn = create_cnn(data2, models.resnet34).load('#nameofmodel')
pred_class,pred_idx,outputs = learn.predict(img)
print(pred_class)
print(pred_idx)
print(outputs)