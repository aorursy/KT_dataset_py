%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *
fnames = get_files('/kaggle/input/pap-smear-datasets/herlev_pap_smear/',extensions=('.bmp','.BMP'),recurse=recurse);fnames[:5]
pat=r'(\w+)(?=\/\d)'

data = ImageDataBunch.from_name_re('/kaggle/input/pap-smear-datasets/herlev_pap_smear/', 

                                   fnames, pat, ds_tfms=get_transforms(), size=224, bs=64

                                  ).normalize(imagenet_stats)
print(data.classes)

data.show_batch(rows=3,figsize=(9,6))
training_models = {

    'resnet34': models.resnet34,

    'resnet18':models.resnet18,

    'densenet121':models.densenet121,

    'resnet50':models.resnet50,

    'resnet101':models.resnet101,

}
interpreters=[]

for model in list(training_models.keys()):

    print("Training with model "+model)

    learn = cnn_learner(data, training_models[model], metrics=accuracy)

    learn.fit_one_cycle(10)

    interp = ClassificationInterpretation.from_learner(learn)

    interpreters.append(interp)
interpreters[0].plot_confusion_matrix(figsize=(6,6),title='resnet34')
interpreters[1].plot_confusion_matrix(figsize=(6,6),title='resnet18')
interpreters[2].plot_confusion_matrix(figsize=(6,6),title='densenet121')
interpreters[3].plot_confusion_matrix(figsize=(6,6),title='resnet50')
interpreters[4].plot_confusion_matrix(figsize=(6,6),title='resnet101')