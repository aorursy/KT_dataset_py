#You have to install torch 1.6, Fastai >=2.0.0 version.



!pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html



#Upgrade kornia and allennlp version since current version does not support torch 1.6



!pip install --upgrade kornia

!pip install allennlp==1.1.0.rc4



#Install/upgrade fastai package



!pip install --upgrade fastai
#Load the libraries and verify the versions



import torch

print(torch.__version__)

print(torch.cuda.is_available())



import fastai

print(fastai.__version__)



from fastai.vision.all import *

from fastai.vision.widgets import *
path = Path ('../input')
from PIL import Image

img = Image.open(path/'covidxray/03BF7561-A9BA-4C3C-B8A0-D3E585F73F3C.jpeg') 

print(img.shape)

img.to_thumb(128,128)
tfms = aug_transforms(do_flip = True, flip_vert = False, mult=2.0)
data=ImageDataLoaders.from_folder(path, train = "train",valid_pct=0.2, item_tfms=Resize(128), batch_tfms=tfms, bs = 30, num_workers = 4) 
data.train.show_batch(max_n=2, nrows=1)
learn = cnn_learner(data, resnet34, metrics=error_rate)

learn.fine_tune(4)
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()
interp.most_confused(min_val=2)
interp.plot_top_losses(5, nrows=2, figsize = (25,5))
btn_run = widgets.Button(description='Classify')

btn_upload = widgets.FileUpload()

out_pl = widgets.Output()

lbl_pred = widgets.Label()
def on_click_classify(change):

    img = PILImage.create(btn_upload.data[-1])

    out_pl.clear_output()

    with out_pl: display(img.to_thumb(128,128))

    pred,pred_idx,probs = learn.predict(img) #use learn_inf if the .pkl file is used

    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'



btn_run.on_click(on_click_classify)
VBox([widgets.Label('Select a X-ray'),btn_upload, btn_run, out_pl, lbl_pred])