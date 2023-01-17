import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai import * # import the FastAI v3 lib which includes pytorch

from fastai.vision import  * # import all of the computer vision related libs from vision 



# lets import our necessary magic libs

%reload_ext autoreload

%autoreload 2

%matplotlib inline



import warnings

warnings.filterwarnings("ignore")

BATCH_SIZE = 64 

IMG_SIZE = 224

WORKERS = 0 

DATA_PATH_STR = '../input/parkinsons-drawings/'

DATA_PATH_OBJ = Path(DATA_PATH_STR)
tfms = get_transforms() # standard data augmentation ()



data = (ImageList.from_folder(DATA_PATH_OBJ)        # get data from path object

        .split_by_rand_pct()                        # separate 20% of data for validation set

        .label_from_folder()                          # label based on directory

        .transform(tfms, size=IMG_SIZE)                   # added image data augmentation

        .databunch(bs=BATCH_SIZE, num_workers=WORKERS)    # create ImageDataBunch

        .normalize(imagenet_stats))                   # normalize RGB vals using imagenet stats

('training DS size:', len(data.train_ds), 'validation DS size:' ,len(data.valid_ds))

data.classes
data.show_batch(rows=4, figsize=(10,8))
learn = cnn_learner(data, models.resnet34, metrics=accuracy, model_dir='/models')
learn.lr_find()

learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(10, max_lr = slice(5.74e-03/10))
learn.export('/kaggle/working/model.pkl')

learn.save('stage-1-93')
interp = ClassificationInterpretation.from_learner(learn)

interp.top_losses()

interp.plot_top_losses(9, figsize=(15,11))
interp.most_confused()
learn.save('/kaggle/working/R34_93_model')
learn.precompute=False

learn.unfreeze()
learn.load('/kaggle/working/R34_93_model')
import torch

import torch.nn as nn

import numpy as np

from torch.autograd import Variable

import torch.onnx

import torchvision

import onnx



sz = 299



def check_onnx_compatible(model, model_name, sz, input_names, output_names):

    dummy_input = Variable(torch.randn(3, sz, sz)).cuda()



    torch.onnx.export(model, dummy_input, \

                      model_name, input_names = input_names, output_names = output_names, verbose=True)

    

    

    # Check again by onnx

    # Load the ONNX model

    onnx_model = onnx.load(model_name)



    # Check that the IR is well formed

    onnx.checker.check_model(onnx_model)



    # Print a human readable representation of the graph

#     onnx.helper.printable_graph(onnx_model.graph)

    print("Done")

    return onnx_model
class ImageScale(nn.Module):

    def __init__(self): 

        super().__init__()

        self.denorminator = torch.full((3, sz, sz), 255.0, device=torch.device("cuda"))



    def forward(self, x): return torch.div(x, self.denorminator).unsqueeze(0)

final_model = [ImageScale()] + (list(learn.model.children())[:-1] + [nn.Softmax()])
final_model = nn.Sequential(*final_model)
model_name = "parkinsons-detector.onnx"



# Convert Pytorch model to onnx model & check if it is convertible

onnx_model = check_onnx_compatible(final_model, model_name, sz, input_names=['image'], output_names=['parkinsonsorhealthy'])