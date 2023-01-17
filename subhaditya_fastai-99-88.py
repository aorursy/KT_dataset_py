from fastai import *

from fastai.vision import *

import torch.onnx

from torch.autograd import Variable

from fastai.callbacks import *

import os



# os.environ["TORCH_HOME"] = "/media/subhaditya/DATA/COSMO/Datasets-Useful"
path = Path('../input/fruits/fruits-360_dataset/fruits-360/Training/')
data = (

    (

        ImageList.from_folder(path)

        .split_by_rand_pct()

        .label_from_folder()

        .transform(get_transforms(), size=64)

    )

    .databunch(bs=64)

    .normalize(imagenet_stats)

)
data.show_batch(4)
data.c
data
learn = None

gc.collect()
learn = cnn_learner(

    data,models.resnet50, metrics=[accuracy,error_rate], callback_fns=[ShowGraph,partial(EarlyStoppingCallback, monitor='accuracy', min_delta=0.01, patience=3)]

).to_fp16()
learn.summary()
# learn.lr_find()

# learn.recorder.plot()
learn.fit_one_cycle(5, 2e-3)
learn.show_results()
# dummy_input = Variable(torch.randn(1, 3, 224, 224)).cuda().half()

# torch.onnx.export(learn.model, dummy_input, "model.onnx")
# netron.start("model.onnx")
# !rm "model.onnx"