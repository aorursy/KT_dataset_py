%reload_ext autoreload

%autoreload 2

%matplotlib inline



import numpy as np

import pandas as pd



import os



from fastai import *

from fastai.vision import *



from fastai import __version__ as fastai_version

print(f"fastai version: {fastai_version}")
# input paths

data_path = Path("../input/train")

label_path =Path("../input/train_labels.csv")

test_path = Path("../input/test")

models_path = Path("../models")



#output path

submission_path = Path("submission.csv")
models_path.mkdir(exist_ok=True)
input_paths = [data_path, label_path, models_path,test_path]

for p in input_paths:

    assert p.exists()

else:

    print("OK, all input paths exist")
size = 32 #96 #224 



tfms = get_transforms(flip_vert=True, max_rotate=180, max_lighting=0.1)

def crop_transform(img):

    return crop(img, size)

crop_transform = Transform(crop_transform, order=99)

# tfms = ([crop_transform()], [crop_transform()])

# tfms[0][0]



tfms = ([], []) # for debugging



# tfms[0].append(crop_transform())

# tfms[1].append(crop_transform())

tfms
import torch

d = torch.cuda.current_device()

torch.cuda.get_device_name(d)
np.random.seed(42)

image_lists = (ImageList.from_csv(label_path.parent, folder=data_path, csv_name=label_path.name, suffix=".tif")

    .random_split_by_pct(0.2)

    .label_from_df("label")

    .add_test(ImageList.from_folder(test_path))

    .transform(tfms)

    )

image_lists
bs = 2**6

print(f"batch size: {bs}")



np.random.seed(42)

data = (image_lists.databunch(bs=bs)

    .normalize(imagenet_stats)

    )
# data.show_batch()
# arch = models.resnet34

arch = models.resnet50

arch_name = arch.__name__

display(arch_name)



learn = create_cnn(data, arch, metrics=accuracy, path=models_path)

learn.save(f"{arch_name}-stage-0")
learn.fit_one_cycle(4, 3e-3)

learn.save(f"{arch_name}-stage-1")
learn.lr_find()

learn.recorder.plot()
# learn.load("resnet34-stage-1")

learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2, slice(2e-6, 5e-5))

learn.save(f"{arch_name}-stage-2")
learn.load(f"{arch_name}-stage-2")

learn.fit_one_cycle(2, slice(2e-6, 5e-5))

learn.save(f"{arch_name}-stage-3")
learn.load(f"{arch_name}-stage-3")

learn.fit_one_cycle(2, slice(2e-6, 5e-5))

learn.save(f"{arch_name}-stage-4")
learn.load(f"{arch_name}-stage-4");

p_valid, r_valid = learn.get_preds(DatasetType.Valid)

l_valid = p_valid.argmax(dim=1)

acc = float((l_valid == r_valid).sum()) / len(l_valid)

print(f"final accuracy: {acc}")
data.classes
probabilities, _ = learn.get_preds(DatasetType.Test)

labels = probabilities.argmax(dim=1)



ids = list(map(lambda p: p.stem, test_path.ls()))



submission_df= pd.DataFrame(dict(id=ids, label=labels))

display(submission_df.head(20))



display(submission_df.describe())



submission_df.to_csv(submission_path, index=False)



%cat $submission_path | wc -l