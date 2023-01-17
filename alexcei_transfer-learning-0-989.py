import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pathlib import Path

from datetime import datetime

import numpy as np



import torch

import torch.nn as nn

from torch.utils import data



import catalyst

from catalyst import dl

from catalyst.utils import metrics, set_global_seed
set_global_seed(42)
from catalyst.utils import (

    create_dataset, create_dataframe, get_dataset_labeling, map_dataframe

)



dataset = create_dataset(dirs=f"../input/Imagenette-comp/train/*", extension="*.jpg")

df = create_dataframe(dataset, columns=["class", "filepath"])



tag_to_label = get_dataset_labeling(df, "class")

class_names = [

    name for name, id_ in sorted(tag_to_label.items(), key=lambda x: x[1])

]



df_with_labels = map_dataframe(

    df, 

    tag_column="class", 

    class_column="label", 

    tag2class=tag_to_label, 

    verbose=False

)

df_with_labels.head()
from catalyst.utils import split_dataframe_train_test



train_data, valid_data = split_dataframe_train_test(

    df_with_labels, test_size=0.2, random_state=42

)

train_data, valid_data = (

    train_data.to_dict("records"),

    valid_data.to_dict("records"),

)
from catalyst.data.cv.reader import ImageReader

from catalyst.dl import utils

from catalyst.data import ScalarReader, ReaderCompose



num_classes = len(tag_to_label)



open_fn = ReaderCompose(

    [

        ImageReader(

            input_key="filepath", output_key="features", rootpath="../input/Imagenette-comp/train"

        ),

        ScalarReader(

            input_key="label",

            output_key="targets",

            default_value=-1,

            dtype=np.int64,

        ),

        ScalarReader(

            input_key="label",

            output_key="targets_one_hot",

            default_value=-1,

            dtype=np.int64,

            one_hot_classes=num_classes,

        ),

    ]

)
import albumentations as albu

from albumentations.pytorch import ToTensorV2 as ToTensor



IMAGE_SIZE = 224



train_transform = albu.Compose([

    albu.HorizontalFlip(p=0.5),

    albu.LongestMaxSize(IMAGE_SIZE),

    albu.PadIfNeeded(IMAGE_SIZE, IMAGE_SIZE, border_mode=0),

    albu.RandomResizedCrop(IMAGE_SIZE, IMAGE_SIZE, p=0.3),

    albu.Normalize(),

    ToTensor(),

])



valid_transform = albu.Compose([

    albu.LongestMaxSize(IMAGE_SIZE),

    albu.PadIfNeeded(IMAGE_SIZE, IMAGE_SIZE, border_mode=0),

    albu.Normalize(),

    ToTensor(),

])

from catalyst.data import Augmentor



train_data_transform = Augmentor(

    dict_key="features", augment_fn=lambda x: train_transform(image=x)["image"]

)



valid_data_transform = Augmentor(

    dict_key="features", augment_fn=lambda x: valid_transform(image=x)["image"]

)

batch_size = 256

num_workers = 4



train_loader = utils.get_loader(

    train_data,

    open_fn=open_fn,

    dict_transform=train_data_transform,

    batch_size=batch_size,

    num_workers=num_workers,

    shuffle=True,

    sampler=None,

    drop_last=True,

)



valid_loader = utils.get_loader(

    valid_data,

    open_fn=open_fn,

    dict_transform=valid_data_transform,

    batch_size=batch_size,

    num_workers=num_workers,

    shuffle=False, 

    sampler=None,

    drop_last=True,

)



loaders = {

    "train": train_loader,

    "valid": valid_loader

}
from torchvision import transforms, models



class MyResNet50(torch.nn.Module):

    def __init__(self):

        super(MyResNet50, self).__init__()

        self.net = models.resnet50(pretrained=True)

        

        # Disable grad for all conv layers

        for param in self.net.parameters():

            param.requires_grad = False                

        

        # Create some additional layers for ResNet model

        fc_inputs = self.net.fc.in_features

        self.net.fc = torch.nn.Sequential(

            torch.nn.Linear(fc_inputs, 256),

            torch.nn.ReLU(),

            torch.nn.Linear(256, 128),

            torch.nn.Sigmoid(),

            torch.nn.Linear(128, 10),

        )  

    def forward(self, x):

        x = self.net(x)

        return x
class ResNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, p=0.1):

        super().__init__()



        self.input = nn.Sequential(

            nn.Conv2d(

                in_channels,

                out_channels,

                kernel_size=3,

                stride=stride,

                padding=1,

            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(),

            nn.Conv2d(

                out_channels, out_channels, kernel_size=3, stride=1, padding=1

            ),

            nn.BatchNorm2d(out_channels),

        )

        self.res = nn.Conv2d(

            in_channels, out_channels, kernel_size=1, stride=stride

        )

        self.output = nn.Sequential(nn.BatchNorm2d(out_channels), nn.ReLU())



    def forward(self, x):

        input = self.input(x)

        res = self.res(x)

        return self.output(res + input)





class BaselineModel(nn.Module):

    def __init__(self, channels=3, in_features=64, num_classes=10, p=0.1):

        super().__init__()



        self.input = nn.Sequential(

            nn.Conv2d(

                channels, in_features, kernel_size=7, stride=2, padding=3

            ),

            nn.BatchNorm2d(in_features),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2),

        )



        self.layer_0 = self._make_layer(in_features, 1)

        self.layer_1 = self._make_layer(in_features)

        in_features *= 2

        self.layer_2 = self._make_layer(in_features)

        in_features *= 2

        self.layer_3 = self._make_layer(in_features)



        self.fc = nn.Sequential(

            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),

            nn.Linear(2 * in_features, num_classes),

        )



    def _make_layer(self, in_features, multiplier=2, p=0.1):

        return nn.Sequential(

            ResNetBlock(in_features, in_features * multiplier, stride=2, p=p),

            ResNetBlock(

                in_features * multiplier,

                in_features * multiplier,

                stride=1,

                p=p,

            ),

        )



    def forward(self, x):

        x = self.input(x)

        x = self.layer_0(x)

        x = self.layer_1(x)

        x = self.layer_2(x)

        x = self.layer_3(x)

        return self.fc(x)
from catalyst.dl import SupervisedRunner



class ClassificationRunner(SupervisedRunner):

    def predict_batch(self, batch):

        prediction = {

            "filepath": batch["filepath"],

            "log_probs": self.model(batch[self.input_key].to(self.device))

        }

        return prediction
model = MyResNet50()



# model = BaselineModel()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)



runner = ClassificationRunner(input_key="features", input_target_key="targets")

runner.train(

    model=model,

    optimizer=optimizer,

    criterion=criterion,

    loaders=loaders,

    logdir=Path("logs") / datetime.now().strftime("%Y%m%d-%H%M%S"),

    num_epochs=10,

    verbose=True,

    load_best_on_end=True,

    callbacks={

        "optimizer": dl.OptimizerCallback(

            metric_key="loss", accumulation_steps=1, grad_clip_params=None,

        ),

        "criterion": dl.CriterionCallback(

            input_key="targets", output_key="logits", prefix="loss",

        ),

        "accuracy": dl.AccuracyCallback(num_classes=10),

    },

)
import pandas as pd

from PIL import Image

from tqdm.notebook import tqdm



submission = {"Id": [], "Category": []}

model.eval()



test_dataset = create_dataset(dirs=f"../input/Imagenette-comp/test/", extension="*.jpg")

test_data = list({"filepath": filepath} for filepath in test_dataset["test"])



test_open_fn = ReaderCompose(

    [

        ImageReader(

            input_key="filepath", output_key="features", rootpath=""

        ),

        ScalarReader(

            input_key="filepath",

            output_key="filepath",

            default_value="",

            dtype=str,

        ),

    ]

)



test_loader = utils.get_loader(

    test_data,

    open_fn=test_open_fn,

    dict_transform=valid_data_transform,

    batch_size=batch_size,

    num_workers=num_workers,

    shuffle=False,

    sampler=None,

    drop_last=False,

)



for prediction in runner.predict_loader(loader=test_loader):

    prediction["labels"] = [class_names[c] for c in torch.max(prediction["log_probs"], axis=1)[1]]

    submission["Id"].extend(f.split("/")[4].split(".")[0] for f in prediction["filepath"])

    submission["Category"].extend(prediction["labels"])
pd.DataFrame(submission).to_csv("baseline.csv", index=False)