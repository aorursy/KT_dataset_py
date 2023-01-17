import pandas as pd

import os



from sklearn.model_selection import train_test_split

import torch

import torch.nn as nn

import torchvision.transforms as transforms

import torchvision

import torch.optim as optim

from torch.optim import lr_scheduler

import time

import copy

from torchvision.datasets.folder import default_loader, DatasetFolder
train = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")

test = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")



train = train[

    train.id_code.isin([i.split(".")[0] for i in os.listdir("../input/aptos2019-blindness-detection/train_images")])

]

test = test[

    test.id_code.isin([i.split(".")[0] for i in os.listdir("../input/aptos2019-blindness-detection/test_images")])

]



test["diagnosis"] = test["id_code"].copy()  # copy this here for ease of retrieval later



train.id_code = train.id_code.apply(lambda x: f"../input/aptos2019-blindness-detection/train_images/{x}.png")

test.id_code = test.id_code.apply(lambda x: f"../input/aptos2019-blindness-detection/test_images/{x}.png")

class MyDatasetFolder(DatasetFolder):

    def __init__(

        self,

        samples,

        loader=default_loader,

        extensions=None,

        transform=None,

        target_transform=None,

        is_valid_file=None,

    ):

        self.transform = transform

        self.target_transform = target_transform

        classes, class_to_idx = self._find_classes()

        self.samples = samples

        if len(samples) == 0:

            raise (RuntimeError("Empty list of samples passed"))



        self.loader = loader

        self.extensions = extensions



        self.classes = classes

        self.class_to_idx = class_to_idx

        self.samples = samples

        self.targets = [s[1] for s in samples]



    def _find_classes(self):

        """ Hardcoded, as the folders aren't organised by class. """

        classes = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]

        class_to_idx = {

            "No DR": 0,

            "Mild": 1,

            "Moderate": 2,

            "Severe": 3,

            "Proliferative DR": 4,

        }

        return classes, class_to_idx
def train_model(

    model,

    dataloaders,

    criterion,

    optimizer,

    scheduler,

    device,

    dataset_sizes,

    num_epochs=25,

    patience=3,

):

    since = time.time()



    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0

    no_improvement = 0



    for epoch in range(num_epochs):

        print("Epoch {}/{}".format(epoch, num_epochs - 1))

        print("-" * 10)



        # Each epoch has a training and validation phase

        for phase in ["train", "val"]:

            if phase == "train":

                scheduler.step()  # dafuq is scheduler?

                model.train()  # Set model to training mode

            else:

                model.eval()  # Set model to evaluate mode



            running_loss = 0.0

            running_corrects = 0



            # Iterate over data.

            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(

                    device

                )  # device? what's that? I think it's just 'cpu'

                # I think it's more crucial when you've got a gpu

                labels = labels.to(device)



                # zero the parameter gradients

                optimizer.zero_grad()



                # forward

                # track history if only in train

                with torch.set_grad_enabled(phase == "train"):

                    outputs = model(inputs)

                    _, preds = torch.max(

                        outputs, 1

                    )  # just gives you the maximum in each row

                    loss = criterion(outputs, labels)



                    # backward + optimize only if in training phase

                    if phase == "train":

                        loss.backward()

                        optimizer.step()



                # statistics

                running_loss += loss.item() * inputs.size(0)  # loss.item()?

                running_corrects += torch.sum(preds == labels.data)



            epoch_loss = running_loss / dataset_sizes[phase]

            epoch_acc = running_corrects.double() / dataset_sizes[phase]



            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))



            # deep copy the model. ok, so there is a kind of early stopping going on here...

            if phase == "val":

                if epoch_acc > best_acc:

                    best_acc = epoch_acc

                    best_model_wts = copy.deepcopy(model.state_dict())

                    no_improvement = 0

                else:

                    no_improvement += 1

                    print(

                        f"No improvement for {no_improvement} round{'s'*int(no_improvement>1)}"

                    )

        if no_improvement > patience:

            break



        print()



    time_elapsed = time.time() - since

    print(

        "Training complete in {:.0f}m {:.0f}s".format(

            time_elapsed // 60, time_elapsed % 60

        )

    )

    print("Best val Acc: {:4f}".format(best_acc))



    # load best model weights

    model.load_state_dict(best_model_wts)

    return model
data_transforms = {

    "train": transforms.Compose(

        [

            transforms.RandomResizedCrop(224),

            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

        ]

    ),

    "val": transforms.Compose(

        [

            transforms.Resize(256),

            transforms.CenterCrop(224),

            transforms.ToTensor(),

            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

        ]

    ),

}



train_data, val_data = train_test_split(

    train, stratify=train.diagnosis, test_size=0.1

)



train_dataset = MyDatasetFolder(

    train_data.values.tolist(), transform=data_transforms["train"]

)

val_dataset = MyDatasetFolder(

    val_data.values.tolist(), transform=data_transforms["val"]

)

test_dataset = MyDatasetFolder(

    test.values.tolist(), transform=data_transforms["val"]

)

image_datasets = {"train": train_dataset, "val": val_dataset, "test": test_dataset}

dataloaders = {

    x: torch.utils.data.DataLoader(

        image_datasets[x], batch_size=4, shuffle=True, num_workers=4

    )

    for x in ["train", "val", "test"]

}

dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val", "test"]}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



model_conv = torchvision.models.resnet50()

model_conv.load_state_dict(torch.load('../input/pytorch-pretrained-models/resnet50-19c8e357.pth'))

for param in model_conv.parameters():

    param.requires_grad = (

        False

    )  # right. settings this to false so these parameters don't get retrained.



# Parameters of newly constructed modules have requires_grad=True by default

num_ftrs = model_conv.fc.in_features

model_conv.fc = nn.Linear(num_ftrs, 5)  # again, we're replacing the last layer.



model_conv = model_conv.to(device)



criterion = nn.CrossEntropyLoss()



# Observe that only parameters of final layer are being optimized as

# opposed to before.

optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# so...the only difference is that this time, all the other parameters are blocked? cool!



# Decay LR by a factor of 0.1 every 7 epochs

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)



model_conv = train_model(

    model_conv,

    dataloaders,

    criterion,

    optimizer_conv,

    exp_lr_scheduler,

    device,

    dataset_sizes,

    num_epochs=25,

    patience=10

)



sub = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv").set_index("id_code")



for (inputs, _spam) in dataloaders["test"]:

    inputs = inputs.to(device)

    outputs = model_conv(inputs)

    _, preds = torch.max(outputs, 1)



    preds = preds.to('cpu')

    sub.loc[list(_spam), "diagnosis"] = preds.numpy()



sub.reset_index().to_csv("submission.csv", index=False)