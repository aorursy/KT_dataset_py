import pandas as pd



import time

import torchvision

import torch.nn as nn

from tqdm import tqdm_notebook as tqdm



from PIL import Image, ImageFile

from torch.utils.data import Dataset

import torch

import torch.optim as optim

from torchvision import transforms

from torch.optim import lr_scheduler

import os

from torch.autograd import Variable

from PIL import ImageOps 

device = torch.device("cuda:0")

# ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.cuda.get_device_name()
class RetinopathyDataset(Dataset):



    def __init__(self, csv_file, transform, test=False):

        self.test_ = test

        self.transform = transform

        self.data = pd.read_csv(csv_file)

        self.data_test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        if not self.test_:

            img_name = os.path.join('../input/aptos2019-blindness-detection/train_images', self.data.loc[idx, 'id_code'] + '.png')

            image = Image.open(img_name)

            image = image.resize((350, 350), resample=Image.BILINEAR)   

            image = ImageOps.equalize(image)

            label = torch.tensor(self.data.loc[idx, 'diagnosis'])

            return {'image': self.transform(image),

                    'labels': label

                    }

        else:

            img_name = os.path.join('../input/aptos2019-blindness-detection/test_images', self.data_test.loc[idx, 'id_code'] + '.png')

            image = Image.open(img_name)

            image = image.resize((350, 350), resample=Image.BILINEAR)

            image = ImageOps.equalize(image)

            return  self.transform(image)
model = torchvision.models.resnet101(pretrained=False)

model.load_state_dict(torch.load("../input/pytorch-pretrained-models/resnet101-5d3b4d8f.pth"))

num_features = model.fc.in_features

model.fc = nn.Sequential(

                          nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          nn.Dropout(p=0.25),

                          nn.Linear(in_features=2048, out_features=2048, bias=True),

                          nn.ReLU(),

                          nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

                          nn.Dropout(p=0.5),

                          nn.Linear(in_features=2048, out_features=5, bias=True),

                         )



model = model.to(device)
model.eval()
for name, param in model.named_parameters():

    print(name, param.requires_grad)
train_transform = transforms.Compose([

    transforms.RandomChoice(

        [

            transforms.RandomHorizontalFlip(),

            transforms.RandomVerticalFlip(),

            transforms.RandomAffine(20),

        ]

    ),

    transforms.ToTensor(),

    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

test_dataset = RetinopathyDataset(csv_file='../input/aptos2019-blindness-detection/sample_submission.csv',

                                      transform=train_transform)
train_dataset = RetinopathyDataset(csv_file='../input/aptos2019-blindness-detection/train.csv', transform=train_transform)

data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=24, shuffle=True)



optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

scheduler = lr_scheduler.StepLR(optimizer, step_size=10)
int(len(data_loader))
def accuracy_score(output, labels):

    score = 0

    for c, cc in zip(output, labels):

        if c == cc:

            score += 1

    return score/len(output)
def train():

    since = time.time()

    criterion = nn.criterion = nn.CrossEntropyLoss()

    num_epochs = 50

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        print('-' * 10)

        scheduler.step()

        model.train()

        running_loss = 0.0

        tk0 = tqdm(data_loader, total=int(len(data_loader)))

        counter = 0

        for bi, d in enumerate(tk0):

            inputs = d["image"]

            labels = d["labels"].view(-1, 1)

            inputs = inputs.to(device)

            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                outputs = model(inputs)

                loss = criterion(outputs, labels.squeeze_())

                loss.backward()

                optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            counter += 1

            tk0.set_postfix(loss=(running_loss / (counter * data_loader.batch_size)))

        epoch_loss = running_loss / len(data_loader)

        

        print('Training Loss: {:.4f}'.format(epoch_loss))

        print('Epoch Accuracy: ', accuracy_score(outputs.argmax(1), labels))

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    torch.save(model.state_dict(), "model.bin")
train()
test_dataset = RetinopathyDataset(csv_file='../input/aptos2019-blindness-detection/test.csv', transform=train_transform, test=True)

test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
result = []



tk0 = tqdm(test_data_loader, total=int(len(test_data_loader)))

for d in enumerate(tk0):

    inputs = Variable(d[1])

    inputs = inputs.to(device, dtype=torch.float)

    with torch.set_grad_enabled(False):

        outputs = model(inputs)

        for pred in outputs.argmax(1).tolist():

            result.append(pred)
result[:5]
sample = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
sample.diagnosis = result
sample.to_csv("submission.csv", index=False)