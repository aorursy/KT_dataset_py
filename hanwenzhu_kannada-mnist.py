import os



import numpy as np

import torch

from torch import nn

import torch.nn.functional as F

import torchvision

from torchvision import transforms

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.metrics import confusion_matrix

import seaborn as sns



plt.ion()
dataset_path = '/kaggle/input/Kannada-MNIST/'

output_path = '/kaggle/working/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using %r' % device)

batch_size = 1024

num_workers = 4

num_folds = 4

num_epochs = 50
csv_cache = {}

def read_csv(path):

    if path in csv_cache:

        return csv_cache[path]

    else:

        frame = pd.read_csv(path)

        csv_cache[path] = frame

        return frame



class MNIST(torch.utils.data.Dataset):



    def __init__(self, *paths, train=True, transform=None, split=None):

        self.train = train

        self.transform = transform

        values = pd.concat(read_csv(path) for path in paths).values

        if train:

            self.labels = values[:, 0].astype('int64')

            self.images = values[:, 1:].astype('float32').reshape(-1, 28, 28)

        else:

            # First column dropped since it's a 0 to 5000 index

            self.images = values.astype('float32')[:, 1:].reshape(-1, 28, 28)



        if split is not None:

            if train:

                self.labels = self.labels[split]

            self.images = self.images[split]



    def __len__(self):

        return self.images.shape[0]



    def __getitem__(self, key):

        if torch.is_tensor(key):

            key = tuple(key.tolist())

        if isinstance(key, tuple):

            raise NotImplementedError

        elif isinstance(key, slice):

            image = self.images[key, :, :].copy()

            if self.transform is not None:

                for i in range(image.shape[0]):

                    image[i] = self.transform(image[i])

        elif isinstance(key, int):

            image = self.images[key]

            if self.transform is not None:

                image = self.transform(image)



        if self.train:

            label = self.labels[key]

            return image, label

        else:

            return image
augmented_transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.RandomAffine(degrees=10, translate=(0.25, 0.25),

                            scale=(0.9, 1.1), shear=10,

                            fillcolor=0),

    transforms.ToTensor(),

    transforms.Normalize(mean=(128,), std=(128,)),

])

transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(mean=(128,), std=(128,)),

])



trainsets = []

trainloaders = []

for i in range(num_folds):

    trainset = MNIST(os.path.join(dataset_path, 'train.csv'), train=True,

                     transform=augmented_transform,

                     split=np.arange(i * 60000 // num_folds, (i + 1) * 60000 // num_folds))

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,

                                              shuffle=True, num_workers=num_workers,

                                              pin_memory=True)

    trainsets.append(trainset)

    trainloaders.append(trainloader)



testset = MNIST(os.path.join(dataset_path, 'test.csv'), train=False,

                transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,

                                         shuffle=False, num_workers=num_workers,

                                         pin_memory=True)



# Seems like a discrepency between the training set & Dig-MNIST

# Just checked the docs it's since Dig-MNIST is not sampled the same way

# So we'll use it as a harsh validation set

valset = MNIST(os.path.join(dataset_path, 'Dig-MNIST.csv'), train=True,

               transform=transform)

valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,

                                        shuffle=False, num_workers=num_workers,

                                        pin_memory=True)
dataiter = iter(trainloaders[0])

images, labels = dataiter.next()

images = images[:64, ...]

labels = labels[:64]

plt.figure(figsize=(20, 20))

plt.imshow(torchvision.utils.make_grid(images).numpy().transpose((1, 2, 0)) / 2 + 0.5,

           cmap='gray')

plt.axis('off')

plt.title(', '.join(str(label) for label in labels.tolist()));
class Model(nn.Module):

    

    def __init__(self):

        super(Model, self).__init__()

        self.conv1 = nn.Sequential(

#             nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),

#             nn.BatchNorm2d(64),

#             nn.ReLU(),

#             nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),

#             nn.BatchNorm2d(64),

#             nn.ReLU(),

#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Dropout(0.4),



#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),

#             nn.BatchNorm2d(64),

#             nn.ReLU(),

#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),

#             nn.BatchNorm2d(64),

#             nn.ReLU(),

#             nn.Dropout(0.4),



#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),

#             nn.BatchNorm2d(128),

#             nn.ReLU(),

#             nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),

#             nn.BatchNorm2d(128),

#             nn.ReLU(),

#             nn.MaxPool2d(kernel_size=2, stride=2),

#             nn.Dropout(0.4),



#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),

#             nn.BatchNorm2d(256),

#             nn.ReLU(),

#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),

#             nn.BatchNorm2d(256),

#             nn.ReLU(),

#             nn.Dropout(0.4),



            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),

            nn.LeakyReLU(0.1),

            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),

            nn.LeakyReLU(0.1),

            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),

            nn.LeakyReLU(0.1),

            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout(0.2),



            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),

            nn.LeakyReLU(0.1),

            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),

            nn.LeakyReLU(0.1),

            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),

            nn.LeakyReLU(0.1),

            nn.BatchNorm2d(128, eps=1e-5, momentum=0.1),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout(0.2),



            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),

            nn.LeakyReLU(0.1),

            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),

            nn.LeakyReLU(0.1),

            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Dropout(0.2),

        )

        self.dense = nn.Sequential(

#             nn.Linear(7 * 7 * 256, 256),

#             nn.BatchNorm1d(256),

#             nn.ReLU(),

#             nn.Dropout(0.4),

#             nn.Linear(256, 128),

#             nn.BatchNorm1d(128),

#             nn.ReLU(),

#             nn.Dropout(0.2),

#             nn.Linear(128, 10),



            nn.Linear(2304, 256),

            nn.LeakyReLU(0.1),

            nn.BatchNorm1d(256),

            nn.Linear(256, 10),

        )

    def forward(self, x):

        x = self.conv1(x)

        x = x.flatten(start_dim=1)

        x = self.dense(x)

        return x
models = []

for _ in range(num_folds):

    model = Model().to(device)

    print(model(iter(trainloader).next()[0].to(device)).argmax(1).tolist())

    models.append(model)

models
criterion = nn.CrossEntropyLoss()

optimizers = [torch.optim.RMSprop(model.parameters(), lr=0.002,

                                  alpha=0.9, momentum=0.1,

                                  eps=1e-7, centered=True)

              for model in models]

schedulers = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',

                                                         factor=0.25, patience=2,

                                                         verbose=True, min_lr=0.00001)

              for optimizer in optimizers]
histories = []

for k, (optimizer, model) in enumerate(zip(optimizers, models)):

    schedulers[k]

    histories.append([])

    if k > 0:

        print()

        print()

        print()

    print(f'Model fold {k + 1}')

    trainsets[k].transform = transform

    for epoch in range(num_epochs):

        sample = 0

        running_total = running_errors = running_loss = 0

        epoch_total = epoch_errors = epoch_loss = 0

        model.train()

        for i, trainloader in enumerate(trainloaders):

            if i == k:

                continue

            for images, labels in trainloader:

                optimizer.zero_grad()

                outputs = model(images.to(device))

                loss = criterion(outputs, labels.to(device))

                loss.backward()

                optimizer.step()



                total = labels.size(0)

                errors = (outputs.argmax(1).cpu() != labels).sum().item()

                loss = loss.item()

                running_total += total

                running_errors += errors

                running_loss += loss

                epoch_total += total

                epoch_errors += errors

                epoch_loss += loss



                if sample % 10 == 9:

                    print(f'\rEpoch {epoch + 1}, sample {(sample + 1) * batch_size:6d}, '

                          f'loss {running_loss / running_total:.5f}, '

                          f'acc {(1 - running_errors / running_total) * 100:3.2f}',

                          end='')

                    running_total = running_errors = running_loss = 0

                sample += 1



        model.eval()

        total = 0

        corrects = 0

        for i, (images, labels) in enumerate(trainloaders[k]):

#         for i, (images, labels) in enumerate(valloader):

            total += labels.size(0)

            corrects += (model(images.to(device)).argmax(1) == labels.to(device)).sum().cpu().item()

            print(f'\r{total} / {len(valset)} inferred', end='')

        print('\r', end='')

        print(f'Epoch {epoch + 1}                                         ')

        print(f'- Val acc: {100 * corrects / total:3.2f}%')

        print(f'- Train acc: {100 * (1 - epoch_errors / epoch_total):3.2f}%, '

              f'loss: {epoch_loss:.5f}')

        histories[k].append((corrects / total, 1 - epoch_errors / epoch_total))

        schedulers[k].step(epoch_loss)

    trainsets[k].transform = augmented_transform
for history in histories:

    plt.plot(range(1, len(history) + 1), list(zip(*history))[0], label='Validation accuracy')

    plt.plot(range(1, len(history) + 1), list(zip(*history))[1], label='Training accuracy')

    plt.legend()

    plt.ylim(None, 1)

    plt.show()
for model in models:

    model.eval()

total = 0

corrects = 0

predictions = []

for i, (images, labels) in enumerate(valloader):

    total += labels.size(0)

    prediction = sum(model(images.to(device)) for model in models).argmax(1)

    corrects += (prediction == labels.to(device)).sum().cpu().item()

    predictions.extend(prediction.tolist())

    print(f'\r{total} / {len(valset)} inferred', end='')

print('\r', end='')

print(f'{100 * corrects / total:.2f}%                  ')

conf = confusion_matrix(valset[:][1], predictions)

conf = pd.DataFrame(conf, index=range(0,10), columns=range(0,10))

plt.figure(figsize=(12,10))

sns.heatmap(conf, cmap="coolwarm", annot=True , fmt="d");
dataiter = iter(valloader)

images, labels = dataiter.next()

images = images[:8, :, :, :]

labels = labels[:8]

plt.imshow(torchvision.utils.make_grid(images).numpy().transpose((1, 2, 0)) / 2 + 0.5,

           cmap='gray')

plt.axis('off')

plt.title(', '.join(str(label) for label in labels.tolist()))

predictions = sum(model(images.to(device)) for model in models)

preds = ', '.join(str(prediction.item()) for prediction in predictions.argmax(1).cpu())

print(f'Predictions:  {preds}')

print(f'Ground truth: {", ".join(str(label) for label in labels.tolist())}')
for i, model in enumerate(models):

    torch.save(model.state_dict(), os.path.join(output_path, f'model-{i + 1}.pt'))
submission = []

for images in testloader:

    predictions = sum(model(images.to(device)) for model in models)

    submission.extend(predictions.argmax(1).tolist())

len(submission)
df = pd.DataFrame.from_records(np.array(submission).reshape(-1, 1))

df.to_csv(os.path.join(output_path, 'submission.csv'),

          index_label='id', header=['label'])