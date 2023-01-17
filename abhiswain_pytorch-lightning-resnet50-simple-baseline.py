!pip install pytorch-lightning > /dev/null
# %% [code]

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import torch
import torchvision

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from PIL import Image
import os
from tqdm import tqdm

BASE_DIR = '../input/plant-pathology-2020-fgvc7/'

class config:
    BATCH_SIZE = 32

class PlantNet(torch.nn.Module):
    def __init__(self, output_features, pretrained):
        super(PlantNet, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=pretrained)
        in_features = self.model.fc.in_features

        self.model.fc = torch.nn.Linear(in_features, output_features)

    def forward(self, x):
        return self.model(x)


class PlantDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __getitem__(self, item):
        image_ids = self.df['image_id'].values
        labels = self.df[self.df.columns[1:]].values

        image = Image.open(BASE_DIR + 'images/' + image_ids[item] + '.jpg')
        label = torch.argmax(torch.tensor(labels[item]))

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.df)


class TestDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __getitem__(self, item):
        image_ids = self.df['image_id'].values

        image = Image.open(BASE_DIR + 'images/' + image_ids[item] + '.jpg')

        if self.transform:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.df)

# Lightning module!
class PlantLightning(pl.LightningModule):

    def __init__(self, csv_file, pretrained):
        super(PlantLightning, self).__init__()
        self.model = PlantNet(4, pretrained=pretrained)
        self.csv_file = csv_file

        self.best_loss = 10

    def prepare_data(self, valid_size=0.2, random_seed=42, shuffle=True):
        """Can be done in __init__() method also"""
        transforms = {
            'train': torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.Resize((256, 256)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]
            ),
            'valid': torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.Resize((256, 256)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]
            ),
        }

        self.train_dataset = PlantDataset(csv_file=self.csv_file, transform=transforms['train'])
        self.valid_dataset = PlantDataset(csv_file=self.csv_file, transform=transforms['valid'])

        num_train = len(self.train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        self.valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

    def train_dataloader(self):
        """REQUIRED"""
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=config.BATCH_SIZE, sampler=self.train_sampler,
            num_workers=10
        )

    def val_dataloader(self):
        """REQUIRED"""
        return torch.utils.data.DataLoader(
            self.valid_dataset, batch_size=config.BATCH_SIZE, sampler=self.valid_sampler,
            num_workers=10
        )

    def forward(self, x):
        """REQUIRED"""
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        """REQUIRED"""
        images, labels = batch

        preds = self.forward(images)

        loss = torch.nn.functional.cross_entropy(preds, labels)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        """REQUIRED"""
        images, labels = batch

        preds = self.forward(images)

        loss = torch.nn.functional.cross_entropy(preds, labels)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        """(OPTIONAL) Tocompute statistics"""
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        print(f'Validation Loss: {avg_loss}')

        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            torch.save({'best_loss': avg_loss, 'model': self.model, 'model_state_dict': self.model.state_dict()},
                       'best_model.pt')

        return {'val_loss': avg_loss}


if __name__ == "__main__":
    
    model = PlantLightning(BASE_DIR + 'train.csv', pretrained=True)
    early_stopping = EarlyStopping('val_loss', patience=3) #Early-stopping

    trainer = pl.Trainer(gpus=[0], max_nb_epochs=10,
                         early_stop_callback=early_stopping)
    
    #Start training! simpe as that!
    trainer.fit(model)

    test_df = pd.read_csv(BASE_DIR +'test.csv')

    print('Loading pre-trained model')
    model = PlantNet(4, False)
    model_ckpt = torch.load('best_model.pt')
    print(model.load_state_dict(model_ckpt['model_state_dict']))

    print('Testing!')

    test_dataset = TestDataset(BASE_DIR + 'test.csv', transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize((256, 256)),
                    torchvision.transforms.ToTensor()
                ]
            )
    )

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)

    predictions = np.zeros((1, 4))
    with torch.no_grad():
        for images in tqdm(test_dataloader):

            images = images.to('cuda')
            model = model.to('cuda')

            preds = torch.nn.functional.softmax(model(images), 1)

            predictions = np.concatenate((predictions, preds.cpu().detach().numpy()), 0)

    output = pd.DataFrame(predictions, columns=['healthy', 'multiple_diseases', 'rust', 'scab'])
    output.drop(0, inplace=True)
    output.reset_index(drop=True, inplace=True)
    output['image_id'] = test_df.image_id
    output = output[['image_id', 'healthy', 'multiple_diseases', 'rust', 'scab']]

    output.to_csv('submission.csv', index=False)

#     print(predictions)


