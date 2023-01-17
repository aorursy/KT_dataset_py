import pandas as pd

import numpy as np

import torch

import torch.nn.functional as F

import torchvision



from tqdm import tqdm

from torch.utils.data import Dataset

from albumentations import Compose

from albumentations.pytorch import ToTensorV2



INPUT_PATH = '/kaggle/input/bengaliai-cv19'
# ======================

# Params

BATCH_SIZE = 32

N_WORKERS = 4

N_EPOCHS = 5



HEIGHT = 137

WIDTH = 236

TARGET_SIZE = 256



# My weights dataset for this compeititon; feel free to vote the dataste ;)

# https://www.kaggle.com/pestipeti/bengali-ai-model-weights

WEIGHTS_FILE = '/kaggle/input/bengali-ai-model-weights/baseline_weights.pth'
def make_square(img, target_size=256):

    img = img[0:-1, :]

    height, width = img.shape



    x = target_size

    y = target_size



    square = np.ones((x, y), np.uint8) * 255

    square[(y - height) // 2:y - (y - height) // 2, (x - width) // 2:x - (x - width) // 2] = img



    return square



class BengaliParquetDataset(Dataset):



    def __init__(self, parquet_file, transform=None):



        self.data = pd.read_parquet(parquet_file)

        self.transform = transform



    def __len__(self):

        return len(self.data)



    def __getitem__(self, idx):

        tmp = self.data.iloc[idx, 1:].values.reshape(HEIGHT, WIDTH)

        img = np.zeros((TARGET_SIZE, TARGET_SIZE, 3))

        img[..., 0] = make_square(tmp, target_size=TARGET_SIZE)

        img[..., 1] = img[..., 0]

        img[..., 2] = img[..., 0]



        image_id = self.data.iloc[idx, 0]



        if self.transform:

            transformed = self.transform(image=img)

            img = transformed['image']



        return {

            'image_id': image_id,

            'image': img

        }



class BengaliModel(torch.nn.Module):



    def __init__(self):

        super().__init__()

        self.backbone = torchvision.models.resnet18(pretrained=False)



        in_features = self.backbone.fc.in_features



        self.fc_graph = torch.nn.Linear(in_features, 168)

        self.fc_vowel = torch.nn.Linear(in_features, 11)

        self.fc_conso = torch.nn.Linear(in_features, 7)



    def forward(self, x):



        x = self.backbone.conv1(x)

        x = self.backbone.bn1(x)

        x = self.backbone.relu(x)

        x = self.backbone.maxpool(x)



        x = self.backbone.layer1(x)

        x = self.backbone.layer2(x)

        x = self.backbone.layer3(x)

        x = self.backbone.layer4(x)



        x = self.backbone.avgpool(x)

        x = torch.flatten(x, 1)



        fc_graph = self.fc_graph(x)

        fc_vowel = self.fc_vowel(x)

        fc_conso = self.fc_conso(x)



        return fc_graph, fc_vowel, fc_conso

test_df = pd.read_csv(INPUT_PATH + '/test.csv')

submission_df = pd.read_csv(INPUT_PATH + '/sample_submission.csv')



transform_test = Compose([

    ToTensorV2()

])



device = torch.device("cuda:0")

model = BengaliModel()

model.load_state_dict(torch.load(WEIGHTS_FILE))

model.to(device)



results = []
for i in range(4):

    parq = INPUT_PATH + '/test_image_data_{}.parquet'.format(i)

    test_dataset = BengaliParquetDataset(

        parquet_file=parq,

        transform=transform_test

    )

    data_loader_test = torch.utils.data.DataLoader(

        test_dataset,

        batch_size=BATCH_SIZE,

        num_workers=N_WORKERS,

        shuffle=False

    )



    print('Parquet {}'.format(i))



    model.eval()



    tk0 = tqdm(data_loader_test, desc="Iteration")



    for step, batch in enumerate(tk0):

        inputs = batch["image"]

        image_ids = batch["image_id"]

        inputs = inputs.to(device, dtype=torch.float)



        out_graph, out_vowel, out_conso = model(inputs)

        out_graph = F.softmax(out_graph, dim=1).data.cpu().numpy().argmax(axis=1)

        out_vowel = F.softmax(out_vowel, dim=1).data.cpu().numpy().argmax(axis=1)

        out_conso = F.softmax(out_conso, dim=1).data.cpu().numpy().argmax(axis=1)



        for idx, image_id in enumerate(image_ids):

            results.append(out_conso[idx])

            results.append(out_graph[idx])

            results.append(out_vowel[idx])
submission_df['target'] = results

submission_df.to_csv('./submission.csv', index=False)