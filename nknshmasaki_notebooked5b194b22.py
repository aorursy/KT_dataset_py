import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

train = pd.read_csv(r"/kaggle/input/digit-recognizer/train.csv")
class image_data_set(Dataset):
    def __init__(self, df, flg="train"):
        self.flg = flg
        if self.flg == "train":
            self.images = df.iloc[:,1:].values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]
            self.labels = torch.from_numpy(df.iloc[:,0].values)
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomAffine(
                    degrees=45,
                    translate=(0.1, 0.1),
                    scale=(0.8, 1.2),
                    fillcolor=0
                ),
                transforms.ToTensor(),
            ])
        else:
            self.images = df.values.reshape((-1,28,28)).astype(np.uint8)[:,:,:,None]
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.flg == "train":
            return self.transform(self.images[idx]), self.labels[idx]
        else:
            return self.transform(self.images[idx])
train_set = image_data_set(train)
train_loader = DataLoader(train_set, batch_size = 100, shuffle = False)
def conv_layer(
    channel_in,
    channel_out,
    kernel_size=3,
    stride=1,
    padding=0,
    activation=nn.ReLU(True),
    use_norm=True
):
    sequence = []
    sequence += [nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, padding=padding, stride=stride, padding_mode="replicate", bias=False)]
    sequence += [activation]
    sequence += [nn.BatchNorm2d(channel_out)]

    return nn.Sequential(*sequence)

# モデルの定義
class ImprovedLeNet5(nn.Module):
    def __init__(self):
        super(ImprovedLeNet5, self).__init__()
        
        # conv block
        sequence = []
        sequence += [conv_layer(1, 32)]
        sequence += [conv_layer(32, 32)]
        sequence += [conv_layer(32, 32, kernel_size=5, padding=2, stride=2)]
        sequence += [nn.Dropout(0.4)]
        sequence += [conv_layer(32, 64)]
        sequence += [conv_layer(64, 64)]
        sequence += [conv_layer(64, 64, kernel_size=5, padding=2, stride=2)]
        sequence += [nn.Dropout(0.4)]
        sequence += [conv_layer(64, 128, kernel_size=4)]
        self.conv_block = nn.Sequential(*sequence)
        
        # dense block
        sequence = []
        sequence += [nn.Linear(128, 10)]
        sequence += [nn.Dropout(0.1)]
        sequence += [nn.Softmax(dim=1)]
        self.dense_block = nn.Sequential(*sequence)
        
    def forward(self, input_image):
        feature_map = self.conv_block(input_image)
        output = self.dense_block(feature_map.flatten(1))
        return output
device = torch.device("cuda")
model = ImprovedLeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()
for epoch in range(45):
    train_loss = 0
    correct = 0
    for images, labels in train_loader:
        #GPUの利用
        images = images.to(device)
        labels = labels.to(device)

        # 勾配の初期化
        optimizer.zero_grad()

        # 順伝播
        outputs = model(images)
        
        # ロスの計算と逆伝播
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()

        # 正答数を計算
        predicted = outputs.data.max(1, keepdim=True)[1] 
        correct += predicted.eq(labels.data.view_as(predicted)).cpu().sum()
        
        train_loss += loss.item()
        
    print(f"Epoch: {epoch+1}/{45}.. ",
          f"Training Loss: {train_loss/len(train_loader):.3f}.. ",
          f"Training Accu: {correct/len(train_loader):.3f}.. ")
test = pd.read_csv(r"/kaggle/input/digit-recognizer/test.csv")
test_set = image_data_set(test, flg='test')
test_loader = DataLoader(test_set, batch_size = 100, shuffle = False)

model.eval()
test_predicted = torch.LongTensor()
for i, data in enumerate(test_loader):
    data = data.cuda()
    output = model(data)
    predicted = output.cpu().data.max(1, keepdim=True)[1]
    test_predicted = torch.cat((test_predicted, predicted), dim=0)
submission_df = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
submission_df['Label'] = test_predicted.numpy().squeeze()
submission_df.to_csv("my_submission.csv", index=False, header=True)