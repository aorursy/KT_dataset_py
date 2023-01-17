import os
from typing import Union, List, Optional
from collections import namedtuple

from tqdm.notebook import tqdm
os.listdir('../input/sleepedf-lite-0')
os.listdir('../input/isruc-lite/subgroup3')
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
Config = namedtuple('Config', ['seq_len', 'input_channels', 'hidden_channels', 'stride', 'sampling_rate',
                               'data_category', 'batch_size', 'pred_steps', 'feature_dim',
                               'learning_rate', 'epochs', 'save_path', 'finetune_lr',
                               'num_classes', 'finetune_ratio', 'finetune_epochs', 'selected_patient'])
args = Config(
    seq_len=6,
    stride=1,
    data_category='isruc',
    input_channels=6, # need change 6  2
    hidden_channels=16,
    sampling_rate=200,  # need change 200 100
    batch_size=32, # need change  32   128
    pred_steps=3,
    feature_dim=128,
    learning_rate=1e-3,
#     train_ratio=0.7,
    epochs=50,
    save_path='/kaggle/working/check_points/',
    num_classes=5,
    finetune_ratio=0.1,
    finetune_epochs=10,
    finetune_lr=1e-3,
    selected_patient=6
)
SLEEPEDF_SUBJECTS = ['SC4041E0.npz', 'SC4012E0.npz', 'SC4092E0.npz', 'SC4072E0.npz', 
                      'SC4051E0.npz', 'SC4071E0.npz', 'SC4022E0.npz', 'SC4062E0.npz', 
                      'SC4081E0.npz', 'SC4031E0.npz', 'SC4061E0.npz', 'SC4021E0.npz', 
                      'SC4011E0.npz', 'SC4001E0.npz', 'SC4082E0.npz', 'SC4032E0.npz', 
                      'SC4052E0.npz', 'SC4091E0.npz', 'SC4002E0.npz', 'SC4042E0.npz']
ISRUC_SUBJECTS = ['subject9.npz', 'subject8.npz', 'subject10.npz', 'subject6.npz', 
                      'subject2.npz', 'subject7.npz', 'subject3.npz', 'subject4.npz', 
                      'subject5.npz', 'subject1.npz']
if args.data_category == 'sleepedf':
    DATA_PATH = '../input/sleepedf-lite-0'
else:
    DATA_PATH = '../input/isruc-lite/subgroup3/'
def prepare_dataset(path, data_category, patients: List = None):
    assert os.path.exists(path)
    assert data_category in ['sleepedf', 'isruc']
    file_names = patients

    data_list = []
    target_list = []

    if isinstance(patients, list):
        candidate_files = list(map(lambda p: os.path.join(path, p), patients))
    else:
        raise ValueError('Invalid patients param!')

    for filename in candidate_files:
        tmp = np.load(filename)
        
        if data_category=='sleepedf':
            current_data = np.concatenate(
                    (tmp['eeg_fpz_cz'].reshape(-1, 1, tmp['eeg_fpz_cz'].shape[-1]),
                     tmp['eeg_pz_oz'].reshape(-1, 1, tmp['eeg_pz_oz'].shape[-1])),
                    axis=1)
            current_target = tmp['annotation']
        else:
            current_data = []
            for channel in ['F3_A2', 'C3_A2', 'F4_A1', 'C4_A1', 'O1_A2', 'O2_A1']:
                current_data.append(np.expand_dims(tmp[channel], 1))
            current_data = np.concatenate(current_data, axis=1)
            current_target = tmp['label']
        data_list.append(current_data)
        target_list.append(current_target)
        
    data_list = np.concatenate(data_list)
    target_list = np.concatenate(target_list).reshape(-1, 1)
 
    return data_list, target_list
class SleepDataset(Dataset):
    def __init__(self, x, y, return_label=False):
        self.return_label = return_label
        
        self.data = x
        self.targets = y
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, item):
        if self.return_label:
            return (
                torch.from_numpy(self.data[item].astype(np.float32)), 
                torch.from_numpy(self.targets[item].astype(np.long))
            )
        else:
            return torch.from_numpy(self.data[item].astype(np.float32))
        
    def __repr__(self):
        return f"""
               ****************************************
               Model  : {self.__class__.__name__}
               Length : {len(self)}
               ****************************************
                """
class ResidualBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_sizes=[7, 11, 7], stride=1, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride

        assert len(kernel_sizes) == 3

        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=kernel_sizes[0], stride=1,
                      padding=kernel_sizes[0] // 2, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True)
        )

        # Only conv2 degrades the scale
        self.conv2 = nn.Sequential(
            nn.Conv1d(output_channels, output_channels, kernel_size=kernel_sizes[1], stride=stride,
                      padding=kernel_sizes[1] // 2, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(output_channels, output_channels, kernel_size=kernel_sizes[2], stride=1,
                      padding=kernel_sizes[2] // 2, bias=False),
            nn.BatchNorm1d(output_channels),
        )

        self.relu = nn.ReLU(inplace=True)

        # If stride == 1, the length of the time dimension will not be changed
        # If input_channels == output_channels, the number of channels will not be changed
        # If the channels are mismatch, the conv1d is used to upgrade the channel
        # If the time dimensions are mismatch, the conv1d is used to downsample the scale
        self.downsample = nn.Sequential()
        if stride != 1 or input_channels != output_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(input_channels, output_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm1d(output_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        # Downsampe is an empty list if the size of inputs and outputs are same
        residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out
class ResNet(nn.Module):
    def __init__(self, input_channels, hidden_channels, feature_dim, kernel_sizes=[7, 11, 7]):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        # Residual layers
        self.layer1 = self.__make_layer(ResidualBlock, hidden_channels, hidden_channels, 2, kernel_sizes, stride=1)
        self.layer2 = self.__make_layer(ResidualBlock, hidden_channels, hidden_channels*2, 2, kernel_sizes, stride=2)
        self.layer3 = self.__make_layer(ResidualBlock, hidden_channels*2, hidden_channels*4, 2, kernel_sizes, stride=2)
        self.layer4 = self.__make_layer(ResidualBlock, hidden_channels*4, hidden_channels*8, 2, kernel_sizes, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool1d(1) # Pooling operation computes the average of the last dimension (time dimension)

        # A dense layer for output
        self.fc = nn.Linear(hidden_channels*8, feature_dim)

    def __make_layer(self, block, input_channels, output_channels, num_blocks, kernel_sizes, stride):
        layers = []
        layers.append(block(input_channels, output_channels, kernel_sizes, stride=stride))
        for i in range(1, num_blocks):
            layers.append(block(output_channels, output_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        L_out = floor[(L_in + 2*padding - kernel) / stride + 1]
        """
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avg_pool(out)
        out = out.view(x.size(0), -1)
        out = self.fc(out)

        return out
class ResidualBlockTransposed(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_sizes=[7, 11, 7], stride=1, dropout=0.2):
        super(ResidualBlockTransposed, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride

        assert len(kernel_sizes) == 3

        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(input_channels, output_channels, kernel_size=kernel_sizes[0], stride=1,
                      padding=kernel_sizes[0] // 2, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True)
        )

        # Only conv2 degrades the scale
        self.conv2 = nn.Sequential(
            nn.ConvTranspose1d(output_channels, output_channels, kernel_size=kernel_sizes[1], stride=stride,
                      padding=kernel_sizes[1] // 2, output_padding=stride-1 if stride>1 else 0, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.conv3 = nn.Sequential(
            nn.ConvTranspose1d(output_channels, output_channels, kernel_size=kernel_sizes[2], stride=1,
                      padding=kernel_sizes[2] // 2, bias=False),
            nn.BatchNorm1d(output_channels),
        )

        self.relu = nn.ReLU(inplace=True)

        # If stride == 1, the length of the time dimension will not be changed
        # If input_channels == output_channels, the number of channels will not be changed
        # If the channels are mismatch, the conv1d is used to upgrade the channel
        # If the time dimensions are mismatch, the conv1d is used to downsample the scale
        self.downsample = nn.Sequential()
        if stride != 1 or input_channels != output_channels:
            self.downsample = nn.Sequential(
                nn.ConvTranspose1d(input_channels, output_channels, kernel_size=1, stride=stride, output_padding=stride-1 if stride>1 else 0, padding=0, bias=False),
                nn.BatchNorm1d(output_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        # Downsampe is an empty list if the size of inputs and outputs are same
        residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out
class ResNetTransposed(nn.Module):
    def __init__(self, input_channels, hidden_channels, feature_dim, time_length, kernel_sizes=[7, 11, 7]):
        super(ResNetTransposed, self).__init__()
        
        final_length = time_length//8
        
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_channels*8*final_length),
        )
        
        self.hidden_channels = hidden_channels
        

        # Residual layers
        self.layer1 = self.__make_layer(ResidualBlockTransposed, hidden_channels*8, hidden_channels*4, 2, kernel_sizes, stride=1)
        self.layer2 = self.__make_layer(ResidualBlockTransposed, hidden_channels*4, hidden_channels*2, 2, kernel_sizes, stride=2)
        self.layer3 = self.__make_layer(ResidualBlockTransposed, hidden_channels*2, hidden_channels, 2, kernel_sizes, stride=2)
        self.layer4 = self.__make_layer(ResidualBlockTransposed, hidden_channels, hidden_channels, 2, kernel_sizes, stride=2)

#         self.avg_pool = nn.AdaptiveAvgPool1d(1) # Pooling operation computes the average of the last dimension (time dimension)

        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(hidden_channels, input_channels, kernel_size=1, padding=0, bias=False),
        )

    def __make_layer(self, block, input_channels, output_channels, num_blocks, kernel_sizes, stride):
        layers = []
        layers.append(block(input_channels, output_channels, kernel_sizes, stride=stride))
        for i in range(1, num_blocks):
            layers.append(block(output_channels, output_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        L_out = floor[(L_in + 2*padding - kernel) / stride + 1]
        """
        
        out = self.fc(x)
        out = out.view(out.size(0), self.hidden_channels*8, -1)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.conv1(out)

        return out
class AutoEncoder(nn.Module):
    def __init__(self, input_channels, hidden_channels, feature_dim, time_length, kernel_sizes=[7, 11, 7]):
        super(AutoEncoder, self).__init__()

        self.encoder = ResNet(input_channels, hidden_channels, feature_dim, kernel_sizes)

        self.decoder = ResNetTransposed(input_channels, hidden_channels, feature_dim, time_length, kernel_sizes)

    def forward(self, x):
        # print("x.shape", x.shape)
        z = self.encoder(x)
#         z = z.unsqueeze(1)
        # print("z.shape", z.shape)
        x_hat = self.decoder(z)
        return z, x_hat
if args.data_category == 'sleepedf':
    TRAIN_SUBJECTS = list(set(SLEEPEDF_SUBJECTS) - set([SLEEPEDF_SUBJECTS[args.selected_patient]]))
    TEST_SUBJECTS = [SLEEPEDF_SUBJECTS[args.selected_patient]]
elif args.data_category == 'isruc':
    TRAIN_SUBJECTS = list(set(ISRUC_SUBJECTS) - set([ISRUC_SUBJECTS[args.selected_patient]]))
    TEST_SUBJECTS = [ISRUC_SUBJECTS[args.selected_patient]]
else:
    raise ValueError
train_data, train_targets = prepare_dataset(DATA_PATH, data_category=args.data_category, patients=TRAIN_SUBJECTS)
test_data, test_targets = prepare_dataset(DATA_PATH, data_category=args.data_category, patients=TEST_SUBJECTS)
print(train_data.shape)
print(train_targets.shape)
print(test_data.shape)
print(test_targets.shape)
pretrain_dataset = SleepDataset(train_data, train_targets, return_label=True)
pretrain_loader = DataLoader(pretrain_dataset, batch_size=args.batch_size, 
                          drop_last=True, shuffle=True, pin_memory=True)
model = AutoEncoder(input_channels=args.input_channels, hidden_channels=args.hidden_channels, time_length=args.sampling_rate*30, feature_dim=args.feature_dim, kernel_sizes=[7,11,7])
model = model.cuda()
optimizer = optim.Adam(model.parameters(), 
                       lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-09, 
                       weight_decay=1e-4, amsgrad=True)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.MSELoss().cuda()
model.train()
for epoch in range(args.epochs):
    loss_list = []
    
    for x, y in tqdm(pretrain_loader, desc=f'EPOCH:[{epoch+1}/{args.epochs}]'):
        x, y = x.cuda(), y.cuda()
        
        optimizer.zero_grad()
        z, x_hat = model(x)
        loss = criterion(x_hat, x)
        
        loss.backward()
        optimizer.step()
        
#         acc_list.append(acc.item())
        loss_list.append(loss.item())
        
#         progress_bar.set_postfix({'loss': np.mean(loss_list)})
    
#     scheduler.step()
    print(f'Loss: {np.mean(loss_list)}')

    if (epoch+1) % 10 == 0:
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        torch.save(model.state_dict(), os.path.join(args.save_path, f'model_epoch_{epoch}.pth'))
if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)
torch.save(model.state_dict(), os.path.join(args.save_path, f'isruc_ae_final_{args.selected_patient}.pth'))
