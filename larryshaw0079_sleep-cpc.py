import os
import sys
import gc
import time
import warnings
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
data = np.load('../input/isruc-lite/subgroup3/subject1.npz')
print(data['F3_A2'].shape)
print(data['label'].shape)
Config = namedtuple('Config', ['seq_len', 'input_channels', 'hidden_channels', 'stride', 
                               'data_category', 'batch_size', 'pred_steps', 'feature_dim',
                               'learning_rate', 'epochs', 'save_path', 'finetune_lr',
                               'num_classes', 'finetune_ratio', 'finetune_epochs', 'rp',
                               'low_memory', 'selected_patient', 'load_path'])
args = Config(
    seq_len=10,
    stride=1,
    data_category='sleepedf',
    input_channels=2,
    hidden_channels=16,
    batch_size=64,
    pred_steps=5,
    feature_dim=128,
    learning_rate=1e-3,
#     train_ratio=0.7,
    epochs=50,
    save_path='/kaggle/working/check_points/',
    load_path='/kaggle/input/sleepdpcweights/SleepEDF_rp/',
    num_classes=5,
    finetune_ratio=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    finetune_epochs=10,
    finetune_lr=1e-3,
    rp=True,
    low_memory = True,
    selected_patient = 0
)
SLEEPEDF_SUBJECTS = ['SC4042E0.npz',
                     'SC4061E0.npz',
                     'SC4051E0.npz',
                     'SC4062E0.npz',
                     'SC4022E0.npz',
                     'SC4072E0.npz',
                     'SC4041E0.npz',
                     'SC4052E0.npz',
                     'SC4011E0.npz',
                     'SC4012E0.npz',
                     'SC4002E0.npz',
                     'SC4032E0.npz',
                     'SC4021E0.npz',
                     'SC4001E0.npz',
                     'SC4091E0.npz',
                     'SC4031E0.npz',
                     'SC4082E0.npz',
                     'SC4081E0.npz',
                     'SC4071E0.npz',
                     'SC4092E0.npz']

ISRUC_SUBJECTS = [f'subject{i}.npz' for i in range(1, 11)]
if args.data_category == 'sleepedf':
    DATA_PATH = '../input/sleepedf-lite-0'
else:
    DATA_PATH = '../input/isruc-lite/subgroup3/'
def prepare_pretraining_dataset(path, seq_len, data_category, patients: List = None):
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
                    
        for i in range(0, len(current_data), seq_len):
            if i+seq_len > len(current_data):
                break
            data_list.append(np.expand_dims(current_data[i:i+seq_len], axis=0))
            target_list.append(np.expand_dims(current_target[i:i+seq_len], axis=0))
        
    data_list = np.concatenate(data_list)
    target_list = np.concatenate(target_list)
 
    return data_list, target_list
def prepare_evaluation_dataset(path, seq_len, data_category, patients: List, sample_ratio=1.0):
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
        if sample_ratio == 1.0:
            for i in range(len(current_data)):
                if i+seq_len > len(current_data):
                    break
                data_list.append(np.expand_dims(current_data[i:i+seq_len], axis=0))
                target_list.append(np.expand_dims(current_target[i:i+seq_len], axis=0))
        else:
            idx = np.arange(seq_len-1, len(current_data))
            selected_idx = np.random.choice(idx, size=int(len(idx)*sample_ratio), replace=False)
            for i in selected_idx:
                data_list.append(np.expand_dims(current_data[i-seq_len+1:i+1], axis=0))
                target_list.append(np.expand_dims(current_target[i-seq_len+1:i+1], axis=0))
        
    data_list = np.concatenate(data_list)
    target_list = np.concatenate(target_list)
 
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
                      padding=kernel_sizes[0]//2, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True)
        )

        # Only conv2 degrades the scale
        self.conv2 = nn.Sequential(
            nn.Conv1d(output_channels, output_channels, kernel_size=kernel_sizes[1], stride=stride, 
                      padding=kernel_sizes[1]//2, bias=False),
            nn.BatchNorm1d(output_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(output_channels, output_channels, kernel_size=kernel_sizes[2], stride=1, 
                      padding=kernel_sizes[2]//2, bias=False),
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
    def __init__(self, input_channels, hidden_channels, num_classes, kernel_sizes=[7, 11, 7]):
        super(ResNet, self).__init__()

        # The first convolution layer
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(input_channels, hidden_channels, kernel_size=15, stride=2, padding=7, bias=False),
#             nn.BatchNorm1d(hidden_channels),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
#         )
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
        self.fc = nn.Linear(hidden_channels*8, num_classes)

        # Initialize weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 n = m.kernel_size[0] * m.kernel_size[0] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm1d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

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
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.3):
        super(GRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                          batch_first=True, dropout=dropout)
        
    def forward(self, x, h_0):
        # x:   (batch, seq_len,    input_size)
        # h_0: (num_layers, batch, hidden_size)
        
        out, h_n = self.gru(x, h_0)
        
        # out: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        return out, h_n
        
    
    def init_hidden(self, batch_size):
        return torch.randn(self.num_layers, batch_size, self.hidden_size).cuda()
class StatePredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(StatePredictor, self).__init__()
        
        self.pred = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x):
        return self.pred(x)
def compute_targets(batch_size, pred_steps, seq_len):
    targets = torch.zeros(batch_size, pred_steps, seq_len, batch_size).long()
    for i in range(batch_size):
        for j in range(pred_steps):
            targets[i, j, seq_len-pred_steps+j, i] = 1
            
    targets = targets.cuda()
    targets = targets.view(batch_size*pred_steps, seq_len*batch_size)
    targets = targets.argmax(dim=1)
    
    return targets

def compute_position_targets(batch_size, seq_len):
    targets = torch.zeros(batch_size, seq_len, seq_len, batch_size).long()
    for i in range(batch_size):
        for j in range(seq_len):
            for k in range(seq_len):
                targets[i, j, k, i] = 1
            
    targets = targets.cuda()
    targets = targets.view(batch_size*seq_len, seq_len*batch_size)
    targets = targets.argmax(dim=1)
    
    return targets
class SleepContrast(nn.Module):
    def __init__(self, input_channels, hidden_channels, feature_dim, pred_steps, num_seq, batch_size, kernel_sizes, rp=True):
        super(SleepContrast, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.feature_dim = feature_dim
        self.pred_steps = pred_steps
        self.batch_size = batch_size
        self.kernel_sizes = kernel_sizes
        self.num_seq = num_seq
        self.rp = rp
        
        self.targets = None
        self.position_targets = None
        
        # Local Encoder
        self.encoder = ResNet(input_channels, hidden_channels, feature_dim, kernel_sizes=kernel_sizes)
        
        # Aggregator
        self.gru = GRU(input_size=feature_dim, hidden_size=feature_dim, num_layers=2)
        
        # Predictor
        self.predictor = StatePredictor(input_dim=feature_dim, output_dim=feature_dim)
        
#     def _initialize_weights(self, module):
#         for name, param in module.named_parameters():
#             if 'bias' in name:
#                 nn.init.constant_(param, 0.0)
#             elif 'weight' in name:
#                 nn.init.orthogonal_(param, 0.1)
        
    def forward(self, x):
        # Extract feautres
        # x: (batch, num_seq, channel, seq_len)
        (batch, num_seq, channel, seq_len) = x.shape
        x = x.view(batch*num_seq, channel, seq_len)
        feature = self.encoder(x)
        feature = feature.view(batch, num_seq, self.feature_dim) # (batch, num_seq, feature_dim)
        feature_trans = feature.transpose(0, 2).contiguous()
        
        if self.rp:
            position_score = torch.einsum('ijk,kmn->ijmn', [feature, feature_trans])
            position_score = position_score.view(batch*num_seq, num_seq*batch)
        
        # Get context feature
        h_0 = self.gru.init_hidden(self.batch_size)
        # out: (batch, num_seq, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        out, h_n = self.gru(feature[:, :-self.pred_steps,:], h_0)
        
        # Get predictions
        pred = []
        h_next = h_n
        c_next = out[:,-1,:].squeeze(1)
        for i in range(self.pred_steps):
            z_pred = self.predictor(c_next)
            pred.append(z_pred)
            c_next, h_next = self.gru(z_pred.unsqueeze(1), h_next)
            c_next = c_next[:,-1,:].squeeze(1)
        pred = torch.stack(pred, 1) # (batch, pred_step, feature_dim)
        
        # Compute scores
#         feature = feature.transpose(0, 2).contiguous() # (feature_dim, num_seq, batch)
        pred = pred.contiguous()
        
        score = torch.einsum('ijk,kmn->ijmn', [pred, feature_trans]) # (batch, pred_step, num_seq, batch)
        score = score.view(batch*self.pred_steps, num_seq*batch)
        
        if self.rp:
            return score, position_score
        else:
            return score
if not os.path.exists(args.save_path):
    warnings.warn(f'The path {args.save_path} dose not existed, created.')
    os.makedirs(args.save_path)
if args.data_category == 'sleepedf':
    all_subjects = SLEEPEDF_SUBJECTS
else:
    all_subjects = ISRUC_SUBJECTS

if args.data_category == 'sleepedf':
    DATA_PATH = '/kaggle/input/sleepedf-lite-0'
else:
    DATA_PATH = '/kaggle/input/isruc-lite/subgroup3'
train_subjects = list(set(all_subjects) - {all_subjects[args.selected_patient]})
test_subjects = [all_subjects[args.selected_patient]]

print('Train subjects:', train_subjects)
print('Test subjects:', test_subjects)
# pretrain_data, pretrain_targets = prepare_pretraining_dataset(DATA_PATH, data_category=args.data_category,
#                                                               seq_len=args.seq_len, patients=train_subjects)
# pretrain_dataset = SleepDataset(pretrain_data, pretrain_targets, return_label=True)
# pretrain_loader = DataLoader(pretrain_dataset, batch_size=args.batch_size, 
#                           drop_last=True, shuffle=True, pin_memory=True)
# model = SleepContrast(input_channels=args.input_channels, hidden_channels=args.hidden_channels, 
#                       feature_dim=args.feature_dim, pred_steps=args.pred_steps, 
#                       batch_size=args.batch_size, num_seq=args.seq_len, kernel_sizes=[7, 11, 7],
#                       rp=args.rp)
# model = model.cuda()
# optimizer = optim.Adam(model.parameters(), 
#                        lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-09, 
#                        weight_decay=1e-4, amsgrad=True)
# criterion = nn.CrossEntropyLoss()
# targets = compute_targets(args.batch_size, args.pred_steps, args.seq_len)
# if args.rp:
#     position_targets = compute_position_targets(args.batch_size, args.seq_len)
# model.train()
# for epoch in range(args.epochs):
#     acc_list = []
#     loss_list = []
    
#     for x, y in tqdm(pretrain_loader, desc=f'EPOCH:[{epoch+1}/{args.epochs}]'):
#         x, y = x.cuda(), y.cuda()
        
#         optimizer.zero_grad()

#         if args.rp:
#             score, position_score = model(x)
#             loss = criterion(score, targets) + criterion(position_score, position_targets)
#         else:
#             score = model(x)
#             loss = criterion(score, targets)
        
#         loss.backward()
#         optimizer.step()
        
#         loss_list.append(loss.item())
        
#     print(f'Loss: {np.mean(loss_list)}')

#     if (epoch+1) % 10 == 0:
#         if not os.path.exists(args.save_path):
#             os.mkdir(args.save_path)
#         torch.save(model.state_dict(), os.path.join(args.save_path, f'model_epoch_{epoch}.pth'))
# if args.low_memory:
#     torch.save(model.state_dict(), os.path.join(args.save_path, f'encoder_{args.selected_patient}_final.pth'))
#     del model
#     del pretrain_data, pretrain_targets, pretrain_dataset, pretrain_loader
class SleepClassifier(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_classes, feature_dim, pred_steps, num_seq, batch_size, kernel_sizes):
        super(SleepClassifier, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.pred_steps = pred_steps
        self.num_seq = num_seq
        self.batch_size = batch_size
        self.kernel_sizes = kernel_sizes
        
        
        # Local Encoder
        self.encoder = ResNet(input_channels, hidden_channels, feature_dim, kernel_sizes=kernel_sizes)
        
        # Aggregator
        self.gru = GRU(input_size=feature_dim, hidden_size=feature_dim, num_layers=2)
        
        # Classifier
        self.mlp = nn.Sequential(
            nn.ReLU(inplace=True),
#             nn.Linear(feature_dim, feature_dim),
#             nn.BatchNorm1d(feature_dim),
#             nn.ReLU(inplace=True),
            nn.Linear(feature_dim, num_classes)
        )
        
    def freeze_parameters(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.gru.parameters():
            p.requires_grad = False
        
    def forward(self, x):
        # Extract feautres
        # x: (batch, num_seq, channel, seq_len)
        (batch, num_seq, channel, seq_len) = x.shape
        x = x.view(batch*num_seq, channel, seq_len)
        feature = self.encoder(x)
        feature = feature.view(batch, num_seq, self.feature_dim) # (batch, num_seq, feature_dim)
        
        # Get context feature
        h_0 = self.gru.init_hidden(self.batch_size)
        # context: (batch, num_seq, hidden_size)
        # h_n:     (num_layers, batch, hidden_size)
        context, h_n = self.gru(feature[:, :-self.pred_steps,:], h_0)
        
        context = context[:, -1, :]
#         out = self.relu(context)
        out = self.mlp(context)
        
        return out
if args.data_category == 'sleepedf':
    all_subjects = SLEEPEDF_SUBJECTS
else:
    all_subjects = ISRUC_SUBJECTS

if args.data_category == 'sleepedf':
    DATA_PATH = '/kaggle/input/sleepedf-lite-0'
else:
    DATA_PATH = '/kaggle/input/isruc-lite/subgroup3'
for patient in range(len(os.listdir(DATA_PATH))):
    print(f'********************** Running Subject {patient} **********************')
    
    train_subjects = list(set(all_subjects) - {all_subjects[patient]})
    test_subjects = [all_subjects[patient]]
    
    print('Train subjects:', train_subjects)
    print('Test subjects:', test_subjects)
    
    results = {}
    
    st = time.time()
    for ratio in args.finetune_ratio:
        print(f'Test finetune ratio {ratio}...')
        classifier = SleepClassifier(input_channels=args.input_channels, hidden_channels=args.hidden_channels, 
                                 num_classes=args.num_classes, feature_dim=args.feature_dim, 
                                 pred_steps=args.pred_steps, batch_size=args.batch_size, 
                                 num_seq=args.seq_len, kernel_sizes=[7, 11, 7])
        classifier = classifier.cuda()
        
        classifier.load_state_dict(
            torch.load(
                os.path.join(args.load_path, f'encoder_{patient}_final.pth')
            ), strict=False
        )
        
        classifier.freeze_parameters()
        
        finetune_data, finetune_targets = prepare_evaluation_dataset(DATA_PATH,
                                                                     data_category=args.data_category,
                                                                     seq_len=args.seq_len,
                                                                     patients=train_subjects,
                                                                     sample_ratio=ratio)
        finetune_dataset = SleepDataset(finetune_data, finetune_targets, return_label=True)
        finetune_loader = DataLoader(finetune_dataset, batch_size=args.batch_size,
                                         drop_last=True, shuffle=True, pin_memory=True)
        
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), 
                               lr=args.finetune_lr, betas=(0.9, 0.98), eps=1e-09, 
                               weight_decay=1e-4, amsgrad=True)
        criterion = nn.CrossEntropyLoss()
        
        classifier.train()
    
        for epoch in range(args.finetune_epochs):
            losses = []
            for x, y in finetune_loader:
                x, y = x.cuda(), y.cuda()
                    
                optimizer.zero_grad()
                y_hat = classifier(x)
                loss = criterion(y_hat, y[:,-args.pred_steps-1])
                    
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
            print(f'EPOCH:[{epoch+1}/{args.finetune_epochs}]', f'Loss: {np.mean(losses)}')
            
        del finetune_data, finetune_targets, finetune_dataset, finetune_loader
        gc.collect()
            
        test_data, test_targets = prepare_evaluation_dataset(DATA_PATH,
                                                             data_category=args.data_category,
                                                             seq_len=args.seq_len,
                                                             patients=test_subjects,
                                                             sample_ratio=1.0)
        test_dataset = SleepDataset(test_data, test_targets, return_label=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 drop_last=True, shuffle=True, pin_memory=True)
        
        classifier.eval()
    
        predictions = []
        labels = []
        for x, y in test_loader:
            x = x.cuda()
            
            with torch.no_grad():
                y_hat = classifier(x)
                
            labels.append(y.numpy()[:,-args.pred_steps-1])
            predictions.append(y_hat.cpu().numpy())
            
        labels = np.concatenate(labels, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        predictions = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(labels, predictions)
        f1_micro = f1_score(labels, predictions, average='micro')
        f1_macro = f1_score(labels, predictions, average='macro')
        
        results[ratio] = {'accuracy': accuracy, 'f1_micro': f1_micro, 'f1_macro': f1_macro}
    
        del test_data, test_targets, test_dataset, test_loader
        del classifier
        gc.collect()
        
    print(results)
    with open(os.path.join(args.save_path, f'results_{args.data_category}_{patient}.pkl'), 'wb') as f:
        pickle.dump(results, f)
        
    ed = time.time()
    print(f'Time elapsed: {ed-st} s')
