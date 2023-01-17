import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
class Mydataset(Dataset):
    
    def __init__(self, dir_list):
        self.data = []
        self.label = []
        self.dir_list = []
        self.transform = transform
        for d in dir_list:
            self.data_path = '../input/data/Data/' + d + '/image/'
            self.label_path = '../input/data/Data/' + d + '/label/'
            for i in os.listdir(self.data_path):
                self.data.append(cv2.imread(self.data_path + i, cv2.IMREAD_GRAYSCALE))
            for i in os.listdir(self.label_path):
                self.label.append(cv2.imread(self.label_path + i, cv2.IMREAD_GRAYSCALE))
        
        self.data_len = len(self.data)
    
    #return data & label
    def __getitem__(self, index):
        img, mask = self.data[index], self.label[index]
        img = np.array([img])
        mask = np.array([mask])
        
#         img = (img - img.min()) / (img.max() - img.min())
        img = img / 255
        img = torch.from_numpy(img).float()
        mask = mask / 255
#         mask = mask.round()
        mask = torch.from_numpy(mask).float()
#         img = torch.Tensor(img)
#         mask = torch.Tensor(mask)
#         mask = mask.round()
        return img, mask
    
    #return count of data
    def __len__(self):
        return self.data_len
# train_data_set = Mydataset(dir_list = f[1])
# train_data_set[0]
import torch
import torch.nn as nn
import torch.nn.functional as functional

def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
    if useBN:
        return nn.Sequential(
          nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
          nn.BatchNorm2d(dim_out),
          nn.LeakyReLU(0.1),
          nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
          nn.BatchNorm2d(dim_out),
          nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
          nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
          nn.ReLU(),
          nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
          nn.ReLU()
        )

def add_merge_stage(ch_coarse, ch_fine, in_coarse, in_fine, upsample):
    conv = nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False)
    torch.cat(conv, in_fine)

    return nn.Sequential(
        nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False)
    )
    upsample(in_coarse)

def upsample(ch_coarse, ch_fine):
    return nn.Sequential(
        nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
        nn.ReLU()
    )

class Net(nn.Module):
    def __init__(self, useBN=True):
        super(Net, self).__init__()

        self.conv1   = add_conv_stage(1, 32, useBN=useBN)
        self.conv2   = add_conv_stage(32, 64, useBN=useBN)
        self.conv3   = add_conv_stage(64, 128, useBN=useBN)
        self.conv4   = add_conv_stage(128, 256, useBN=useBN)
        self.conv5   = add_conv_stage(256, 512, useBN=useBN)

        self.conv4m = add_conv_stage(512, 256, useBN=useBN)
        self.conv3m = add_conv_stage(256, 128, useBN=useBN)
        self.conv2m = add_conv_stage(128,  64, useBN=useBN)
        self.conv1m = add_conv_stage( 64,  32, useBN=useBN)

        self.conv0  = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        self.max_pool = nn.MaxPool2d(2)

        self.upsample54 = upsample(512, 256)
        self.upsample43 = upsample(256, 128)
        self.upsample32 = upsample(128,  64)
        self.upsample21 = upsample(64 ,  32)

        ## weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        conv1_out = self.conv1(x)
        #return self.upsample21(conv1_out)
        conv2_out = self.conv2(self.max_pool(conv1_out))
        conv3_out = self.conv3(self.max_pool(conv2_out))
        conv4_out = self.conv4(self.max_pool(conv3_out))
        conv5_out = self.conv5(self.max_pool(conv4_out))

        conv5m_out = torch.cat((self.upsample54(conv5_out), conv4_out), 1)
        conv4m_out = self.conv4m(conv5m_out)

        conv4m_out_ = torch.cat((self.upsample43(conv4m_out), conv3_out), 1)
        conv3m_out = self.conv3m(conv4m_out_)

        conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)
        conv2m_out = self.conv2m(conv3m_out_)

        conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)
        conv1m_out = self.conv1m(conv2m_out_)

        conv0_out = self.conv0(conv1m_out)

        return conv0_out
def dice_coef(output, target):
    smooth = 1.

    iflat = output.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
def train():
    num_epochs = 10
    display_steps = 10
    for epoch in range(num_epochs):
#         print('Starting epoch {}/{}'.format(epoch+1, num_epochs))
        # train
        model.train()
        running_loss = 0.0
        running_dc = 0.0
        for batch_idx, (img, label) in enumerate(train_data_loader):
            images = Variable(img.cuda())
            masks = Variable(label.cuda())
        
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            dc = dice_coef(outputs.round(), masks.round())
            running_loss += loss.item()
            running_dc += dc.item()

            if batch_idx % display_steps == 0:
                print('    ', end='')
                print('batch {:>3}/{:>3} loss: {:.4f}, dc: {:.4f}\r'.format(batch_idx+1, len(train_data_loader), loss.item(), dc.item()))

        train_loss = running_loss / len(train_data_loader)
        train_dc = running_dc / len(train_data_loader)
#         print('loss: {:.4f}  dice: {:.4f} \n'.format(train_loss, train_dc))
    
    return train_loss, train_dc, model
    
def evalute():
    model.eval()
    val_running_loss = 0.0
    val_running_dc = 0.0
    for batch_idx, (img, label) in enumerate(valid_data_loader):
        images = Variable(img.cuda())
        masks = Variable(label.cuda())

        outputs = model(images)
        loss = criterion(outputs, masks)
        
        dc = dice_coef(outputs.round(), masks.round())
        val_running_loss += loss.item()
        val_running_dc += dc.item()

    val_dc = val_running_dc / len(valid_data_loader)
    val_loss = val_running_loss / len(valid_data_loader)
#     print('val_dc: {:4.4f}  val_loss: {:4.4f}\n'.format(val_dc, val_loss))
    return val_loss, val_dc
        
val_1 = [["data01","data02"], ["data03","data04","data05","data06","data07","data08"]]
val_2 = [["data03","data04"], ["data01","data02","data05","data06","data07","data08"]]
val_3 = [["data05","data06"], ["data01","data02","data03","data04","data07","data08"]]
val_4 = [["data07","data08"], ["data01","data02","data03","data04","data05","data06"]]
fold = [val_1, val_2, val_3, val_4]
total_loss = []
transform=transforms.Compose([transforms.ToTensor()])
#cross validation
for index, f in enumerate(fold):
    print("This is fold-{}".format(index+1))
    model = Net()
    criterion = nn.BCELoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_data_set = Mydataset(dir_list = f[1])
    valid_data_set = Mydataset(dir_list = f[0])
    train_data_loader = DataLoader(train_data_set, batch_size=5, num_workers=0, shuffle=True)
    valid_data_loader = DataLoader(valid_data_set, batch_size=5, num_workers=0, shuffle=True)
    train_loss, train_dc, model = train()
    val_loss, val_dc = evalute()
    print("fold->{}/{} : loss : {}, dc : {}, val_loss : {}, val_dc : {}".format(index+1, len(fold), train_loss, train_dc, val_loss, val_dc))
    total_loss.append(val_dc)
    torch.save(model.state_dict(), 'model_{}.pkl'.format(index))
print(total_loss)

