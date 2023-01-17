# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import os

import cv2

import sys

import torch

import numpy as np

from os import listdir

from os.path import isfile, isdir

from torch.utils.data import Dataset, DataLoader



from torch import nn

from os import listdir

# from model import Encoder

import matplotlib.pyplot as plt

# from data_preprosess import get_clean_data

from torch.optim import Adam, RMSprop, SGD

from PIL import Image

from torch.optim.lr_scheduler import StepLR





# Any results you write to the current directory are saved as output.
import torch

from torch import nn

import torch.nn.functional as f

from torch.autograd import Variable

import torch.nn.functional as F





# Define some constants

KERNEL_SIZE = 3

PADDING = KERNEL_SIZE // 2



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")







class ConvLSTMCell(nn.Module):

    """

    Generate a convolutional LSTM cell

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    def __init__(self, input_size, hidden_size):

        super().__init__()

        self.input_size = input_size

        self.hidden_size = hidden_size

        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, KERNEL_SIZE, padding=PADDING).cuda()



    def forward(self, input_, prev_state):



        # get batch and spatial sizes

        batch_size = input_.data.size()[0]

        spatial_size = input_.data.size()[2:]



        # generate empty prev_state, if None is provided

        if prev_state is None:

            state_size = [batch_size, self.hidden_size] + list(spatial_size)

            prev_state = (

                Variable(torch.zeros(state_size).to(device)),

                Variable(torch.zeros(state_size).to(device))

            )



        prev_hidden, prev_cell = prev_state



        # data size is [batch, channel, height, width]

        stacked_inputs = torch.cat((input_.to(device), prev_hidden), 1)

        gates = self.Gates(stacked_inputs)



        # chunk across channel dimension

        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)



        # apply sigmoid non linearity

        in_gate = torch.sigmoid(in_gate)

        remember_gate = torch.sigmoid(remember_gate)

        out_gate = torch.sigmoid(out_gate)



        # apply tanh non linearity

        cell_gate = torch.tanh(cell_gate)



        # compute current cell and hidden state

        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)

        hidden = out_gate * torch.tanh(cell)



        return hidden, cell



class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size, padding, batch_norm):

        super(UNetConvBlock, self).__init__()

        block = []



        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))

        block.append(nn.ELU(inplace = True))

        block.append(nn.Dropout2d(p = 0.2))

        if batch_norm:

            block.append(nn.BatchNorm2d(out_size))



        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))

        block.append(nn.ELU(inplace=True))

        block.append(nn.Dropout2d(p = 0.2))

        if batch_norm:

            block.append(nn.BatchNorm2d(out_size))



        self.block = nn.Sequential(*block)



    def forward(self, x):

        out = self.block(x)

        return out





class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):

        super(UNetUpBlock, self).__init__()

        if up_mode == 'upconv':

            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)

        elif up_mode == 'upsample':

            self.up = nn.Sequential(

                nn.Upsample(mode='bilinear', scale_factor=2),

                nn.Conv2d(in_size, out_size, kernel_size=1),

            )



        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)



    def center_crop(self, layer, target_size):

        _, _, layer_height, layer_width = layer.size()

        diff_y = (layer_height - target_size[0]) // 2

        diff_x = (layer_width - target_size[1]) // 2

        return layer[

            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])

        ]



    def forward(self, x, bridge):

        up = self.up(x)

        crop1 = self.center_crop(bridge, up.shape[2:])

        out = torch.cat([up, crop1], 1)

        out = self.conv_block(out)



        return out





class Encoder(nn.Module):

    def __init__(

        self,

        in_channels=1,

        n_classes=1,

        depth=4,

        wf=3,

        padding=True,

        batch_norm=True,

    ):

        """

        Args:

            in_channels (int): number of input channels

            n_classes (int): number of output channels

            depth (int): depth of the network

            wf (int): number of filters in the first layer is 2**wf

            padding (bool): if True, apply padding such that the input shape

                            is the same as the output.

                            This may introduce artifacts

            batch_norm (bool): Use BatchNorm after layers with an

        """

        super(Encoder, self).__init__()

        self.padding = padding

        self.depth = depth

        prev_channels = in_channels

        

        self.final_cell = ConvLSTMCell(64, 64)

        

        self.down_path = nn.ModuleList()

        

        for i in range(depth):

            self.down_path.append(

                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm)

            )

            prev_channels = 2 ** (wf + i)

        



    def forward(self, x, state):

        blocks = []

        for i, down in enumerate(self.down_path):

            x = down(x)

            if i != len(self.down_path) - 1:

                blocks.append(x)

                x = F.max_pool2d(x, 2) 

            

        hidden_h, hidden_c = self.final_cell(x, state)

        

        state = [hidden_h, hidden_c]

        return state, blocks





class Decoder(nn.Module):

    def __init__(

        self,

        in_channels=64,

        n_classes=1,

        depth=4,

        wf=3,

        padding=True,

        batch_norm=True,

        up_mode='upconv'

    ):

        """

        Args:

            in_channels (int): number of input channels

            n_classes (int): number of output channels

            depth (int): depth of the network

            wf (int): number of filters in the first layer is 2**wf

            padding (bool): if True, apply padding such that the input shape

                            is the same as the output.

                            This may introduce artifacts

            batch_norm (bool): Use BatchNorm after layers with an

        """

        super(Decoder, self).__init__()

        self.padding = padding

        self.depth = depth

        prev_channels = in_channels

        self.final_cell = ConvLSTMCell(64, 64)

        self.up_path = nn.ModuleList()

       

        

        for i in reversed(range(depth - 1)):

            self.up_path.append(

                UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm)

            )

            prev_channels = 2 ** (wf + i)

        



        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

        

        

    def forward(self, x, blocks, state):



        x, hidden_c = self.final_cell(x, state)



        state = [x, hidden_c] 



        for i, up in enumerate(self.up_path):

            x = up(x, blocks[-i - 1])



        return self.last(x), state
class LandmarksDataset(Dataset):

    """Face Landmarks dataset."""



    def __init__(self, img_paths, IMAGES_WIDTH, IMAGES_HEIGHT, IMAGES_DEPTH, frame =6):



        self.frame = frame

        self.IMAGES_WIDTH = IMAGES_WIDTH

        self.IMAGES_HEIGHT = IMAGES_HEIGHT

        self.IMAGES_DEPTH = IMAGES_DEPTH

        self.img_paths = img_paths



        self.images, self.masks = self.get_proper_preprosess_data(self.img_paths, self.frame)



    def __len__(self):

        return len(self.images)

    

    def get_proper_preprosess_data(self, image_ids, frame):

        length = len(image_ids)

        imgs = []

        lbls = []

        cnt_y = 0

        for i in range(0, length, frame):

            images = []

            labels = []

            cnt = 0



            for j in range(i, i + frame):



                if j < length:

                    images.append(image_ids[j])

                else:

                    images.append(image_ids[cnt])

                    cnt += 1   



                le = j+6

                if le < length:

                    try:

                        labels.append(image_ids[le])

                    except:

                        labels.append(image_ids[0])

                else:

                    labels.append(image_ids[cnt_y])

                    cnt_y +=1



            imgs.append(images)

            lbls.append(labels)



        return imgs, lbls



    def __getitem__(self, idx):

        

        image = self.images[idx]

        label = self.masks[idx]

                

        features_list = np.zeros((self.frame, self.IMAGES_WIDTH, self.IMAGES_HEIGHT, self.IMAGES_DEPTH), dtype = np.uint8)

        labels_list = np.zeros((self.frame, self.IMAGES_WIDTH, self.IMAGES_HEIGHT, self.IMAGES_DEPTH), dtype = np.uint8)



        def get_feature_images(paths, features_list):

            for i, path in enumerate(paths):

                image = cv2.imread(path, -1)

                image = cv2.resize(image, (self.IMAGES_WIDTH, self.IMAGES_HEIGHT))

                image = image.reshape(self.IMAGES_WIDTH, self.IMAGES_HEIGHT, self.IMAGES_DEPTH)

                image = image.astype(np.uint8)

                features_list[i] = image

            return features_list

        

        

        feature = get_feature_images(image, features_list) / 255

        labels = get_feature_images(label, labels_list) 

        

        sample = [feature, labels]



        return sample
IMAGES_WIDTH = 256

IMAGES_HEIGHT = 256

IMAGES_DEPTH = 1



TIME_STEMP = 6

np.random.seed(14)



data_dir = '../input/data/mapped'

img_paths = [os.path.join(data_dir, f) for f in listdir(data_dir) if isfile(os.path.join(data_dir, f))]

train_paths =img_paths[:int(len(img_paths) * 0.8)] 

test_paths = img_paths[int(len(img_paths) * 0.8):]



train_paths = train_paths[:20000]
dataset = LandmarksDataset(train_paths, IMAGES_WIDTH, IMAGES_HEIGHT, IMAGES_DEPTH, TIME_STEMP)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, drop_last=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



encoder = Encoder().to(device)

decoder = Decoder().to(device)

device
def model_train(inputs, targets,encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, clip = 1):

    

    encoder_optimizer.zero_grad()

    decoder_optimizer.zero_grad()

    

    state = None

    

    batch_size = targets.shape[1]

    input_seq_length = inputs.shape[0]

    output_seq_length = targets.shape[0]

    

    loss = 0



    for time in range(input_seq_length):

        state, block = encoder(inputs[time], state)    

    

    decoder_hidden = state

    

    layer = block

    

    decoder_input = torch.zeros(state[0].shape).type(torch.FloatTensor).to(device)

        

    def get_features(x, state):

        state, block = encoder(x, state)

        return block, state

            

    for t in range(output_seq_length):



        decoder_output, decoder_hidden = decoder(decoder_input, layer, decoder_hidden)



        layer, decoder_hidden = get_features(targets[t], decoder_hidden)



        loss += criterion(decoder_output, targets[t])

        decoder_input = decoder_hidden[0]

    

    loss.backward()

    

    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)

    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    

    encoder_optimizer.step()

    decoder_optimizer.step()

    return loss.item() / output_seq_length
criterion = nn.MSELoss()





encoder_optimizer = SGD(encoder.parameters(), lr=0.1, momentum=0.9, nesterov=True)

decoder_optimizer = SGD(decoder.parameters(), lr=0.01, momentum=0.9, nesterov=True)



encoder_scheduler = StepLR(encoder_optimizer, step_size=2, gamma=0.96)

decoder_scheduler = StepLR(decoder_optimizer, step_size=4, gamma=0.96)



epochs = 10

teacher_force_ratio = 0.5
for epoch in range(1, epochs+1):

    

    encoder_scheduler.step()

    decoder_scheduler.step()

    

    print('Epoch:', epoch,'Encoder LR:', encoder_scheduler.get_lr())

    print('Epoch:', epoch,'Decoder LR:', decoder_scheduler.get_lr())

    

    encoder.train()

    decoder.train()

    total_loss_iterations = 0

    train_loss = 0

    for i, [x, y] in enumerate(train_loader):

          

        tensor = x.type(torch.FloatTensor).view(6, 8, 1, 256, 256).cuda()

        label = y.type(torch.FloatTensor).view(6, 8, 1, 256, 256).cuda()

        

        loss = model_train(tensor, label, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_force_ratio)

        total_loss_iterations += loss

        train_loss = total_loss_iterations / len(train_paths) 



    print('Epochs : {}/15'.format(epoch), '...Training loss : {0:.4f}'.format(train_loss))

    torch.save(encoder.state_dict(), 'encoder_by_jimit_jayswal.pt')

    torch.save(decoder.state_dict(), 'decoder_by_jimit_jayswal.pt')

    torch.cuda.empty_cache()