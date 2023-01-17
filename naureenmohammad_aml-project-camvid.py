#   1. Initial block of the model:

  #         Input

  #        /     \

  #       /       \

  #maxpool2d    conv2d-3x3

  #       \       /  

  #        \     /

  #      concatenate -->







#   2.Regular Downsampling&Dilation bottlenecks:

        #class for regular downsampling, dilated convolution 

  #

  #     Bottleneck Input

  #        /        \

  #       /          \

  # maxpooling2d   conv2d-1x1

  #      |             | PReLU

  #      |         conv2d-3x3

  #      |             | PReLU

  #      |         conv2d-1x1

  #      |             |

  #  Padding2d     Regularizer

  #       \           /  

  #        \         /

  #      Summing + PReLU

  #

#   3.Asymetric bottleneck:#class for separable convolutions(n.n convolutions are made into n.1 and 1.n)

  #

  #     Bottleneck Input

  #        /        \

  #       /          \

  #      |         conv2d-1x1

  #      |             | PReLU

  #      |         conv2d-1x5

  #      |             |

  #      |         conv2d-5x1

  #      |             | PReLU

  #      |         conv2d-1x1

  #      |             |

  #  Padding2d     Regularizer

  #       \           /  

  #        \         /

  #      Summing + PReLU

    

#   4. Upsampling bottleneck: - #class for upsampling bottlenecks

  #           Input

  #        /        \

  #       /          \

  # conv2d-1x1     convTrans2d-1x1

  #      |             | PReLU

  #      |         convTrans2d-3x3

  #      |             | PReLU

  #      |         convTrans2d-1x1

  #      |             |

  # maxunpool2d    Regularizer

  #       \           /  

  #        \         /

  #      Summing + PReLU

  #
import torch

import torch.nn as nn

import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import StepLR

import cv2

import os

from tqdm import tqdm

from PIL import Image

class Initial_Encoding(nn.Module):

    

    def __init__ (self,chanel_in = 3,chanel_out = 13):

        super().__init__()





        self.maxpool = nn.MaxPool2d(kernel_size=2, stride = 2, padding = 0)



        self.conv = nn.Conv2d(in_channels=chanel_in, out_channels=chanel_out, kernel_size = 3,

                              stride = 2, padding = 1)



        self.prelu = nn.PReLU(16)



        self.batchnorm = nn.BatchNorm2d(chanel_out)

  

    def forward(self, x):

        

        main = self.conv(x)

        main = self.batchnorm(main)

        

        side = self.maxpool(x)

        

        # concatenating on the channels axis

        x = torch.cat((main, side), dim=1)

        x = self.prelu(x)

        

        return x
   #  dilation (bool) - if True: creating dilation bottleneck

  #  stride_p ={1,2} - if 2: creating downsampling bottleneck

  #  projection_ratio - ratio between input and output channels

  #  a: activation - if a=1: relu used as the activation function else if a=2: Prelu us used

  #  p - dropout ratio

class Downsampling_Dilation(nn.Module):

    def __init__(self, dilation, chanel_in, chanel_out, stride_p, a=1, projection_ratio=4, p=0.1):

             

        super().__init__()

        

        # Define class variables

        self.chanel_in = chanel_in

        

        self.chanel_out = chanel_out

        self.dilation = dilation

        self.stride_p = stride_p

        

        # calculating the number of reduced channels

        if stride_p:

            self.stride = 2

            self.reduced_depth = int(chanel_in // projection_ratio)

        else:

            self.stride = 1

            self.reduced_depth = int(chanel_out // projection_ratio)

        

        if (a==1):

            activation = nn.ReLU()

        elif(a==2):

            activation = nn.PReLU()

        

        self.maxpool = nn.MaxPool2d(kernel_size = 2,

                                      stride = 2,

                                      padding = 0, return_indices=True)

        



        

        self.dropout = nn.Dropout2d(p=p)



        self.conv1 = nn.Conv2d(in_channels = self.chanel_in, out_channels = self.reduced_depth,

                               kernel_size = 1,stride = 1,padding = 0, bias = False, dilation = 1)

        

        self.prelu1 = activation

        

        self.conv2 = nn.Conv2d(in_channels = self.reduced_depth,out_channels = self.reduced_depth,

                                  kernel_size = 3, stride = self.stride,padding = self.dilation,

                                  bias = True,dilation = self.dilation)

                                  

        self.prelu2 = activation

        

        self.conv3 = nn.Conv2d(in_channels = self.reduced_depth,out_channels = self.chanel_out,

                                  kernel_size = 1,stride = 1,padding = 0,bias = False,dilation = 1)

        

        self.prelu3 = activation

        

        self.batchnorm = nn.BatchNorm2d(self.reduced_depth)

        self.batchnorm2 = nn.BatchNorm2d(self.chanel_out)

        

        

    def forward(self, x):

        

        bs = x.size()[0]

        x_copy = x

        

        # Side Branch

        x = self.conv1(x)

        x = self.batchnorm(x)

        x = self.prelu1(x)

        

        x = self.conv2(x)

        x = self.batchnorm(x)

        x = self.prelu2(x)

        

        x = self.conv3(x)

        x = self.batchnorm2(x)

                

        x = self.dropout(x)

        

        # Main Branch

        if self.stride_p:

            x_copy, indices = self.maxpool(x_copy)

          

        if self.chanel_in != self.chanel_out:

            out_shape = self.chanel_out - self.chanel_in

            

            #padding and concatenating in order to match the channels axis of the side and main branches

            extras = torch.zeros((bs, out_shape, x.shape[2], x.shape[3]))

            if torch.cuda.is_available():

                extras = extras.cuda()

            x_copy = torch.cat((x_copy, extras), dim = 1)



        # Summing main and side branches

        x = x + x_copy

        x = self.prelu3(x)

        

        if self.stride_p:

            return x, indices

        else:

            return x
  

  #  projection_ratio - ratio between input and output channels

class ASNeck(nn.Module):

    def __init__(self, chanel_in, chanel_out, projection_ratio=4):

             

        super().__init__()

        

        # Define class variables

        self.chanel_in = chanel_in

        self.reduced_depth = int(chanel_in / projection_ratio)

        self.chanel_out= chanel_out

        

        self.dropout = nn.Dropout2d(p=0.1)

        

        self.conv1 = nn.Conv2d(in_channels= self.chanel_in,out_channels = self.reduced_depth,

                               kernel_size = 1,stride = 1,padding = 0,bias = False)

        

        self.prelu1 = nn.PReLU()

        

        self.conv21 = nn.Conv2d(in_channels = self.reduced_depth,out_channels = self.reduced_depth,

                                  kernel_size = (1, 5), stride = 1,padding = (0, 2), bias = False)

        

        self.conv22 = nn.Conv2d(in_channels = self.reduced_depth, out_channels = self.reduced_depth,

                                  kernel_size = (5, 1),stride = 1,padding = (2, 0),bias = False)

        

        self.prelu2 = nn.PReLU()

        

        self.conv3 = nn.Conv2d(in_channels = self.reduced_depth,out_channels = self.chanel_out,

                                  kernel_size = 1, stride = 1,padding = 0,bias = False)

        

        self.prelu3 = nn.PReLU()

        

        self.batchnorm = nn.BatchNorm2d(self.reduced_depth)

        self.batchnorm2 = nn.BatchNorm2d(self.chanel_out)

        

    def forward(self, x):

        bs = x.size()[0]

        x_copy = x

        

        # Side Branch

        x = self.conv1(x)

        x = self.batchnorm(x)

        x = self.prelu1(x)

        

        x = self.conv21(x)

        x = self.conv22(x)

        x = self.batchnorm(x)

        x = self.prelu2(x)

        

        x = self.conv3(x)

                

        x = self.dropout(x)

        x = self.batchnorm2(x)

        

        # Main Branch

        

        if self.chanel_in != self.chanel_out:

            out_shape = self.chanel_out - self.chanel_in

            

            #padding and concatenating in order to match the channels axis of the side and main branches

            extras = torch.zeros((bs, out_shape, x.shape[2], x.shape[3]))

            if torch.cuda.is_available():

                extras = extras.cuda()

            x_copy = torch.cat((x_copy, extras), dim = 1)

        

        # Summing main and side branches

        x = x + x_copy

        x = self.prelu3(x)

        

        return x
 #  projection_ratio - ratio between input and output channels

a=1 #activation function,if 1: relu used as the activation function else if a=2: Prelu us used

class Upsampl_layer(nn.Module):

    

    def __init__(self, chanel_in, chanel_out, a, projection_ratio=4):

        

        super().__init__()

        

        # Define class variables

        self.chanel_in = chanel_in

        self.reduced_depth = int(chanel_in / projection_ratio)

        self.chanel_out = chanel_out

        

        

        if (a==1):

            activation = nn.ReLU()

        elif(a==2):

            activation = nn.PReLU()

        elif(a==3):

            activation = nn.Softmax()    

        

        self.unpool = nn.MaxUnpool2d(kernel_size = 2,

                                     stride = 2)

        

        self.main_conv = nn.Conv2d(in_channels = self.chanel_in,out_channels = self.chanel_out,

                                   kernel_size = 1)

        

        self.dropout = nn.Dropout2d(p=0.1)

        

        

        self.convt1 = nn.ConvTranspose2d(in_channels = self.chanel_in,out_channels = self.reduced_depth,

                               kernel_size = 1,padding = 0, bias = False)

        

        

        self.prelu1 = activation

        

        # This layer used for Upsampling

        self.convt2 = nn.ConvTranspose2d(in_channels = self.reduced_depth,

                                  out_channels = self.reduced_depth, kernel_size = 3,

                                  stride = 2, padding = 1, output_padding = 1,bias = False)

        

        self.prelu2 = activation

        

        self.convt3 = nn.ConvTranspose2d(in_channels = self.reduced_depth,out_channels = self.chanel_out,

                                  kernel_size = 1, padding = 0,bias = False)

        

        self.prelu3 = activation

        

        self.batchnorm = nn.BatchNorm2d(self.reduced_depth)

        self.batchnorm2 = nn.BatchNorm2d(self.chanel_out)

        

    def forward(self, x, indices):

        x_copy = x

        

        # Side Branch

        x = self.convt1(x)

        x = self.batchnorm(x)

        x = self.prelu1(x)

        

        x = self.convt2(x)

        x = self.batchnorm(x)

        x = self.prelu2(x)

        

        x = self.convt3(x)

        x = self.batchnorm2(x)

        

        x = self.dropout(x)

        

        # Main Branch

        

        x_copy = self.main_conv(x_copy)

        x_copy = self.unpool(x_copy, indices, output_size=x.size())

        

        # summing the main and side branches

        x = x + x_copy

        x = self.prelu3(x)

        

        return x
# Class - number of classes

class ENet(nn.Module):

  

  # Creating Enet model!

  

    def __init__(self, n_Class):

        super().__init__()

        

        self.Class = n_Class

        

        # The initial block

        self.init = Initial_Encoding()

        

        

        # The first bottleneck

        self.b10 = Downsampling_Dilation(dilation=1,chanel_in=16,chanel_out=64,stride_p=True,p=0.01)

        

        self.b11 = Downsampling_Dilation(dilation=1,chanel_in=64,chanel_out=64,stride_p=False,p=0.01)

        

        self.b12 = Downsampling_Dilation(dilation=1,chanel_in=64,chanel_out=64,stride_p=False,p=0.01)

        

        self.b13 = Downsampling_Dilation(dilation=1,chanel_in=64,chanel_out=64,stride_p=False,p=0.01)

        

        self.b14 = Downsampling_Dilation(dilation=1,chanel_in=64,chanel_out=64,stride_p=False,p=0.01)

        

        

        # The second bottleneck

        self.b20 = Downsampling_Dilation(dilation=1, chanel_in=64,chanel_out=128, stride_p=True)

        

        self.b21 = Downsampling_Dilation(dilation=1, chanel_in=128, chanel_out=128,stride_p=False)

        

        self.b22 = Downsampling_Dilation(dilation=2, chanel_in=128, chanel_out=128, stride_p=False)

        

        self.b23 = ASNeck(chanel_in=128,chanel_out=128)

        

        self.b24 = Downsampling_Dilation(dilation=4, chanel_in=128,chanel_out=128,stride_p=False)

        

        self.b25 = Downsampling_Dilation(dilation=1, chanel_in=128, chanel_out=128, stride_p=False)

        

        self.b26 = Downsampling_Dilation(dilation=8,chanel_in=128,  chanel_out=128, stride_p=False)

        

        self.b27 = ASNeck(chanel_in=128,chanel_out=128)

        

        self.b28 = Downsampling_Dilation(dilation=16, chanel_in=128,chanel_out=128, stride_p=False)

        

        

        # The third bottleneck

        self.b31 = Downsampling_Dilation(dilation=1, chanel_in=128, chanel_out=128,stride_p=False)

        

        self.b32 = Downsampling_Dilation(dilation=2,  chanel_in=128, chanel_out=128, stride_p=False)

        

        self.b33 = ASNeck(chanel_in=128, chanel_out=128)

        

        self.b34 = Downsampling_Dilation(dilation=4,chanel_in=128, chanel_out=128,stride_p=False)

        

        self.b35 = Downsampling_Dilation(dilation=1, chanel_in=128, chanel_out=128, stride_p=False)

        

        self.b36 = Downsampling_Dilation(dilation=8, chanel_in=128,chanel_out=128,  stride_p=False)

        

        self.b37 = ASNeck(chanel_in=128, chanel_out=128)

        

        self.b38 = Downsampling_Dilation(dilation=16, chanel_in=128,chanel_out=128,stride_p=False)

        

        

        # The fourth bottleneck

        self.b40 = Upsampl_layer(chanel_in=128, chanel_out=64,  a=1)

        

        self.b41 = Downsampling_Dilation(dilation=1,chanel_in=64,chanel_out=64, stride_p=False,a=1)

        

        self.b42 = Downsampling_Dilation(dilation=1,chanel_in=64,chanel_out=64, stride_p=False,a=1)

        

        

        # The fifth bottleneck

        self.b50 = Upsampl_layer(chanel_in=64, chanel_out=16, a=1)

        

        self.b51 = Downsampling_Dilation(dilation=1,chanel_in=16,chanel_out=16,stride_p=False,a=1)

        

        

        # Final ConvTranspose Layer

        self.fullconv = nn.ConvTranspose2d(in_channels=16,out_channels=self.Class,  kernel_size=3, 

                                           stride=2, padding=1, output_padding=1, bias=False)

        

        

    def forward(self, x):

        

        # The initial block

        x = self.init(x)

        

        # The first bottleneck

        x, i1 = self.b10(x)

        x = self.b11(x)

        x = self.b12(x)

        x = self.b13(x)

        x = self.b14(x)

        

        # The second bottleneck

        x, i2 = self.b20(x)

        x = self.b21(x)

        x = self.b22(x)

        x = self.b23(x)

        x = self.b24(x)

        x = self.b25(x)

        x = self.b26(x)

        x = self.b27(x)

        x = self.b28(x)

        

        # The third bottleneck

        x = self.b31(x)

        x = self.b32(x)

        x = self.b33(x)

        x = self.b34(x)

        x = self.b35(x)

        x = self.b36(x)

        x = self.b37(x)

        x = self.b38(x)

        

        # The fourth bottleneck

        x = self.b40(x, i2)

        x = self.b41(x)

        x = self.b42(x)

        

        # The fifth bottleneck

        x = self.b50(x, i1)

        x = self.b51(x)

        

        # Final ConvTranspose Layer

        x = self.fullconv(x)

        

        return x
def loader(training_path, segmented_path, batch_size, h=320, w=1000):

    filenames_t = os.listdir(training_path)

    total_files_t = len(filenames_t)

    

    filenames_s = os.listdir(segmented_path)

    segmented_img = len(filenames_s)

    

    assert(total_files_t == segmented_img)

    

    if str(batch_size).lower() == 'all':

        batch_size = segmented_img

    

    idx = 0

    while(1):

      # 1d array containing random indexes of images and labels

        batch_idxs = np.random.randint(0, segmented_img, batch_size)

            

        

        inputs = []

        labels = []

        

        for jj in batch_idxs:

          # Reading normalized photo

            img = plt.imread(training_path + filenames_t[jj])

          # Resizing using nearest neighbor method to get sharp, jaggy image

            img = cv2.resize(img, (h, w), cv2.INTER_NEAREST)

            inputs.append(img)

          

          # Reading semantic image

            img = Image.open(segmented_path + filenames_s[jj])

            img = np.array(img)

          # Resizing using nearest neighbor method

            img = cv2.resize(img, (h, w), cv2.INTER_NEAREST)

            labels.append(img)

        

        #Joining 3d-images along axis=2 --> (h,w,c) to (h,c,w)

        inputs = np.stack(inputs, axis=2) 

      # Changing image format to C x H x W matrices

        inputs = torch.tensor(inputs).transpose(0, 2).transpose(1, 3)

        

        labels = torch.tensor(labels)

        print(labels.shape)

        

        yield inputs, labels #similar to return 
def get_class_weights(num_classes, c=1.02):

    pipe = loader('../input/camvid-dataset/train/', '../input/camvid-dataset/trainannot/', batch_size='all')

    _, labels = next(pipe)

    all_labels = labels.flatten()

    each_class = np.bincount(all_labels, minlength=num_classes)

    prospensity_score = each_class / len(all_labels)

    class_weights = 1 / (np.log(c + prospensity_score))

    return class_weights



class IoULoss(nn.Module):

    def __init__(self, weight=None, size_average=True):

        super(IoULoss, self).__init__()



    def forward(self, inputs, targets, smooth=1):

        

        #comment out if your model contains a sigmoid or equivalent activation layer

        inputs = F.sigmoid(inputs)       

        

        #flatten label and prediction tensors

        inputs = inputs.view(-1)

        targets = targets.view(-1)

        

        #intersection is equivalent to True Positive count

        #union is the mutually inclusive area of all labels & predictions 

        intersection = (inputs * targets).sum()

        total = (inputs + targets).sum()

        union = total - intersection 

        

        IoU = (intersection + smooth)/(union + smooth)

                

        return 1 - IoU



enet = ENet(12)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

enet = enet.to(device)

class_weights = get_class_weights(12)
iters_t = []

iters_v=[]

train_losses = []

val_losses = []

iou_train=[]

iou_val=[]



opt=1

if(opt==1):

    lr = 5e-4

    optimizer = torch.optim.Adam(enet.parameters(),lr=lr,weight_decay=2e-4)

elif(opt==2):

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.99)

    

print_every = 5

eval_every = 5

batch_size = 10

minib_train = 367 // batch_size # mini_batch train 

minib_val = 101 // batch_size  # mini_batch validation

loss_f=1



pipe = loader('../input/camvid-dataset/train/', '../input/camvid-dataset/trainannot/', batch_size)

eval_pipe = loader('../input/camvid-dataset/val/', '../input/camvid-dataset/valannot/', batch_size)

epochs = 100

# 100

# Train loop

for e in range(1, epochs+1):

    train_loss = 0

    enet.train()

    for _ in tqdm(range(minib_train)):

        X_batch, mask_batch = next(pipe)

        X_batch, mask_batch = X_batch.to(device), mask_batch.to(device) # assign data to cpu/gpu

        

        optimizer.zero_grad()

        out = enet(X_batch.float())

        

        # loss calculation

        if (loss_f==1):

            criteria=nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))

            loss = criteria(out, mask_batch.long())

        elif(loss_f==2):

            loss = IoULoss(out,mask_batch.long())

        

        loss.backward() # update weights

        optimizer.step() #update gradients



        train_loss += loss.item()

#     print ()

    train_losses.append(train_loss)

    iters_t.append(e)

    

#     if (e+1) % print_every == 0:

#         print ('Epoch {}/{}...'.format(e, epochs),

#                 'Loss {:6f}'.format(train_loss))

    

    if e % eval_every == 0:

        with torch.no_grad():

            enet.eval()

            val_loss = 0

            for _ in tqdm(range(minib_val)):

                inputs, labels = next(eval_pipe)

                inputs, labels = inputs.to(device), labels.to(device)

                out = enet(inputs)

                out = out.data.max(1)[1]

                val_loss += (labels.long() - out.long()).sum()  

            print ()

            print ('Loss {:6f}'.format(val_loss))

            val_losses.append(val_loss)

            iters_v.append(e)

            

    if e % print_every == 0:

        checkpoint = {'epochs' : e, 'state_dict' : enet.state_dict()}

        torch.save(checkpoint, '/content/ckpt-enet-{}-{}.pth'.format(e, train_loss))

        print ('Model saved!')



if(loss_f==1):

    print ('Epoch {}/{}...'.format(e, epochs),

       'Total Mean Loss: {:6f}'.format(sum(train_losses) / epochs))



elif(loss_f==2):

    print ('Epoch {}/{}...'.format(e, epochs),

       'IoU: {:6f}'.format(sum(train_losses) / epochs))
plt.figure()

plt.subplot(iters_t, train_losses)

plt.subplot(iters_v, val_losses)

plt.show()
# Load a pretrained model if needed

enet = ENet(12)

state_dict = torch.load('../input/weights/ckpt-enet.pth',map_location='cpu')['state_dict']

enet.load_state_dict(state_dict)
fname = 'Seq05VD_f05100.png'

sample_img = plt.imread('../input/camvid-dataset/test/' + fname)

sample_img = cv2.resize(sample_img, (512, 512), cv2.INTER_NEAREST)

sample_img = torch.tensor(sample_img).unsqueeze(0).float()

sample_img = sample_img.transpose(2, 3).transpose(1, 2).to(device)



enet.to(device)

with torch.no_grad():

    out1 = enet(sample_img.float()).squeeze(0)
sample_img_seg = Image.open('../input/camvid-dataset/testannot/' + fname)

sample_img_seg = cv2.resize(np.array(sample_img_seg), (512, 512), cv2.INTER_NEAREST)
out2 = out1.cpu().detach().numpy()
mno = 8 # Should be between 0 - n-1 | where n is the number of classes



figure = plt.figure(figsize=(20, 10))

plt.subplot(1, 3, 1)

plt.title('Input Image')

plt.axis('off')

plt.imshow(sample_img)

plt.subplot(1, 3, 2)

plt.title('Output Image')

plt.axis('off')

plt.imshow(out2[mno, :, :])

plt.show()
op_label = out1.data.max(0)[1].cpu().numpy()
def decode_segmap(image):

    Sky = [128, 128, 128]

    Building = [128, 0, 0]

    Pole = [192, 192, 128]

    Road_marking = [255, 69, 0]

    Road = [128, 64, 128]

    Pavement = [60, 40, 222]

    Tree = [128, 128, 0]

    SignSymbol = [192, 128, 128]

    Fence = [64, 64, 128]

    Car = [64, 0, 128]

    Pedestrian = [64, 64, 0]

    Bicyclist = [0, 128, 192]



    label_colours = np.array([Sky, Building, Pole, Road_marking, Road, 

                              Pavement, Tree, SignSymbol, Fence, Car, 

                              Pedestrian, Bicyclist]).astype(np.uint8)

    r = np.zeros_like(image).astype(np.uint8)

    g = np.zeros_like(image).astype(np.uint8)

    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, 12):

        r[image == l] = label_colours[l, 0]

        g[image == l] = label_colours[l, 1]

        b[image == l] = label_colours[l, 2]



    rgb = np.zeros((image.shape[0], image.shape[1], 3)).astype(np.uint8)

    rgb[:, :, 0] = b

    rgb[:, :, 1] = g

    rgb[:, :, 2] = r

    return rgb
true_seg = decode_segmap(sample_img_seg)

pred_seg = decode_segmap(op_label)
figure = plt.figure(figsize=(20, 10))

plt.subplot(1, 3, 1)

plt.title('Input Image')

plt.axis('off')

plt.imshow(sample_img)

plt.subplot(1, 3, 2)

plt.title('Predicted Segmentation')

plt.axis('off')

plt.imshow(pred_seg)

plt.subplot(1, 3, 3)

plt.title('Ground Truth')

plt.axis('off')

plt.imshow(true_seg)

plt.show()
# Save the parameters

checkpoint = {

    'epochs' : e,

    'state_dict' : enet.state_dict()

}

torch.save(checkpoint, 'ckpt-enet-1.pth')
# Save the model

torch.save(enet, 'model')





# import pickle 

# filename = 'enet1.sav'

# pickle.dump(enet, open(filename, 'wb'))