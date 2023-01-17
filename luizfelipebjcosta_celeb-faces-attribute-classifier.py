import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

from torch import nn

import math

from PIL import Image

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from torchvision import transforms



device = 'cuda'

hidden_dim=64

im_chan=3

batch_size=128

max_epochs=100000

display_step=5

n_epochs_stop=5

image_size=64

transform = transforms.Compose([

        transforms.Resize(image_size),

        transforms.CenterCrop(image_size),

        transforms.ToTensor(),

        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

    ])

lr=0.01 #learning rate

df = pd.read_csv('/kaggle/input/celeba-dataset/list_attr_celeba.csv')

df = df.sample(frac=1).reset_index(drop=True)

n_rows = df.shape[0]

n_classes=df.shape[1]-1

#dividir em treinamento validacao e teste

train = df.iloc[:n_rows//2,:]

val = df.iloc[n_rows//2:3*n_rows//4,:]

test = df.iloc[3*n_rows//4:,:]

train_ids = [train.iloc[i*batch_size:min(i*batch_size+batch_size,train.shape[0]),0] for i in range(int(math.ceil(train.shape[0]/batch_size)))]

batches = [torch.Tensor(train.iloc[i*batch_size:min(i*batch_size+batch_size,train.shape[0]),1:].values).long().to(device) for i in range(int(math.ceil(train.shape[0]/batch_size)))]

#Estou calculando a validacao por batch para economizar memoria

#como o conjunto de validacao tem metade do tamanho do de treinamento o tamanho do batch ou o numero de batches

#tem que ser dividido por 2 escolhi reduzir o tamanho do batch em vez da quantidade porque assim eu economizo tempo

#(uso o mesmo for)

val_ids = [val.iloc[i*batch_size//2:min(i*batch_size//2+batch_size//2,val.shape[0]),0] for i in range(int(math.ceil(train.shape[0]/batch_size)))]

val_batches = [torch.Tensor(val.iloc[i*batch_size//2:min(i*batch_size//2+batch_size//2,val.shape[0]),1:].values).long().to(device) for i in range(int(math.ceil(train.shape[0]/batch_size)))]
##Definicao do Classificador

class Classif(nn.Module):

    def __init__(self, im_chan=3, hidden_dim=64,n_classes=40):

        super(Classif, self).__init__()

        self.classif = nn.Sequential(

            self.make_classif_block(im_chan, hidden_dim),

            self.make_classif_block(hidden_dim, hidden_dim * 2),

            self.make_classif_block(hidden_dim * 2, hidden_dim * 4, stride=3),

            self.make_classif_block(hidden_dim * 4, n_classes, final_layer=True),

        )



    def make_classif_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):

        '''

        Function to return a sequence of operations corresponding to a block of the classifier

        Parameters:

            input_channels: how many channels the input feature representation has

            output_channels: how many channels the output feature representation should have

            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)

            stride: the stride of the convolution

            final_layer: a boolean, true if it is the final layer and false otherwise 

                      (affects activation and batchnorm)

        '''

        if not final_layer:

            return nn.Sequential(

                nn.Conv2d(input_channels, output_channels, kernel_size, stride),

                nn.BatchNorm2d(output_channels),

                nn.LeakyReLU(0.2, inplace=True),

            )

        else:

            return nn.Sequential(

                nn.Conv2d(input_channels, output_channels, kernel_size, stride),

                nn.Tanh()

            )



    def forward(self, image):

        '''

        Function for completing a forward pass of the classifier

        Parameters:

            image: a flattened image tensor with dimension (im_chan)

        '''

        classif_pred = self.classif(image)

        return classif_pred.view(len(classif_pred), -1)
classif_loss = nn.MultiLabelMarginLoss()
#inicializar e treinar

classif = Classif(im_chan,hidden_dim,n_classes).to(device)

classif_opt = torch.optim.Adam(classif.parameters(), lr=lr)





def weights_init(m):

    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):

        torch.nn.init.normal_(m.weight, 0.0, 0.02)

    if isinstance(m, nn.BatchNorm2d):

        torch.nn.init.normal_(m.weight, 0.0, 0.02)

        torch.nn.init.constant_(m.bias, 0)

classif = classif.apply(weights_init)
#treinamento

min_val_loss=float('inf')

cur_step = 0

epochs_no_improve=0

training_losses = []

validation_losses = []

for epoch in range(max_epochs):

    mean_train_loss=0

    mean_val_loss=0

    for batch_index in tqdm(range(len(batches))):

        batch = batches[batch_index]

        val_batch = val_batches[batch_index]

        train_image_list = [transform(Image.open('/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/'+im_id)) for im_id in train_ids[batch_index]]

        images = torch.stack(train_image_list).to(device)

        classif_opt.zero_grad()

        training_pred = classif(images).float()

        training_loss = classif_loss(training_pred,batch)

        training_loss.backward()



        # Update the weights

        classif_opt.step()

        

        #Estou calculando a validacao por batch para economizar memoria

        val_image_list = [transform(Image.open('/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/'+im_id)) for im_id in val_ids[batch_index]]

        val_images = torch.stack(val_image_list).to(device)

        val_pred = classif(val_images).float()

        val_loss = classif_loss(val_pred,val_batch)

        # Keep track of the average losses

        mean_train_loss += training_loss.item()/len(batches)

        mean_val_loss += val_loss.item()/len(batches)

    training_losses += [mean_train_loss]

    validation_losses += [mean_val_loss]

    ### Visualization code ###

    if cur_step % display_step == 0 and cur_step > 0:

        training_mean = sum(training_losses[-display_step:]) / display_step

        step_bins = 20

        num_examples = (len(training_losses) // step_bins) * step_bins

        plt.plot(

            range(num_examples // step_bins), 

            torch.Tensor(training_losses[:num_examples]).view(-1, step_bins).mean(1),

            label="Training Loss"

        )

        plt.plot(

            range(num_examples // step_bins), 

            torch.Tensor(validation_losses[:num_examples]).view(-1, step_bins).mean(1),

            label="Validation Loss"

        )

        plt.legend()

        plt.show()

    if(mean_val_loss.item() < min_val_loss):

        print('got here!')

        epochs_no_improve=0

        min_val_loss = mean_val_loss.item()

    else:

        print('validation didn\'t improve')

        print(mean_train_loss.item(),mean_val_loss.item())

        epochs_no_improve+=1

        if(epochs_no_improve>n_epochs_stop):

            print('Early stopping!' )

            break;

    cur_step += 1

classif.eval()
#test_images = torch.stack([transform(Image.open('/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba/'+im_id)) for im_id in test.iloc[:,0]])

#test_pred = classif(test_images).float()

#test_loss = classif_loss(test_pred,torch.Tensor(test.iloc[:,1:].values).long())

#print(test_loss.item())