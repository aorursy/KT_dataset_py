import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt

from torchvision import datasets

from torchvision.transforms import transforms

from torch.utils.data import DataLoader

%matplotlib inline

plt.rcParams['image.interpolation'] = 'nearest'

from IPython.display import clear_output
import os

import PIL

import xml.etree.ElementTree as ET

   

def get_all_paths(root_directory):

    list_all_paths = []

    for root, dirs,filenames in os.walk(root_directory):

        for f in filenames:

            list_all_paths.append(os.path.abspath(os.path.join(root, f)))

    return list_all_paths



def make_folders(path_folder):

    if not os.path.exists(path_folder):

        os.makedirs(path_folder)

        print("Folder: " + path_folder + " created")

        

def dogs_only_from_annotations(path_to_data_folder, path_to_annotation_folder, path_to_dogs_folder):

    exceptions = {}

    annotation_directories = get_sub_directories(path_to_annotation_folder)

    for subdir in annotation_directories:

        clear_output()

        print("Processing Directory: " + subdir)

        make_folders(os.path.join(path_to_dogs_folder, subdir))

        files = get_files_in_sub_directory(path_to_annotation_folder, subdir)

        for f in files:

            # No .xml extention (here plain text), good for open image project.

            basename =  os.path.splitext(f)[0] 

            try:

                objects = get_all_objects(os.path.join(path_to_annotation_folder, subdir, basename))

                for i, obj in enumerate(objects):

                    xmin, ymin, xmax, ymax = obj

                    image = PIL.Image.open(os.path.join(path_to_data_folder, basename + ".jpg"))

                    cropped = image.crop((xmin, ymin, xmax, 

                                          ymax)).save(os.path.join(path_to_dogs_folder, subdir, 

                                                                   "cropped_" + basename + 

                                                                   str(i) + ".jpg"),

                                                      "JPEG")

            except Exception as e:

                exceptions[str(e)] = os.path.join(path_to_annotation_folder, subdir, basename)

    return exceptions



def parse_annotation_file(file_name):

    root = ET.parse(file_name).getroot()

    for child in root:

        print(child.tag, child.attrib)

        

def print_annotation_file(file_name):

    root = ET.parse(file_name).getroot()        

    print(ET.tostring(root, encoding='utf8').decode('utf8'))

    

def get_all_objects(file_path):

    bbxs = []

    root = ET.parse(file_path).getroot()

    for obj in root.findall("object"):

        bndbox = obj.find("bndbox")

        bbxs.append([int(it.text) for it in bndbox])

    return bbxs



def get_sub_directories(directory):

    return sorted([name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))])



def get_files_in_sub_directory(parent_directory, sub_directory):

    return os.listdir(os.path.join(parent_directory, sub_directory))



def get_images(imgs_folder, i, j):

    imgs = os.listdir(imgs_folder)[i:j]

    f, axes = plt.subplots(nrows=3, ncols=len(imgs) // 3 + 1, figsize=(20, 5))

    axes = axes.ravel()

    for ax in axes:

        ax.axis('off')

    for i, title in enumerate(imgs):

        image = PIL.Image.open(os.path.join(imgs_folder, title))

        axes[i].imshow(image, cmap='gray')

        axes[i].set_title(title)  

    plt.tight_layout()

    plt.show()
data_folder = "../input/all-dogs/all-dogs"

annotation_folder = "../input/annotation/Annotation"
get_images(data_folder, 600, 620)
dogs_folder = "../Dogs-Cropped-Images"

out_folder = "../output_images"

make_folders(dogs_folder)

make_folders(out_folder)
os.listdir("../")
exceptions = dogs_only_from_annotations(data_folder, annotation_folder, dogs_folder)
batch_size = 128

z_size = 100
def get_dataloader(batch_size, image_size, data_dir='../Dogs-Cropped-Images'):

    """

    Batch the neural network data using DataLoader

    :param batch_size: The size of each batch; the number of images in a batch

    :param img_size: The square size of the image data (x, y)

    :param data_dir: Directory where image data is located

    :return: DataLoader with batched data

    """

    transform = transforms.Compose([transforms.Resize((image_size, image_size)), 

                                transforms.ToTensor(), 

                                transforms.Normalize([0.5, 0.5, 0.5],

                                                    [0.5, 0.5, 0.5])])

                              

                                

    dataset=datasets.ImageFolder(data_dir,transform)

    dataloader=torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size)

    # TODO: Implement function and return a dataloader

    

    return dataloader

# Define function hyperparameters

 

img_size = 64



"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""

# Call your function and get a dataloader

celeba_train_loader = get_dataloader(batch_size, img_size)

# helper display function

def imshow(img):

    npimg = img.numpy()

    npimg = npimg *0.5

    npimg +=0.5

    plt.imshow(np.transpose(npimg, (1, 2, 0)))



"""

DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE

"""

# obtain one batch of training images

dataiter = iter(celeba_train_loader)

images, _ = dataiter.next() # _ for no labels



# plot the images in the batch, along with the corresponding labels

fig = plt.figure(figsize=(20, 4))

plot_size=20

for idx in np.arange(plot_size):

    ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])

    imshow(images[idx])
def init_weight(m):

  classname = m.__class__.__name__

  if classname.find('Conv') != -1:

    m.weight.data.normal_(0.0, 0.02)

  elif classname.find('BatchNorm') != -1:

    m.weight.data.normal_(0.1, 0.02)

    m.bias.data.fill_(0)
def mila(input,beta=1.0):

    return input * torch.tanh(F.softplus(input+beta))
# Discriminator



# Probably a VGG16 or VGG19 for Simple Image Classification pretrained on ImageNet



class Discriminator(nn.Module):

    

    def __init__(self, inhw, c1_channels=64, c2_channels=128, c3_channels=256,

                 c4_channels=512, i_channels_in_2=True):

        '''

        The constructor method for the Discriminator class

        

        Arguments:

        - inhw : The number of 

        - c1_channels : the number of output channels from the

                        first Convolutional Layer [Default - 128]

                        

        - c2_channels : the number of output channels from the

                        second Convolutional Layer [Default - 256]

                        

        - c3_channels : the number of output channels from the

                        third Convolutional Layer [Default - 512]

        

        - i_channels_in_2 : Increase the number of channels by 2

                        in each layer.

        '''

        

        super().__init__()

        

        # Define the class variables

        self.c1_channels = c1_channels

        

        if i_channels_in_2:

            self.c2_channels = self.c1_channels * 2

            self.c3_channels = self.c2_channels * 2

            self.c4_channels = self.c3_channels * 2

        else:

            self.c2_channels = c2_channels

            self.c3_channels = c3_channels

            self.c4_channels = c4_channels

        

        self.conv1 = nn.Conv2d(in_channels=3,

                               out_channels=self.c1_channels,

                               kernel_size=4,

                               stride=2,

                               padding=1,

                               bias=False)

        

        self.conv2 = nn.Conv2d(in_channels=self.c1_channels,

                               out_channels=self.c2_channels,

                               kernel_size=4,

                               stride=2,

                               padding=1,

                               bias=False)

        

        self.bnorm2 = nn.BatchNorm2d(num_features=self.c2_channels)

        

        self.conv3 = nn.Conv2d(in_channels=self.c2_channels,

                               out_channels=self.c3_channels,

                               kernel_size=4,

                               stride=2,

                               padding=1,

                               bias=False)

        

        self.bnorm3 = nn.BatchNorm2d(num_features=self.c3_channels)

        

        self.conv4 = nn.Conv2d(in_channels=self.c3_channels,

                               out_channels=self.c4_channels,

                               kernel_size=4,

                               stride=2,

                               padding=1,

                               bias=False)

        

        self.bnorm4 = nn.BatchNorm2d(num_features=self.c4_channels)

        

        self.conv5 = nn.Conv2d(in_channels=self.c4_channels,

                               out_channels=1,

                               kernel_size=4,

                               padding=0,

                               stride=1,

                               bias=False)

        

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        

        self.sigmoid = nn.Sigmoid()

        

        

    def forward(self, img):

        '''

        The method for the forward pass in the network

        

        Arguments;

        - img : a torch.tensor that is of the shape N x C x H x W

                where, N - the batch_size

                       C - the number of channels

                       H - the height

                       W - the width

       

       Returns:

       - out : the output of the Discriminator 

               whether the passed image is real /fake

        '''

        

        #print (img.shape)

        

        batch_size = img.shape[0]

        

        x = self.lrelu(self.conv1(img))

        x = self.lrelu(self.bnorm2(self.conv2(x)))

        x = self.lrelu(self.bnorm3(self.conv3(x)))

        x = self.lrelu(self.bnorm4(self.conv4(x)))

        x = self.conv5(x)

        

        x = self.sigmoid(x)

        

        return x.view(-1, 1).squeeze()

      

    def out_shape(self, inp_dim, kernel_size=4, padding=1, stride=2):

        return ((inp_dim - kernel_size + (2 * padding)) // stride) + 1
class Generator(nn.Module):

    def __init__(self, ct1_channels=512, ct2_channels=256,

                 ct3_channels=128, ct4_channels=64, d_channels_in_2=False):

        

        '''

        The contructor class for the Generator

        

        Arguments:

        - zin_channels: ###

        

        - ct1_channels: The number of output channels for the

                        first ConvTranspose Layer. [Default - 1024]

        

        - ct2_channels: The number of putput channels for the

                        second ConvTranspose Layer. [Default - 512]

                        

        - ct3_channels: The number of putput channels for the

                        third ConvTranspose Layer. [Default - 256]

                        

        - ct4_channels: The number of putput channels for the

                        fourth ConvTranspose Layer. [Default - 128]

                        

        - d_channnels_in_2 : Decrease the number of channels 

                        by 2 times in each layer.

                        

        '''

        super().__init__()

        

        # Define the class variables

        self.ct1_channels = ct1_channels

        self.pheight = 4

        self.pwidth = 4

        

        if d_channels_in_2:

            self.ct2_channels = self.ct1_channels // 2

            self.ct3_channels = self.ct2_channels // 2

            self.ct4_channels = self.ct3_channels // 2

        else:

            self.ct2_channels = ct2_channels

            self.ct3_channels = ct3_channels

            self.ct4_channels = ct4_channels

        

        self.convt_0 = nn.ConvTranspose2d(in_channels=z_size,

                                          out_channels=self.ct1_channels,

                                          kernel_size=4,

                                          padding=0,

                                          stride=1,

                                          bias=False)

        

        self.bnorm0 = nn.BatchNorm2d(self.ct1_channels)

        

        self.convt_1 = nn.ConvTranspose2d(in_channels=self.ct1_channels,

                                          out_channels=self.ct2_channels,

                                          kernel_size=4,

                                          stride=2,

                                          padding=1,

                                          bias=False)

        

        self.bnorm1 = nn.BatchNorm2d(num_features=self.ct2_channels)

        

        self.convt_2 = nn.ConvTranspose2d(in_channels=self.ct2_channels,

                                          out_channels=self.ct3_channels,

                                          kernel_size=4,

                                          stride=2,

                                          padding=1,

                                          bias=False)

        

        self.bnorm2 = nn.BatchNorm2d(num_features=self.ct3_channels)

        

        self.convt_3 = nn.ConvTranspose2d(in_channels=self.ct3_channels,

                                          out_channels=self.ct4_channels,

                                          kernel_size=4,

                                          stride=2,

                                          padding=1,

                                          bias=False)

        

        self.bnorm3 = nn.BatchNorm2d(num_features=self.ct4_channels)

        

        self.convt_4 = nn.ConvTranspose2d(in_channels=self.ct4_channels,

                                          out_channels=3,

                                          kernel_size=4,

                                          stride=2,

                                          padding=1,

                                          bias=False)

        

        self.relu = nn.ReLU()

        self.tanh = nn.Tanh()

        

    def forward(self, z):

        '''

        The method for the forward pass for the Generator

        

        Arguments:

        - z : the input random uniform vector sampled from uniform distribution

        

        Returns:

        - out : The output of the forward pass through the network

        '''

        

        # Project the input z and reshape

        x = self.relu(self.bnorm0(self.convt_0(z)))

        #print (x.shape)

        x = mila(self.bnorm1(self.convt_1(x)))

        x = mila(self.bnorm2(self.convt_2(x)))

        x = mila(self.bnorm3(self.convt_3(x)))

        out = self.tanh(self.convt_4(x))

        

        return out
dis = Discriminator(64).cuda()

dis.apply(init_weight)



gen = Generator().cuda()

gen.apply(init_weight)
print (dis)

print ()

print (gen)
criterion = nn.BCELoss()
# Optimizers

criterion = nn.BCELoss()

d_lr = 0.0002

g_lr = 0.0002



d_opt = optim.Adam(dis.parameters(), lr=d_lr, betas=[0.5, 0.999])

g_opt = optim.Adam(gen.parameters(), lr=g_lr, betas=[0.5, 0.999])
device=["cuda"]
# Train loop



p_every = 300

t_every = 1

e_every = 1

s_every = 1

epochs = 200



real_label = 0.9

fake_label = 0.1



train_losses = []

eval_losses = []

samples=[]

sample_size=16

fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))

fixed_z = torch.from_numpy(fixed_z).float()

fixed_z = fixed_z.cuda()



for e in range(epochs):

    

    td_loss = 0

    tg_loss = 0

    

    for batch_i, (real_images, _) in enumerate(celeba_train_loader):

        

        real_images = real_images.cuda()

        

        batch_size = real_images.size(0)



        #### Train the Discriminator ####



        d_opt.zero_grad()

		

        d_real = dis(real_images)

        

        label = torch.full((batch_size,), real_label, device='cuda')

        

        r_loss = criterion(d_real,label)

        r_loss.backward()





        z = torch.randn(batch_size, z_size, 1, 1, device='cuda')



        fake_images = gen(z)

        

        label.fill_(fake_label)

        

        d_fake = dis(fake_images.detach())

        

        f_loss = criterion(d_fake,label)

        f_loss.backward()



        d_loss = r_loss + f_loss



        d_opt.step()





        #### Train the Generator ####

        g_opt.zero_grad()

        

        label.fill_(real_label)

        

        d_fake2 = dis(fake_images)

        

        g_loss = criterion(d_fake2, label)

        g_loss.backward()

		

        g_opt.step()

        

        if batch_i % p_every == 0:

          noise = torch.randn(1, 100, 1, 1, device='cuda')

          out = gen(noise)

          out = out.detach().cpu().squeeze(0).transpose(0, 1).transpose(1, 2).numpy()

          out = out * (0.5, 0.5, 0.5)

          out += (0.5, 0.5, 0.5)

          plt.axis('off')

          plt.imshow(out)

          plt.show()

          print ('Epoch [{:5d} / {:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'. \

                    format(e+1, epochs, d_loss, g_loss))

            

train_losses.append([d_loss, g_loss])

    



  

        

print ('[INFO] Training Completed successfully!')

    

    # finally return losses

return train_losses



    # Save training generator samples

 

                

#torch.save(gen.state_dict(),'dog_generator.pt')
#gen.load_state_dict(torch.load('dog_generator.pt'))
noise = torch.randn(1, 100, 1, 1, device='cuda')

out = gen(noise)

out = out.detach().cpu().squeeze(0).transpose(0, 1).transpose(1, 2).numpy()

out = out * (0.5, 0.5, 0.5)

out += (0.5, 0.5, 0.5)

plt.axis('off')

plt.imshow(out)

plt.show()
from torchvision.utils import save_image

n_images=10000

im_batch_size=50



for i_batch in range(0, n_images):

    gen_z = torch.randn(1, 100, 1, 1, device='cuda')

    gen_images = gen(gen_z)

    gen_images = gen_images.to("cpu").clone().detach()

    gen_images.numpy().transpose(0, 2, 3, 1)

    gen_images = gen_images * (0.5)

    gen_images += (0.5)

    for i_image in range(gen_images.size(0)):

        save_image(gen_images, 

                   os.path.join(out_folder, f'image_{i_batch+i_image:05d}.png'))
get_images(out_folder, 0, 25)
len(os.listdir(out_folder))
import shutil

shutil.make_archive('images', 'zip', out_folder)