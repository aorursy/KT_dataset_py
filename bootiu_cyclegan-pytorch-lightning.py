!pip install /kaggle/input/pytorch-lightning/pytorch_lightning-1.0.2-py3-none-any.whl -q
%matplotlib inline

import os, glob, random

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image

import shutil



import torch

from torchvision.utils import make_grid

from torchvision import transforms

import torchvision.transforms.functional as TF

from torch import nn, optim

from torch.optim.lr_scheduler import CosineAnnealingLR

from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

from pytorch_lightning import Trainer

from pytorch_lightning.callbacks import EarlyStopping
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = True
class MyRotationTransform:

    """Rotate by one of the given angles."""



    def __init__(self, angles=[-90, 90], p=0.5):

        self.angles = angles

        self.p = p



    def __call__(self, x):

        a = random.random()

        if a < self.p:

            angle = random.choice(self.angles)

            augmented = TF.rotate(x, angle)

            return augmented

        else:

            return x



class ImageTransform:

    def __init__(self, img_size=256):

        self.transform = {

            'train': transforms.Compose([

                transforms.Resize((img_size, img_size)),

                transforms.RandomHorizontalFlip(),

                transforms.RandomVerticalFlip(),

                MyRotationTransform(),

                transforms.ToTensor(),

                transforms.Normalize(mean=[0.5], std=[0.5])

            ]),

            'test': transforms.Compose([

                transforms.Resize((img_size, img_size)),

                transforms.ToTensor(),

                transforms.Normalize(mean=[0.5], std=[0.5])

            ])}



    def __call__(self, img, phase='train'):

        img = self.transform[phase](img)



        return img





# Monet Dataset ---------------------------------------------------------------------------

class MonetDataset(Dataset):

    def __init__(self, base_img_paths, style_img_paths,  transform, phase='train'):

        self.base_img_paths = base_img_paths

        self.style_img_paths = style_img_paths

        self.transform = transform

        self.phase = phase



    def __len__(self):

        return min([len(self.base_img_paths), len(self.style_img_paths)])



    def __getitem__(self, idx):

        if self.phase == 'train':

            random.shuffle(self.base_img_paths)

            random.shuffle(self.style_img_paths)

        

        base_img_path = self.base_img_paths[idx]

        style_img_path = self.style_img_paths[idx]

        base_img = Image.open(base_img_path)

        style_img = Image.open(style_img_path)



        base_img = self.transform(base_img, self.phase)

        style_img = self.transform(style_img, self.phase)



        return base_img, style_img
# Data Module

class MonetDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, transform, batch_size, phase='train'):

        super(MonetDataModule, self).__init__()

        self.data_dir = data_dir

        self.transform = transform

        self.batch_size = batch_size

        self.phase = phase



    def prepare_data(self):

        self.base_img_paths = glob.glob(os.path.join(self.data_dir, 'photo_jpg', '*.jpg'))

        self.style_img_paths = glob.glob(os.path.join(self.data_dir, 'monet_jpg', '*.jpg'))



    def setup(self, stage=None):

        random.shuffle(self.base_img_paths)

        random.shuffle(self.style_img_paths)

        self.train_dataset = MonetDataset(self.base_img_paths, self.style_img_paths, self.transform, self.phase)



    def train_dataloader(self):

        return DataLoader(self.train_dataset,

                          batch_size=self.batch_size,

                          shuffle=True,

                          num_workers=2,

                          pin_memory=True)
# Sanity Check

data_dir = '../input/gan-getting-started'

transform = ImageTransform(img_size=256)

batch_size = 8



dm = MonetDataModule(data_dir, transform, batch_size, phase='test')

dm.prepare_data()

dm.setup()



dataloader = dm.train_dataloader()

base, style = next(iter(dataloader))



print('Input Shape {}, {}'.format(base.size(), style.size()))
temp = make_grid(base, nrow=4, padding=2).permute(1, 2, 0).detach().numpy()

temp = temp * 0.5 + 0.5

temp = temp * 255.0

temp = temp.astype(int)



fig = plt.figure(figsize=(18, 8), facecolor='w')

plt.imshow(temp)

plt.axis('off')

plt.title('Photo')

plt.show()
temp = make_grid(style, nrow=4, padding=2).permute(1, 2, 0).detach().numpy()

temp = temp * 0.5 + 0.5

temp = temp * 255.0

temp = temp.astype(int)



fig = plt.figure(figsize=(18, 8), facecolor='w')

plt.imshow(temp)

plt.axis('off')

plt.title('Monet Pictures')

plt.show()
class Upsample(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=True):

        super(Upsample, self).__init__()

        self.dropout = dropout

        self.block = nn.Sequential(

            nn.Upsample(scale_factor=2),

            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),

            nn.InstanceNorm2d(out_channels),

            nn.ReLU(inplace=True)

        )

        self.dropout_layer = nn.Dropout2d(0.2)



    def forward(self, x):

        x = self.block(x)

        if self.dropout:

            x = self.dropout_layer(x)



        return x





class Downsample(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):

        super(Downsample, self).__init__()



        self.block = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),

            nn.InstanceNorm2d(out_channels),

            nn.LeakyReLU(inplace=True)

        )



    def forward(self, x):

        x = self.block(x)

        return x





class CycleGAN_Unet_Generator(nn.Module):

    def __init__(self, filter=32):

        super(CycleGAN_Unet_Generator, self).__init__()

        self.downsample_1 = Downsample(3, filter)

        self.downsample_2 = Downsample(filter, filter * 2)

        self.downsample_3 = Downsample(filter * 2, filter * 4)

        self.downsample_4 = Downsample(filter * 4, filter * 8)



        self.upsample_1 = Upsample(filter * 8, filter * 4, dropout=True)

        self.upsample_2 = Upsample(filter * 8, filter * 2, dropout=True)

        self.upsample_3 = Upsample(filter * 4, filter, dropout=False)

        self.last = nn.Sequential(

            nn.Upsample(scale_factor=2),

            nn.Conv2d(filter * 2, 3, kernel_size=3, stride=1, padding=1),

            nn.Tanh()

        )



    def forward(self, x):

        d1 = self.downsample_1(x)

        d2 = self.downsample_2(d1)

        d3 = self.downsample_3(d2)

        d4 = self.downsample_4(d3)



        u1 = self.upsample_1(d4)

        u1 = torch.cat([u1, d3], dim=1)

        u2 = self.upsample_2(u1)

        u2 = torch.cat([u2, d2], dim=1)

        u3 = self.upsample_3(u2)

        u3 = torch.cat([u3, d1], dim=1)

        out = self.last(u3)



        return out





class ConvBatchRelu(nn.Module):

    def __init__(self, in_chennels=3, out_channels=128, kernel_size=3, stride=1, padding=1):

        super(ConvBatchRelu, self).__init__()

        self.block = nn.Sequential(

            nn.Conv2d(in_chennels, out_channels, kernel_size, stride, padding),

            nn.InstanceNorm2d(out_channels),

            nn.LeakyReLU(inplace=True)

        )



    def forward(self, x):

        x = self.block(x)

        return x





class CycleGAN_Discriminator(nn.Module):

    def __init__(self, filter=32):

        super(CycleGAN_Discriminator, self).__init__()



        self.block = nn.Sequential(

            ConvBatchRelu(3, filter, kernel_size=3, stride=2),

            ConvBatchRelu(filter, filter * 2, kernel_size=3, stride=2),

            ConvBatchRelu(filter * 2, filter * 2, kernel_size=3, stride=1),

            ConvBatchRelu(filter * 2, filter * 4, kernel_size=3, stride=2),

            ConvBatchRelu(filter * 4, filter * 4, kernel_size=3, stride=1),

            ConvBatchRelu(filter * 4, filter * 8, kernel_size=3, stride=2),

            ConvBatchRelu(filter * 8, filter * 16, kernel_size=3, stride=1)

        )



        self.last = nn.Conv2d(filter * 16, 1, kernel_size=3, stride=1, padding=1)



    def forward(self, x):

        x = self.block(x)

        x = self.last(x)



        return x
# Sanity Check

net = CycleGAN_Unet_Generator()



out = net(base)

print(out.size())
# Sanity Check

net = CycleGAN_Discriminator()



out = net(base)

print(out.size())
# CycleGAN - Lightning Module ---------------------------------------------------------------------------

class CycleGAN_LightningSystem(pl.LightningModule):

    def __init__(self, G_basestyle, G_stylebase, D_base, D_style, lr, transform):

        super(CycleGAN_LightningSystem, self).__init__()

        self.G_basestyle = G_basestyle

        self.G_stylebase = G_stylebase

        self.D_base = D_base

        self.D_style = D_style

        self.lr = lr

        self.transform = transform

        self.cnt_train_step = 0

        self.step = 0



        self.mse = nn.MSELoss()

        self.mae = nn.L1Loss()

        self.losses = []

        self.G_mean_losses = []

        self.D_mean_losses = []

        self.validity = []

        self.reconstr = []

        self.identity = []



    def configure_optimizers(self):

        self.g_basestyle_optimizer = optim.Adam(self.G_basestyle.parameters(), lr=self.lr['G'], betas=(0.5, 0.999))

        self.g_stylebase_optimizer = optim.Adam(self.G_stylebase.parameters(), lr=self.lr['G'], betas=(0.5, 0.999))

        self.d_base_optimizer = optim.Adam(self.D_base.parameters(), lr=self.lr['D'])

        self.d_style_optimizer = optim.Adam(self.D_style.parameters(), lr=self.lr['D'])



        return [self.g_basestyle_optimizer, self.g_stylebase_optimizer, self.d_base_optimizer, self.d_style_optimizer], []



    def training_step(self, batch, batch_idx, optimizer_idx):

        base_img, style_img = batch

        b = base_img.size()[0]



        valid = torch.ones(b, 1, 16, 16).cuda()

        fake = torch.zeros(b, 1, 16, 16).cuda()



        # Train Generator

        if optimizer_idx == 0 or optimizer_idx == 1:

            # Create Fake Photo from Monet

            fake_base = self.G_stylebase(style_img)

            # Create Fake Monet from Photo

            fake_style = self.G_basestyle(base_img)



            # Validity

            val_base = self.mse(self.D_base(fake_base), valid)

            val_style = self.mse(self.D_style(fake_style), valid)

            val_loss = (val_base + val_style) / 2



            # Reconstruction

            reconstr_base = self.mae(self.G_stylebase(fake_style), base_img)

            reconstr_style = self.mae(self.G_basestyle(fake_base), style_img)

            reconstr_loss = (reconstr_base + reconstr_style) / 2



            # Identity

            id_base = self.mae(self.G_stylebase(base_img), base_img)

            id_style = self.mae(self.G_basestyle(style_img), style_img)

            id_loss = (id_base + id_style) / 2



            # Loss Weight

            G_loss = val_loss + 10 * reconstr_loss + 2 * id_loss



            return {'loss': G_loss, 'validity': val_loss, 'reconstr': reconstr_loss, 'identity': id_loss}



        # Train Discriminator

        elif optimizer_idx == 2 or optimizer_idx == 3:

            # Create Fake Photo from Monet

            fake_base = self.G_stylebase(style_img)

            # Create Fake Monet from Photo

            fake_style = self.G_basestyle(base_img)

            # Validity

            val_base = self.mse(self.D_base(fake_base), fake)

            val_style = self.mse(self.D_style(fake_style), fake)

            val_loss = (val_base + val_style) / 2



            base_loss = self.mse(self.D_base(base_img), valid)

            style_loss = self.mse(self.D_style(style_img), valid)



            # Loss Weight

            D_loss = (val_loss + base_loss + style_loss) / 3



            # Count up

            self.cnt_train_step += 1



            return {'loss': D_loss}



    def training_epoch_end(self, outputs):

        self.step += 1

        

        avg_loss = 0

        G_mean_loss = 0

        D_mean_loss = 0

        validity = 0

        reconstr = 0

        identity = 0

        for i in range(4):

            avg_loss += torch.stack([x['loss'] for x in outputs[i]]).mean().item() / 4

            

        for i in [0, 1]:

            G_mean_loss += torch.stack([x['loss'] for x in outputs[i]]).mean().item() / 2

            validity += torch.stack([x['validity'] for x in outputs[i]]).mean().item() / 2

            reconstr += torch.stack([x['reconstr'] for x in outputs[i]]).mean().item() / 2

            identity += torch.stack([x['identity'] for x in outputs[i]]).mean().item() / 2

            

        for i in [2, 3]:

            D_mean_loss += torch.stack([x['loss'] for x in outputs[i]]).mean().item() / 2

            

        self.losses.append(avg_loss)

        self.G_mean_losses.append(G_mean_loss)

        self.D_mean_losses.append(D_mean_loss)

        self.validity.append(validity)

        self.reconstr.append(reconstr)

        self.identity.append(identity)

        

        if self.step % 20 == 0:

            # Display Model Output

            target_img_paths = glob.glob('../input/gan-getting-started/photo_jpg/*.jpg')[:4]

            target_imgs = [self.transform(Image.open(path), phase='test') for path in target_img_paths]

            target_imgs = torch.stack(target_imgs, dim=0).cuda()



            gen_imgs = self.G_basestyle(target_imgs)

            gen_img = torch.cat([target_imgs, gen_imgs], dim=0)



            # Reverse Normalization

            gen_img = gen_img * 0.5 + 0.5

            gen_img = gen_img * 255



            joined_images_tensor = make_grid(gen_img, nrow=4, padding=2)



            joined_images = joined_images_tensor.detach().cpu().numpy().astype(int)

            joined_images = np.transpose(joined_images, [1,2,0])



            # Visualize

            fig = plt.figure(figsize=(18, 8))

            plt.imshow(joined_images)

            plt.axis('off')

            plt.title(f'Epoch {self.step}')

            plt.show()

            plt.clf()

            plt.close()



        return None
def init_weights(net, init_type='normal', init_gain=0.02):

    """Initialize network weights.

    Parameters:

        net (network)   -- network to be initialized

        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal

        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might

    work better for some applications. Feel free to try yourself.

    """

    def init_func(m):  # define the initialization function

        classname = m.__class__.__name__

        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):

            if init_type == 'normal':

                nn.init.normal_(m.weight.data, 0.0, init_gain)

            elif init_type == 'xavier':

                nn.init.xavier_normal_(m.weight.data, gain=init_gain)

            elif init_type == 'kaiming':

                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            elif init_type == 'orthogonal':

                nn.init.orthogonal_(m.weight.data, gain=init_gain)

            else:

                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

            if hasattr(m, 'bias') and m.bias is not None:

                nn.init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.

            nn.init.normal_(m.weight.data, 1.0, init_gain)

            nn.init.constant_(m.bias.data, 0.0)



    net.apply(init_func)  # apply the initialization function <init_func>
data_dir = '../input/gan-getting-started'

transform = ImageTransform(img_size=256)

batch_size = 30

lr = {

    'G': 0.0002,

    'D': 0.0002

}

epoch = 200

seed = 42

seed_everything(seed)



dm = MonetDataModule(data_dir, transform, batch_size)



G_basestyle = CycleGAN_Unet_Generator()

G_stylebase = CycleGAN_Unet_Generator()

D_base = CycleGAN_Discriminator()

D_style = CycleGAN_Discriminator()



# Init Weight

for net in [G_basestyle, G_stylebase, D_base, D_style]:

    init_weights(net, init_type='normal')



model = CycleGAN_LightningSystem(G_basestyle, G_stylebase, D_base, D_style, lr, transform)



trainer = Trainer(

    logger=False,

    max_epochs=epoch,

    gpus=1,

    checkpoint_callback=False

#     num_sanity_val_steps=0,  # Skip Sanity Check

)





# Train

trainer.fit(model, datamodule=dm)
def submit(model, transform):

    os.makedirs('../images', exist_ok=True)

    net = model.G_basestyle

    

    net.eval()

    photo_img_paths = glob.glob('../input/gan-getting-started/photo_jpg/*.jpg')

    

    for path in photo_img_paths:

        photo_id = path.split('/')[-1]

        img = transform(Image.open(path), phase='test')

        

        gen_img = net(img.unsqueeze(0))[0]

        

        # Reverse Normalization

        gen_img = gen_img * 0.5 + 0.5

        gen_img = gen_img * 255

        gen_img = gen_img.detach().cpu().numpy().astype(np.uint8)

        

        gen_img = np.transpose(gen_img, [1,2,0])

        

        gen_img = Image.fromarray(gen_img)

        gen_img.save(os.path.join('../images', photo_id))

        

        

    import shutil

    shutil.make_archive("/kaggle/working/images", 'zip', "/kaggle/images")

    

    # Delete Origin file

    shutil.rmtree('../images')
submit(model, transform)
losses = model.losses

G_losses = model.G_mean_losses

D_losses = model.D_mean_losses

validity = model.validity

reconstr = model.reconstr

identity = model.identity



fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(18, 12), facecolor='w')



axes[0].plot(np.arange(len(losses)), losses, label='overall')

axes[0].plot(np.arange(len(losses)), G_losses, label='generator')

axes[0].plot(np.arange(len(losses)), D_losses, label='discriminator')

axes[0].legend()

axes[0].set_xlabel('Epoch')



axes[1].plot(np.arange(len(losses)), validity, label='validity')

axes[1].plot(np.arange(len(losses)), reconstr, label='reconstr')

axes[1].plot(np.arange(len(losses)), identity, label='identity')

axes[1].legend()

axes[1].set_xlabel('Epoch')



plt.show()