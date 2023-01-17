import os, time

from PIL import Image, ImageOps

import numpy as np



import torch

import torch.nn as nn

from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F

from torchvision import transforms

import torch.optim as optim



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

lr = 0.0002

L1_lambda = 100

beta1 = 0.5

beta2 = 0.999

inverse_order = True

img_size = 256



train_epoch = 150



save_root = './results'

if not os.path.isdir(save_root):

    os.mkdir(save_root)
class generator(nn.Module):

    # initializers

    def __init__(self, d=64):

        super(generator, self).__init__()

        # Unet encoder

        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)

        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)

        self.conv2_bn = nn.BatchNorm2d(d * 2)

        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)

        self.conv3_bn = nn.BatchNorm2d(d * 4)

        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)

        self.conv4_bn = nn.BatchNorm2d(d * 8)

        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)

        self.conv5_bn = nn.BatchNorm2d(d * 8)

        self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)

        self.conv6_bn = nn.BatchNorm2d(d * 8)

        self.conv7 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)

        self.conv7_bn = nn.BatchNorm2d(d * 8)

        self.conv8 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)

        # self.conv8_bn = nn.BatchNorm2d(d * 8)



        # Unet decoder

        self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)

        self.deconv1_bn = nn.BatchNorm2d(d * 8)

        self.deconv2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)

        self.deconv2_bn = nn.BatchNorm2d(d * 8)

        self.deconv3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)

        self.deconv3_bn = nn.BatchNorm2d(d * 8)

        self.deconv4 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)

        self.deconv4_bn = nn.BatchNorm2d(d * 8)

        self.deconv5 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1)

        self.deconv5_bn = nn.BatchNorm2d(d * 4)

        self.deconv6 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1)

        self.deconv6_bn = nn.BatchNorm2d(d * 2)

        self.deconv7 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)

        self.deconv7_bn = nn.BatchNorm2d(d)

        self.deconv8 = nn.ConvTranspose2d(d * 2, 3, 4, 2, 1)

        self.tanh = nn.Tanh()



    # weight_init

    def weight_init(self, mean, std):

        for m in self._modules:

            normal_init(self._modules[m], mean, std)



    # forward method

    def forward(self, input):

        e1 = self.conv1(input)

        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))

        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))

        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))

        e5 = self.conv5_bn(self.conv5(F.leaky_relu(e4, 0.2)))

        e6 = self.conv6_bn(self.conv6(F.leaky_relu(e5, 0.2)))

        e7 = self.conv7_bn(self.conv7(F.leaky_relu(e6, 0.2)))

        e8 = self.conv8(F.leaky_relu(e7, 0.2))

        # e8 = self.conv8_bn(self.conv8(F.leaky_relu(e7, 0.2)))

        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e8))), 0.5, training=True)

        d1 = torch.cat([d1, e7], 1)

        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(d1))), 0.5, training=True)

        d2 = torch.cat([d2, e6], 1)

        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(d2))), 0.5, training=True)

        d3 = torch.cat([d3, e5], 1)

        d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))

        # d4 = F.dropout(self.deconv4_bn(self.deconv4(F.relu(d3))), 0.5)

        d4 = torch.cat([d4, e4], 1)

        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))

        d5 = torch.cat([d5, e3], 1)

        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))

        d6 = torch.cat([d6, e2], 1)

        d7 = self.deconv7_bn(self.deconv7(F.relu(d6)))

        d7 = torch.cat([d7, e1], 1)

        d8 = self.deconv8(F.relu(d7))

        o = self.tanh(d8)



        return o



class discriminator(nn.Module):

    # initializers

    def __init__(self, d=64):

        super(discriminator, self).__init__()

        self.conv1 = nn.Conv2d(6, d, 4, 2, 1)

        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)

        self.conv2_bn = nn.BatchNorm2d(d * 2)

        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)

        self.conv3_bn = nn.BatchNorm2d(d * 4)

        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)

        self.conv4_bn = nn.BatchNorm2d(d * 8)

        self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)

        self.sigmoid = nn.Sigmoid()



    # weight_init

    def weight_init(self, mean, std):

        for m in self._modules:

            normal_init(self._modules[m], mean, std)



    # forward method

    def forward(self, input, label):

        x = torch.cat([input, label], 1)

        x = F.leaky_relu(self.conv1(x), 0.2)

        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)

        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)

        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)

        x = self.sigmoid(self.conv5(x))



        return x



def normal_init(m, mean, std):

    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):

        m.weight.data.normal_(mean, std)

        m.bias.data.zero_()
train_img_nms = os.listdir('/kaggle/input/cityscapes-image-pairs/cityscapes_data/cityscapes_data/train')

val_img_nms = os.listdir('/kaggle/input/cityscapes-image-pairs/cityscapes_data/cityscapes_data/val')



# tmp_img = Image.open('../input/cityscapes-image-pairs/cityscapes_data/train/1.jpg')

# img = tmp_img.crop((0, 0, 256, 256))

# seg = tmp_img.crop((256, 0, 512, 256))

# tmp_img
data_root_dir = '/kaggle/input/cityscapes-image-pairs/cityscapes_data/cityscapes_data/'

class CityscapesDataset(Dataset):

    def __init__(self, img_nms, data_root_dir, split, transforms=None, augm=True):

        super().__init__()

        self.img_nms = img_nms

        self.data_root_dir = data_root_dir

        self.transforms = transforms

        self.split = split

        self.augm = augm

        

    def __len__(self):

        return len(self.img_nms)

        

    def __getitem__(self, idx):

        tmp_img = Image.open(os.path.join(self.data_root_dir, self.split, self.img_nms[idx]))

        img = tmp_img.crop((0, 0, 256, 256))

        seg = tmp_img.crop((256, 0, 512, 256))

        if self.augm:

            img, seg = self._random_crop(img, seg)

            img, seg = self._random_fliph(img, seg)

        if self.transforms:

            img = self.transforms(img)

            seg = self.transforms(seg)

        return seg, img

    

    def _random_crop(self, img, seg, crop_size=(224, 224)): # inputs are PIL images

        crop = np.random.rand() > 0.1

        if crop:

            w, h = img.size

            new_h, new_w = crop_size



            top = np.random.randint(0, h - new_h)

            left = np.random.randint(0, w - new_w)



            img = img.crop((left, top, left + new_w, top + new_h))

            seg = seg.crop((left, top, left + new_w, top + new_h))

            return img, seg

        else:

            return img, seg

    

    def _random_fliph(self, img, seg): # inputs are PIL images

        flip = np.random.rand() > 0.5

        if flip:

            return ImageOps.mirror(img), ImageOps.mirror(seg)

        else:

            return img, seg
transform = transforms.Compose(

    [

        transforms.Resize((img_size, img_size)),

        transforms.CenterCrop(img_size),

        transforms.ToTensor(),

    ]

)
def get_concat_h(im1, im2):

    dst = Image.new('RGB', (im1.width + im2.width, im1.height))

    dst.paste(im1, (0, 0))

    dst.paste(im2, (im1.width, 0))

    return dst
# val img

val_dataset = CityscapesDataset(['1.jpg'], data_root_dir, 'val', transform, augm=False)

val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

fixed_x_, fixed_y_ = iter(val_dataloader).next()

to_pil = transforms.ToPILImage()

seg_pil = to_pil(fixed_x_.to('cpu')[0])

img_pil = to_pil(fixed_y_.to('cpu')[0])

display(get_concat_h(img_pil,seg_pil))

fixed_x_, fixed_y_ = Variable(fixed_x_.to(device)), Variable(fixed_y_.to(device))
train_dataset = CityscapesDataset(train_img_nms, data_root_dir, 'train', transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)



# network

G = generator()

D = discriminator()

G.weight_init(mean=0.0, std=0.02)

D.weight_init(mean=0.0, std=0.02)

G = G.to(device)

D = D.to(device)



# loss

BCE_loss = nn.BCELoss().to(device)

L1_loss = nn.L1Loss().to(device)



# Adam optimizer

G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(beta1, beta2))

D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(beta1, beta2))



train_hist = {}

train_hist['D_losses'] = []

train_hist['G_losses'] = []

train_hist['D_result'] = []

train_hist['per_epoch_ptimes'] = []

train_hist['total_ptime'] = []
print('training start!')

start_time = time.time()

for epoch in range(train_epoch):

    D_losses = []

    G_losses = []

    epoch_start_time = time.time()

    num_iter = 0

    G.train()

    D.train()

    for x_, y_ in train_dataloader:

        # train discriminator D

        D.zero_grad()



        x_, y_ = Variable(x_.to(device)), Variable(y_.to(device))



        D_result = D(x_, y_).squeeze()

        D_real_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).to(device)))



        G_result = G(x_)

        D_result = D(x_, G_result).squeeze()

        D_fake_loss = BCE_loss(D_result, Variable(torch.zeros(D_result.size()).to(device)))



        D_train_loss = (D_real_loss + D_fake_loss) * 0.5

        D_train_loss.backward()

        D_optimizer.step()



        train_hist['D_losses'].append(D_train_loss.item())



        D_losses.append(D_train_loss.item())



        # train generator G

        G.zero_grad()



        G_result = G(x_)

        D_result = D(x_, G_result).squeeze()

        train_hist['D_result'].append(D_result.mean().item())



        G_train_loss = BCE_loss(D_result, Variable(torch.ones(D_result.size()).to(device))) + L1_lambda * L1_loss(G_result, y_)

        G_train_loss.backward()

        G_optimizer.step()



        train_hist['G_losses'].append(G_train_loss.item())



        G_losses.append(G_train_loss.item())



        num_iter += 1



    epoch_end_time = time.time()

    per_epoch_ptime = epoch_end_time - epoch_start_time



    print('[%d/%d] - ptime: %.2f, loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, torch.mean(torch.FloatTensor(D_losses)),

                                                              torch.mean(torch.FloatTensor(G_losses))))

    if epoch % 10 == 0:

        G.eval()

        G_fixed_result = G(fixed_x_)

        results_img = get_concat_h(to_pil(fixed_y_.to('cpu')[0]), to_pil(G_fixed_result.to('cpu')[0]))

        results_img = get_concat_h(results_img, to_pil(fixed_x_.to('cpu')[0]))

#         display(results_img)

        fixed_p = os.path.join(save_root, str(epoch + 1).zfill(4) + '.png')

        results_img.save(fixed_p)

    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)



end_time = time.time()

total_ptime = end_time - start_time

train_hist['total_ptime'].append(total_ptime)

print('minute', total_ptime / 60)
if not os.path.isdir('./model'):

    os.mkdir('./model')

G_model_path = f'./model/G_{epoch}.pth'

torch.save(G.state_dict(), G_model_path)

D_model_path = f'./model/D_{epoch}.pth'

torch.save(D.state_dict(), D_model_path)
import matplotlib.pyplot as plt

plt.plot(list(range(len(train_hist['D_losses']))), train_hist['D_losses'], label="D_losses")

plt.legend()
plt.plot(list(range(len(train_hist['D_losses']))), train_hist['G_losses'], label="G_losses")

plt.legend()
plt.plot(list(range(len(train_hist['D_losses']))), train_hist['D_result'], label="D_resultb")

plt.legend()