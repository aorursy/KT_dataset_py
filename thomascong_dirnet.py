# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

import torch.nn as nn

import os

import skimage.io

import torch.nn.functional as F

# from models import DIRNet

# from config import get_config

# from data import MNISTDataHandler

# from ops import mkdir

from torch.utils.data import DataLoader

from torchvision.datasets import MNIST

import torchvision.transforms as tf

from torch.optim.lr_scheduler import StepLR

import numpy as np

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
! ls ../
class Config(object):

  pass



def get_config(is_train):

  config = Config()

  if is_train:

    config.batch_size = 64

    config.im_size = [16, 16]

    config.lr = 1e-4

    config.iteration = 10000



    config.tmp_dir = "../tmp"

    config.ckpt_dir = "../ckpt"

  else:

    config.batch_size = 10

    config.im_size = [16, 16]



    config.result_dir = "../result"

    config.ckpt_dir = "../ckpt"

  return config
def mse(x, y):

    print('x is ',x.size())

    print('y is ',y.size())

    batch_size = x.size(0)

    return ((x - y) ** 2).sum() / batch_size



def mkdir(dir_path):

  try :

    os.makedirs(dir_path)

  except: pass 



def save_image_with_scale(path, arr):

  arr = np.clip(arr, 0., 1.)

  arr = arr * 255.

  arr = arr.astype(np.uint8)

  skimage.io.imsave(path, arr)







def conv2d(in_channels, out_channels, kernel_size, stride=1,

           padding=0, dilation=1, groups=1,

           bias=True, padding_mode='zeros',

           gain=1., bias_init=0.):

  m = nn.Conv2d(in_channels, out_channels, kernel_size, stride,

                padding, dilation, groups, bias, padding_mode)



  nn.init.orthogonal_(m.weight, gain)

  if bias:

    nn.init.constant_(m.bias, bias_init)



  return m







class Conv2dBlock(nn.Module):



  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):

    super().__init__()



    self.m = conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)



  def forward(self, x):

    x = F.elu(self.m(x))

    return F.layer_norm(x, x.size()[1:])
class MNISTDataHandler(object):

  """

    Members :

      is_train - Options for sampling

      path - MNIST data path

      data - a list of np.array w/ shape [batch_size, 28, 28, 1]

  """

  def __init__(self, digit):

    self.data = digit



  def sample_pair(self, batch_size, label=None):

    label = np.random.randint(10) if label is None else label

    images = self.data[label]

    

    choice1 = np.random.choice(images.shape[0], batch_size)

    choice2 = np.random.choice(images.shape[0], batch_size)

    x = images[choice1]

    y = images[choice2]



    return x, y
def bicubic_interp_2d(input_, new_size):

    shape = input_.size()

    batch_size = shape[0]

    height  = shape[1]

    width   = shape[2]

    channel = shape[3]

 

    def _hermite(A, B, C, D, t):

        a = A * -0.5 + B * 1.5 + C * -1.5 + D * 0.5

        b = A + B * -2.5 + C * 2.0 + D * -0.5

        c = A * -0.5 + C * 0.5

        d = B

        t = t.type_as(A)



        return a*t*t*t + b*t*t + c*t + d



    def _get_grid_array(n_i, y_i, x_i, c_i):

        n, y, x, c = np.meshgrid(n_i, y_i, x_i, c_i, indexing='ij')

        n = np.expand_dims(n, axis=4)

        y = np.expand_dims(y, axis=4)

        x = np.expand_dims(x, axis=4)

        c = np.expand_dims(c, axis=4)

        return np.concatenate([n,y,x,c], axis=4)



    def _get_frac_array(x_d, y_d, n, c):

        x = x_d.shape[0]

        y = y_d.shape[0]

        x_t = x_d.reshape([1, 1, -1, 1])

        y_t = y_d.reshape([1, -1, 1, 1])

        y_t = np.tile(y_t, (n,1,x,c))

        x_t = np.tile(x_t, (n,y,1,c))

        return x_t, y_t



    def _get_index_tensor(grid, x, y):

        new_grid = np.array(grid)



        grid_y = grid[:,:,:,:,1] + y

        grid_x = grid[:,:,:,:,2] + x

        grid_y = np.clip(grid_y, 0, height-1)

        grid_x = np.clip(grid_x, 0, width-1)



        new_grid[:,:,:,:,1] = grid_y

        new_grid[:,:,:,:,2] = grid_x



        return torch.LongTensor(new_grid)



    new_height = new_size[0]

    new_width  = new_size[1]



    n_i = np.arange(batch_size)

    c_i = np.arange(channel)



    y_f = np.linspace(0., height-1, new_height)

    y_i = y_f.astype(np.int32)

    y_d = y_f - np.floor(y_f)



    x_f = np.linspace(0., width-1, new_width)

    x_i = x_f.astype(np.int32)

    x_d = x_f - np.floor(x_f)



    grid = _get_grid_array(n_i, y_i, x_i, c_i)

    x_t, y_t = _get_frac_array(x_d, y_d, batch_size, channel)



    i_00 = _get_index_tensor(grid, -1, -1)

    i_10 = _get_index_tensor(grid, +0, -1)

    i_20 = _get_index_tensor(grid, +1, -1)

    i_30 = _get_index_tensor(grid, +2, -1)



    i_01 = _get_index_tensor(grid, -1, +0)

    i_11 = _get_index_tensor(grid, +0, +0)

    i_21 = _get_index_tensor(grid, +1, +0)

    i_31 = _get_index_tensor(grid, +2, +0)



    i_02 = _get_index_tensor(grid, -1, +1)

    i_12 = _get_index_tensor(grid, +0, +1)

    i_22 = _get_index_tensor(grid, +1, +1)

    i_32 = _get_index_tensor(grid, +2, +1)



    i_03 = _get_index_tensor(grid, -1, +2)

    i_13 = _get_index_tensor(grid, +0, +2)

    i_23 = _get_index_tensor(grid, +1, +2)

    i_33 = _get_index_tensor(grid, +2, +2)



    p_00 = input_[i_00[:, :, :, :, 0], i_00[:, :, :, :, 1], i_00[:, :, :, :, 2], i_00[:, :, :, :, 3]]

    p_10 = input_[i_10[:, :, :, :, 0], i_10[:, :, :, :, 1], i_10[:, :, :, :, 2], i_10[:, :, :, :, 3]]

    p_20 = input_[i_20[:, :, :, :, 0], i_20[:, :, :, :, 1], i_20[:, :, :, :, 2], i_20[:, :, :, :, 3]]

    p_30 = input_[i_30[:, :, :, :, 0], i_30[:, :, :, :, 1], i_30[:, :, :, :, 2], i_30[:, :, :, :, 3]]



    p_01 = input_[i_01[:, :, :, :, 0], i_01[:, :, :, :, 1], i_01[:, :, :, :, 2], i_01[:, :, :, :, 3]]

    p_11 = input_[i_11[:, :, :, :, 0], i_11[:, :, :, :, 1], i_11[:, :, :, :, 2], i_11[:, :, :, :, 3]]

    p_21 = input_[i_21[:, :, :, :, 0], i_21[:, :, :, :, 1], i_21[:, :, :, :, 2], i_21[:, :, :, :, 3]]

    p_31 = input_[i_31[:, :, :, :, 0], i_31[:, :, :, :, 1], i_31[:, :, :, :, 2], i_31[:, :, :, :, 3]]



    p_02 = input_[i_02[:, :, :, :, 0], i_02[:, :, :, :, 1], i_02[:, :, :, :, 2], i_02[:, :, :, :, 3]]

    p_12 = input_[i_12[:, :, :, :, 0], i_12[:, :, :, :, 1], i_12[:, :, :, :, 2], i_12[:, :, :, :, 3]]

    p_22 = input_[i_22[:, :, :, :, 0], i_22[:, :, :, :, 1], i_22[:, :, :, :, 2], i_22[:, :, :, :, 3]]

    p_32 = input_[i_32[:, :, :, :, 0], i_32[:, :, :, :, 1], i_32[:, :, :, :, 2], i_32[:, :, :, :, 3]]



    p_03 = input_[i_03[:, :, :, :, 0], i_03[:, :, :, :, 1], i_03[:, :, :, :, 2], i_03[:, :, :, :, 3]]

    p_13 = input_[i_13[:, :, :, :, 0], i_13[:, :, :, :, 1], i_13[:, :, :, :, 2], i_13[:, :, :, :, 3]]

    p_23 = input_[i_23[:, :, :, :, 0], i_23[:, :, :, :, 1], i_23[:, :, :, :, 2], i_23[:, :, :, :, 3]]

    p_33 = input_[i_33[:, :, :, :, 0], i_33[:, :, :, :, 1], i_33[:, :, :, :, 2], i_33[:, :, :, :, 3]]



    col0 = _hermite(p_00, p_10, p_20, p_30, torch.from_numpy(x_t))

    col1 = _hermite(p_01, p_11, p_21, p_31, torch.from_numpy(x_t))

    col2 = _hermite(p_02, p_12, p_22, p_32, torch.from_numpy(x_t))

    col3 = _hermite(p_03, p_13, p_23, p_33, torch.from_numpy(x_t))

    value = _hermite(col0, col1, col2, col3, torch.from_numpy(y_t))



    return value

def diff_clamp(tensor, min, max):

    tensor = tensor - tensor.detach() + torch.max(tensor, torch.full_like(tensor, min)).detach()

    tensor = tensor - tensor.detach() + torch.min(tensor, torch.full_like(tensor, max)).detach()

    return tensor



def WarpST(U, V, out_size, **kwargs):

    """Deformable Transformer Layer with bicubic interpolation

    U : tf.float, [num_batch, height, width, num_channels].

        Input tensor to warp

    V : tf.float, [num_batch, height, width, 2]

        Warp map. It is interpolated to out_size.

    out_size: a tuple of two ints

        The size of the output of the network (height, width)

    ----------

    References :

      https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py

    """



    def _repeat(x, n_repeats):

        rep = torch.ones((n_repeats, )).unsqueeze(1).transpose(1, 0)

        rep = rep.long()

        x = torch.matmul(x.reshape(-1, 1), rep.type_as(x))

        return x.reshape(-1)



    def _interpolate(im, x, y, out_size):

        # constants

        num_batch = im.size(0)

        height = im.size(1)

        width = im.size(2)

        channels = im.size(3)



        x = x.float()

        y = y.float()

        height_f = float(height)

        width_f = float(width)

        out_height = out_size[0]

        out_width = out_size[1]

        zero = torch.zeros([]).long()

        max_y = im.size(1) - 1

        max_x = im.size(2) - 1



        # scale indices from [-1, 1] to [0, width/height]

        x = (x + 1.0)*(width_f) / 2.0

        y = (y + 1.0)*(height_f) / 2.0



        # do sampling

        x0 = torch.floor(x).long()

        x1 = x0 + 1

        y0 = torch.floor(y).long()

        y1 = y0 + 1



        # x0 = torch.clamp(x0, zero.item(), max_x)

        # x1 = torch.clamp(x1, zero.item(), max_x)

        # y0 = torch.clamp(y0, zero.item(), max_y)

        # y1 = torch.clamp(y1, zero.item(), max_y)



        x0 = diff_clamp(x0, zero.item(), max_x)

        x1 = diff_clamp(x1, zero.item(), max_x)

        y0 = diff_clamp(y0, zero.item(), max_y)

        y1 = diff_clamp(y1, zero.item(), max_y)



        dim2 = width

        dim1 = width*height

        base = _repeat(torch.range(0, num_batch - 1, 1)*dim1, out_height*out_width)

        base_y0 = base.type_as(y0) + y0*dim2

        base_y1 = base.type_as(y1) + y1*dim2

        idx_a = base_y0 + x0

        idx_b = base_y1 + x0

        idx_c = base_y0 + x1

        idx_d = base_y1 + x1



        # use indices to lookup pixels in the flat image and restore

        # channels dim

        im_flat = im.reshape(-1, channels)

        im_flat = im_flat.float()

        Ia = im_flat[idx_a]

        Ib = im_flat[idx_b]

        Ic = im_flat[idx_c]

        Id = im_flat[idx_d]



        # and finally calculate interpolated values

        x0_f = x0.float()

        x1_f = x1.float()

        y0_f = y0.float()

        y1_f = y1.float()

        wa = ((x1_f-x) * (y1_f-y)).unsqueeze(1)

        wb = ((x1_f-x) * (y-y0_f)).unsqueeze(1)

        wc = ((x-x0_f) * (y1_f-y)).unsqueeze(1)

        wd = ((x-x0_f) * (y-y0_f)).unsqueeze(1)

        output = wa*Ia + wb*Ib + wc*Ic + wd*Id

        return output



    def _meshgrid(height, width):

        # This should be equivalent to:

        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),

        #                         np.linspace(-1, 1, height))

        #  ones = np.ones(np.prod(x_t.shape))

        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])

        x_t = torch.matmul(torch.ones((height, 1)),

                        torch.linspace(-1.0, 1.0, width).unsqueeze(1).transpose(1, 0))

        y_t = torch.matmul(torch.linspace(-1.0, 1.0, height).unsqueeze(1), torch.ones((1, width)))



        x_t_flat = x_t.reshape(1, -1)

        y_t_flat = y_t.reshape(1, -1)



        grid = torch.cat((x_t_flat, y_t_flat), dim = 0)

        return grid



    def _transform(V, U, out_size):

        num_batch = U.size(0)

        height = U.size(2)

        width = U.size(3)

        num_channels = U.size(1)



        # grid of (x_t, y_t, 1), eq (1) in ref [1]

        # height_f = tf.cast(height, 'float32')

        height_f = float(height)

        # width_f = tf.cast(width, 'float32')

        width_f = float(width)



        out_height = out_size[0]

        out_width = out_size[1]

        grid = _meshgrid(out_height, out_width)     # [2, h*w]



        grid = grid.reshape(-1)               # [2*h*w]

        grid = grid.unsqueeze(0).repeat(num_batch, 1)      # [n, 2*h*w]

        grid = grid.reshape(num_batch, 2, -1)       # [n, 2, h*w]





        # Set source position (x+vx, y+vy)^T

        V = bicubic_interp_2d(V.permute(0, 2, 3, 1), out_size)

        V = V.permute(0, 3, 1, 2)           # [n, 2, h, w]

        V = V.reshape(num_batch, 2, -1)       # [n, 2, h*w]

        T_g = V + grid                       # [n, 2, h*w]



        x_s = T_g[:, 0, :]

        y_s = T_g[:, 1, :]

        x_s_flat = x_s.reshape(-1)

        y_s_flat = y_s.reshape(-1)



        input_transformed = _interpolate(

            U.permute(0, 2, 3, 1), x_s_flat, y_s_flat, out_size)



        output = input_transformed.reshape(num_batch, out_height, out_width, num_channels)

        return output



    output = _transform(V, U, out_size)

    return output
def plot(im):

  im = np.array(im.tolist())

  plt.imshow(im, cmap='gray', vmin=0, vmax=1)

  plt.show()

  return None



class CNN(nn.Module):



  def __init__(self):

    super().__init__()



    self.enc_x = nn.Sequential(

      conv2d(2, 64, 3, 1, 1, bias=False), # 64 x 28 x 28

      nn.BatchNorm2d(64, momentum=0.9, eps=1e-5),

      nn.ELU(),

      nn.AvgPool2d(2, 2, 0), # 64 x 14 x 14



      conv2d(64, 128, 3, 1, 1, bias=False),

      nn.BatchNorm2d(128, momentum=0.9, eps=1e-5),

      nn.ELU(),

      conv2d(128, 128, 3, 1, 1, bias=False),

      nn.BatchNorm2d(128, momentum=0.9, eps=1e-5),

      nn.ELU(),

      nn.AvgPool2d(2, 2, 0),  # 64 x 7 x 7



      conv2d(128, 2, 3, 1, 1), # 2 x 7 x 7

      nn.Tanh()

    )



  def forward(self, x):

    x = self.enc_x(x)

    return x





class DIRNet(nn.Module):

  def __init__(self, config):

    super().__init__()

    self.vCNN = CNN()

    self.config = config



  def forward(self, x, y):

    xy = torch.cat((x, y), dim = 1)

    v = self.vCNN(xy)

    # print(str(v.max().item()) + ' '+ str(v.min().item()))

    print('cnn out ',v.size())

    z = WarpST(x, v, self.config.im_size)

    print('after warp ',z.size())

    # loss = ncc(y, z)

    z = z.permute(0, 3, 1, 2)

    loss = mse(y, z)

    return z, loss



  def deploy(self, dir_path, x, y):

        with torch.no_grad():

            z, _ = self.forward(x, y)

            for i in range(z.shape[0]):

                save_image_with_scale(dir_path+"/{:02d}_x.tif".format(i+1), x.permute(0, 2, 3, 1)[i,:,:,0].numpy())

                save_image_with_scale(dir_path+"/{:02d}_y.tif".format(i+1), y.permute(0, 2, 3, 1)[i,:,:,0].numpy())

                save_image_with_scale(dir_path+"/{:02d}_z.tif".format(i+1), z.permute(0, 2, 3, 1)[i,:,:,0].numpy())


torch.manual_seed(0)

train_batch = 60000

test_batch = 10000





def main():

    

    config = get_config(is_train=True)

    mkdir(config.tmp_dir)

    mkdir(config.ckpt_dir)



    model = DIRNet(config)



    transform = tf.Compose([tf.Resize([16, 16]), tf.ToTensor()])

    train_loader = DataLoader(MNIST('../mnist', train=True, download=True, transform=transform), batch_size=train_batch)

    test_loader = DataLoader(MNIST('../mnist', train=False, download=True, transform=transform), batch_size=test_batch)

    for batch, (data, label) in enumerate(train_loader):

        if batch == 0:

            num_images = 16000

            # num_images = 300

            digit_0 = data.index_select(0, label.eq(0).nonzero().squeeze())[:3000]

            digit_1 = data.index_select(0, label.eq(1).nonzero().squeeze())[:3000]

            digit_2 = data.index_select(0, label.eq(2).nonzero().squeeze())[:3000]

            digit_3 = data.index_select(0, label.eq(3).nonzero().squeeze())[:3000]

            digit_4 = data.index_select(0, label.eq(4).nonzero().squeeze())[:3000]

            digit_5 = data.index_select(0, label.eq(5).nonzero().squeeze())[:3000]

            digit_6 = data.index_select(0, label.eq(6).nonzero().squeeze())[:3000]

            digit_7 = data.index_select(0, label.eq(7).nonzero().squeeze())[:3000]

            digit_8 = data.index_select(0, label.eq(8).nonzero().squeeze())[:3000]

            digit_9 = data.index_select(0, label.eq(9).nonzero().squeeze())[:3000]



    digit = torch.stack([digit_0, digit_1, digit_2, digit_3, digit_4, digit_5, digit_6, digit_7,

                       digit_8, digit_9], dim=0)

    #3000张1*16*16的1，2，。。。9

    print('digit is ',digit_0.size(),digit.size())



    for batch, (data, label) in enumerate(test_loader):

        if batch == 0:

            num_images = 16000

            # num_images = 300

            digit_0_t = data.index_select(0, label.eq(0).nonzero().squeeze())[:500]

            digit_1_t = data.index_select(0, label.eq(1).nonzero().squeeze())[:500]

            digit_2_t = data.index_select(0, label.eq(2).nonzero().squeeze())[:500]

            digit_3_t = data.index_select(0, label.eq(3).nonzero().squeeze())[:500]

            digit_4_t = data.index_select(0, label.eq(4).nonzero().squeeze())[:500]

            digit_5_t = data.index_select(0, label.eq(5).nonzero().squeeze())[:500]

            digit_6_t = data.index_select(0, label.eq(6).nonzero().squeeze())[:500]

            digit_7_t = data.index_select(0, label.eq(7).nonzero().squeeze())[:500]

            digit_8_t = data.index_select(0, label.eq(8).nonzero().squeeze())[:500]

            digit_9_t = data.index_select(0, label.eq(9).nonzero().squeeze())[:500]



    digit_t = torch.stack([digit_0_t, digit_1_t, digit_2_t, digit_3_t, digit_4_t, digit_5_t, digit_6_t, digit_7_t,

                       digit_8_t, digit_9_t], dim=0)

    #500张1*16*16的1，2，。。。9

    print('digit_t is ',digit_0_t.size(),digit_t.size())





    optim = torch.optim.Adam(model.parameters(), lr = config.lr)

    scheduler = StepLR(optim, step_size=200, gamma=0.5)



    train_pr = MNISTDataHandler(digit)

    test_pr = MNISTDataHandler(digit_t)



    total_loss = 0

    for i in range(config.iteration):

        

        batch_x, batch_y = train_pr.sample_pair(config.batch_size)

        optim.zero_grad()

        print('input ',batch_x.size(),batch_y.size())

        _, loss = model(batch_x, batch_y)

        loss.backward()

        optim.step()

        scheduler.step()

        total_loss += loss



        if (i+1) % 100 == 0:

          print("iter {:>6d} : {}".format(i + 1, total_loss))

          total_loss = 0

          batch_x, batch_y = test_pr.sample_pair(config.batch_size)

          model.deploy(config.tmp_dir, batch_x, batch_y)

#       # reg.save(config.ckpt_dir)



if __name__ == "__main__":

  main()
a = torch.ones((64,1,16,16))

b = torch.ones((64,1,16,16))

c = torch.cat((a,b),dim=1)

print(c.size())
! ls ../tmp
import matplotlib.image as mpimg # mpimg 用于读取图片

import matplotlib.pyplot as plt

lena = mpimg.imread("../tmp/01_x.tif")

lena1 = mpimg.imread("../tmp/01_y.tif")

lena2 = mpimg.imread("../tmp/01_z.tif")

plt.subplot(1,3,1)

plt.imshow(lena) # 显示图片

plt.axis('off') # 不显示坐标轴

plt.subplot(1,3,2)

plt.imshow(lena1) # 显示图片

plt.axis('off') # 不显示坐标轴

plt.subplot(1,3,3)

plt.imshow(lena2) # 显示图片

plt.axis('off') # 不显示坐标轴

plt.show()