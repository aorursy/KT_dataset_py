import argparse

import torch

import torchvision.transforms as transforms

import os, sys

from PIL import Image

import glob

import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '..'))

cwd = os.getcwd()

print(cwd)

import numpy as np

#from Utils.utils import str2bool, AverageMeter, depth_read 

#import Models

#import Datasets

from PIL import ImageOps

import matplotlib.pyplot as plt

import time

import os

import sys

import re

import numpy as np

from PIL import Image



sys.path.insert(1, os.path.join(sys.path[0], '..'))
# Utils



import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (35, 30)

from PIL import Image

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import argparse

import os

import torch.optim

from torch.optim import lr_scheduler

import errno

import sys

from torchvision import transforms

import torch.nn.init as init

import torch.distributed as dist



def define_optim(optim, params, lr, weight_decay):

    if optim == 'adam':

        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    elif optim == 'sgd':

        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)

    elif optim == 'rmsprop':

        optimizer = torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)

    else:

        raise KeyError("The requested optimizer: {} is not implemented".format(optim))

    return optimizer





def define_scheduler(optimizer, args):

    if args.lr_policy == 'lambda':

        def lambda_rule(epoch):

            lr_l = 1.0 - max(0, epoch + 1 - args.niter) / float(args.niter_decay + 1)

            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif args.lr_policy == 'step':

        scheduler = lr_scheduler.StepLR(optimizer,

                                        step_size=args.lr_decay_iters, gamma=args.gamma)

    elif args.lr_policy == 'plateau':

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',

                                                   factor=args.gamma,

                                                   threshold=0.0001,

                                                   patience=args.lr_decay_iters)

    elif args.lr_policy == 'none':

        scheduler = None

    else:

        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)

    return scheduler





def define_init_weights(model, init_w='normal', activation='relu'):

    print('Init weights in network with [{}]'.format(init_w))

    if init_w == 'normal':

        model.apply(weights_init_normal)

    elif init_w == 'xavier':

        model.apply(weights_init_xavier)

    elif init_w == 'kaiming':

        model.apply(weights_init_kaiming)

    elif init_w == 'orthogonal':

        model.apply(weights_init_orthogonal)

    else:

        raise NotImplementedError('initialization method [{}] is not implemented'.format(init_w))





def first_run(save_path):

    txt_file = os.path.join(save_path, 'first_run.txt')

    if not os.path.exists(txt_file):

        open(txt_file, 'w').close()

    else:

        saved_epoch = open(txt_file).read()

        if saved_epoch is None:

            print('You forgot to delete [first run file]')

            return ''

        return saved_epoch

    return ''





def depth_read(img, sparse_val):

    # loads depth map D from png file

    # and returns it as a numpy array,

    # for details see readme.txt

    depth_png = np.array(img, dtype=int)

    depth_png = np.expand_dims(depth_png, axis=2)

    # make sure we have a proper 16bit depth map here.. not 8bit!

    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float) / 256.

    depth[depth_png == 0] = sparse_val

    return depth





class show_figs():

    def __init__(self, input_type, savefig=False):

        self.input_type = input_type

        self.savefig = savefig



    def save(self, img, name):

        img.save(name)



    def transform(self, input, name='test.png'):

        if isinstance(input, torch.tensor):

            input = torch.clamp(input, min=0, max=255).int().cpu().numpy()

            input = input * 256.

            img = Image.fromarray(input)



        elif isinstance(input, np.array):

            img = Image.fromarray(input)



        else:

            raise NotImplementedError('Input type not recognized type')



        if self.savefig:

            self.save(img, name)

        else:

            return img



# trick from stackoverflow

def str2bool(argument):

    if argument.lower() in ('yes', 'true', 't', 'y', '1'):

        return True

    elif argument.lower() in ('no', 'false', 'f', 'n', '0'):

        return False

    else:

        raise argparse.ArgumentTypeError('Wrong argument in argparse, should be a boolean')





def mkdir_if_missing(directory):

    if not os.path.exists(directory):

        try:

            os.makedirs(directory)

        except OSError as e:

            if e.errno != errno.EEXIST:

                raise





class AverageMeter(object):

    """Computes and stores the average and current value"""

    def __init__(self):

        self.reset()



    def reset(self):

        self.val = 0

        self.avg = 0

        self.sum = 0

        self.count = 0



    def update(self, val, n=1):

        self.val = val

        self.sum += val * n

        self.count += n

        self.avg = self.sum / self.count





def write_file(content, location):

    file = open(location, 'w')

    file.write(str(content))

    file.close()





class Logger(object):

    """

    Source https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.

    """

    def __init__(self, fpath=None):

        self.console = sys.stdout

        self.file = None

        self.fpath = fpath

        if fpath is not None:

            mkdir_if_missing(os.path.dirname(fpath))

            self.file = open(fpath, 'w')



    def __del__(self):

        self.close()



    def __enter__(self):

        pass



    def __exit__(self, *args):

        self.close()



    def write(self, msg):

        self.console.write(msg)

        if self.file is not None:

            self.file.write(msg)



    def flush(self):

        self.console.flush()

        if self.file is not None:

            self.file.flush()

            os.fsync(self.file.fileno())



    def close(self):

        self.console.close()

        if self.file is not None:

            self.file.close()



def save_image(img_merge, filename):

    img_merge = Image.fromarray(img_merge.astype('uint8'))

    img_merge.save(filename)





def weights_init_normal(m):

    classname = m.__class__.__name__

#    print(classname)

    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:

        init.normal_(m.weight.data, 0.0, 0.02)

        if m.bias is not None:

            m.bias.data.zero_()

    elif classname.find('Linear') != -1:

        init.normal_(m.weight.data, 0.0, 0.02)

        if m.bias is not None:

            m.bias.data.zero_()

    elif classname.find('BatchNorm2d') != -1:

        init.normal_(m.weight.data, 1.0, 0.02)

        init.constant_(m.bias.data, 0.0)





def weights_init_xavier(m):

    classname = m.__class__.__name__

    # print(classname)

    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:

        init.xavier_normal_(m.weight.data, gain=0.02)

        if m.bias is not None:

            m.bias.data.zero_()

    elif classname.find('Linear') != -1:

        init.xavier_normal_(m.weight.data, gain=0.02)

        if m.bias is not None:

            m.bias.data.zero_()

    elif classname.find('BatchNorm2d') != -1:

        init.normal_(m.weight.data, 1.0, 0.02)

        init.constant_(m.bias.data, 0.0)





def weights_init_kaiming(m):

    classname = m.__class__.__name__

    # print(classname)

    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:

        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')

        if m.bias is not None:

            m.bias.data.zero_()

    elif classname.find('Linear') != -1:

        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')

        if m.bias is not None:

            m.bias.data.zero_()

    elif classname.find('BatchNorm2d') != -1:

        init.normal_(m.weight.data, 1.0, 0.02)

        init.constant_(m.bias.data, 0.0)





def weights_init_orthogonal(m):

    classname = m.__class__.__name__

#    print(classname)

    if classname.find('Conv') != -1 or classname.find('ConvTranspose') != -1:

        init.orthogonal(m.weight.data, gain=1)

        if m.bias is not None:

            m.bias.data.zero_()

    elif classname.find('Linear') != -1:

        init.orthogonal(m.weight.data, gain=1)

        if m.bias is not None:

            m.bias.data.zero_()

    elif classname.find('BatchNorm2d') != -1:

        init.normal_(m.weight.data, 1.0, 0.02)

        init.constant_(m.bias.data, 0.0)





def save_fig(inp, name='saved.png'):

    if isinstance(inp, torch.Tensor):

        # inp = inp.permute([2, 0, 1])

        inp = transforms.ToPILImage()(inp.int())

        inp.save(name)

        return

    pil = Image.fromarray(inp)

    pil.save(name)



def setup_for_distributed(is_master):

    """

    This function disables printing when not in master process

    """

    import builtins as __builtin__

    builtin_print = __builtin__.print



    def print(*args, **kwargs):

        force = kwargs.pop('force', False)

        if is_master or force:

            builtin_print(*args, **kwargs)



    __builtin__.print = print



def init_distributed_mode(args):

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:

        args.rank = int(os.environ["RANK"])

        args.world_size = int(os.environ['WORLD_SIZE'])

        args.gpu = int(os.environ['LOCAL_RANK'])

    elif 'SLURM_PROCID' in os.environ:

        args.rank = int(os.environ['SLURM_PROCID'])

        args.gpu = args.rank % torch.cuda.device_count()

    else:

        print('Not using distributed mode')

        args.distributed = False

        return



    args.distributed = True



    torch.cuda.set_device(args.gpu)

    args.dist_backend = 'nccl'

    print('| distributed init (rank {}): {}'.format(

        args.rank, args.dist_url), flush=True)

    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,

                                         world_size=args.world_size, rank=args.rank)

    # Does not seem to work?

    torch.distributed.barrier()

    setup_for_distributed(args.rank == 0)
# Kitti_loader



class Random_Sampler():

    "Class to downsample input lidar points"



    def __init__(self, num_samples):

        self.num_samples = num_samples



    def sample(self, depth):

        mask_keep = depth > 0

        n_keep = np.count_nonzero(mask_keep)



        if n_keep == 0:

            return mask_keep

        else:

            depth_sampled = np.zeros(depth.shape)

            prob = float(self.num_samples) / n_keep

            mask_keep =  np.bitwise_and(mask_keep, np.random.uniform(0, 1, depth.shape) < prob)

            depth_sampled[mask_keep] = depth[mask_keep]

            return depth_sampled





class Kitti_preprocessing(object):

    def __init__(self, dataset_path, input_type='depth', side_selection=''):

        self.train_paths = {'img': [], 'lidar_in': [], 'gt': []}

        self.val_paths = {'img': [], 'lidar_in': [], 'gt': []}

        self.selected_paths = {'img': [], 'lidar_in': [], 'gt': []}

        self.test_files = {'img': [], 'lidar_in': []}

        self.dataset_path = dataset_path

        self.side_selection = side_selection

        self.left_side_selection = 'image_02'

        self.right_side_selection = 'image_03'

        self.depth_keyword = 'proj_depth'

        self.rgb_keyword = 'Rgb'

        # self.use_rgb = input_type == 'rgb'

        self.use_rgb = True

        self.date_selection = '2011_09_26'



    def get_paths(self):

        # train and validation dirs

        for type_set in os.listdir(self.dataset_path):

            for root, dirs, files in os.walk(os.path.join(self.dataset_path, type_set)):

                if re.search(self.depth_keyword, root):

                    self.train_paths['lidar_in'].extend(sorted([os.path.join(root, file) for file in files

                                                        if re.search('velodyne_raw', root)

                                                        and re.search('train', root)

                                                        and re.search(self.side_selection, root)]))

                    self.val_paths['lidar_in'].extend(sorted([os.path.join(root, file) for file in files

                                                              if re.search('velodyne_raw', root)

                                                              and re.search('val', root)

                                                              and re.search(self.side_selection, root)]))

                    self.train_paths['gt'].extend(sorted([os.path.join(root, file) for file in files

                                                          if re.search('groundtruth', root)

                                                          and re.search('train', root)

                                                          and re.search(self.side_selection, root)]))

                    self.val_paths['gt'].extend(sorted([os.path.join(root, file) for file in files

                                                        if re.search('groundtruth', root)

                                                        and re.search('val', root)

                                                        and re.search(self.side_selection, root)]))

                if self.use_rgb:

                    if re.search(self.rgb_keyword, root) and re.search(self.side_selection, root):

                        self.train_paths['img'].extend(sorted([os.path.join(root, file) for file in files

                                                               if re.search('train', root)]))

                                                               # and (re.search('image_02', root) or re.search('image_03', root))

                                                               # and re.search('data', root)]))

                       # if len(self.train_paths['img']) != 0:

                           # test = [os.path.join(root, file) for file in files if re.search('train', root)]

                        self.val_paths['img'].extend(sorted([os.path.join(root, file) for file in files

                                                            if re.search('val', root)]))

                                                            # and (re.search('image_02', root) or re.search('image_03', root))

                                                            # and re.search('data', root)]))

               # if len(self.train_paths['lidar_in']) != len(self.train_paths['img']):

                   # print(root)





    def downsample(self, lidar_data, destination, num_samples=500):

        # Define sampler

        sampler = Random_Sampler(num_samples)



        for i, lidar_set_path in tqdm.tqdm(enumerate(lidar_data)):

            # Read in lidar data

            name = os.path.splitext(os.path.basename(lidar_set_path))[0]

            sparse_depth = Image.open(lidar_set_path)





            # Convert to numpy array

            sparse_depth = np.array(sparse_depth, dtype=int)

            assert(np.max(sparse_depth) > 255)



            # Downsample per collumn

            sparse_depth = sampler.sample(sparse_depth)



            # Convert to img

            sparse_depth_img = Image.fromarray(sparse_depth.astype(np.uint32))



            # Save

            folder = os.path.join(*str.split(lidar_set_path, os.path.sep)[7:12])

            os.makedirs(os.path.join(destination, os.path.join(folder)), exist_ok=True)

            sparse_depth_img.save(os.path.join(destination, os.path.join(folder, name)) + '.png')



    def convert_png_to_rgb(self, rgb_images, destination):

        for i, img_set_path in tqdm.tqdm(enumerate(rgb_images)):

            name = os.path.splitext(os.path.basename(img_set_path))[0]

            im = Image.open(img_set_path)

            rgb_im = im.convert('RGB')

            folder = os.path.join(*str.split(img_set_path, os.path.sep)[8:12])

            os.makedirs(os.path.join(destination, os.path.join(folder)), exist_ok=True)

            rgb_im.save(os.path.join(destination, os.path.join(folder, name)) + '.jpg')

            # rgb_im.save(os.path.join(destination, name) + '.jpg')



    def get_selected_paths(self, selection):

        files = []

        for file in sorted(os.listdir(os.path.join(self.dataset_path, selection))):

            files.append(os.path.join(self.dataset_path, os.path.join(selection, file)))

        return files



    def prepare_dataset(self):

        path_to_val_sel = 'reflection_completion_test'

        path_to_test = 'depth_selection/test_depth_completion_anonymous'

        #self.get_paths()

        self.selected_paths['lidar_in'] = self.get_selected_paths(os.path.join(path_to_val_sel, 'velodyne_raw'))

        self.selected_paths['gt'] = self.get_selected_paths(os.path.join(path_to_val_sel, 'groundtruth_depth'))

        self.selected_paths['img'] = self.get_selected_paths(os.path.join(path_to_val_sel, 'image'))

        #self.test_files['lidar_in'] = self.get_selected_paths(os.path.join(path_to_test, 'velodyne_raw'))

        if self.use_rgb:

            self.selected_paths['img'] = self.get_selected_paths(os.path.join(path_to_val_sel, 'image'))

            #self.test_files['img'] = self.get_selected_paths(os.path.join(path_to_test, 'image'))

            #print(len(self.train_paths['lidar_in']))

            #print(len(self.train_paths['img']))

            #print(len(self.train_paths['gt']))

            #print(len(self.val_paths['lidar_in']))

            #print(len(self.val_paths['img']))

            #print(len(self.val_paths['gt']))

            #print(len(self.test_files['lidar_in']))

            #print(len(self.test_files['img']))



    def compute_mean_std(self):

        nums = np.array([])

        means = np.array([])

        stds = np.array([])

        max_lst = np.array([])

        for i, raw_img_path in tqdm.tqdm(enumerate(self.train_paths['lidar_in'])):

            raw_img = Image.open(raw_img_path)

            raw_np = depth_read(raw_img)

            vec = raw_np[raw_np >= 0]

            # vec = vec/84.0

            means = np.append(means, np.mean(vec))

            stds = np.append(stds, np.std(vec))

            nums = np.append(nums, len(vec))

            max_lst = np.append(max_lst, np.max(vec))

        mean = np.dot(nums, means)/np.sum(nums)

        std = np.sqrt((np.dot(nums, stds**2) + np.dot(nums, (means-mean)**2))/np.sum(nums))

        return mean, std, max_lst
# ERFNet



import torch

import torch.nn as nn

import torch.nn.functional as F





class DownsamplerBlock (nn.Module):

    def __init__(self, ninput, noutput):

        super().__init__()



        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)

        self.pool = nn.MaxPool2d(2, stride=2)

        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)



    def forward(self, input):

        output = torch.cat([self.conv(input), self.pool(input)], 1)

        output = self.bn(output)

        return F.relu(output)





class non_bottleneck_1d (nn.Module):

    def __init__(self, chann, dropprob, dilated):

        super().__init__()



        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)



        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)



        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)



        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation=(dilated, 1))



        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1*dilated), bias=True, dilation=(1, dilated))



        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)



        self.dropout = nn.Dropout2d(dropprob)



    def forward(self, input):



        output = self.conv3x1_1(input)

        output = F.relu(output)

        output = self.conv1x3_1(output)

        output = self.bn1(output)

        output = F.relu(output)



        output = self.conv3x1_2(output)

        output = F.relu(output)

        output = self.conv1x3_2(output)

        output = self.bn2(output)



        if (self.dropout.p != 0):

            output = self.dropout(output)



        return F.relu(output+input)





class Encoder(nn.Module):

    def __init__(self, in_channels, num_classes):

        super().__init__()

        chans = 32 if in_channels > 16 else 16

        self.initial_block = DownsamplerBlock(in_channels, chans)



        self.layers = nn.ModuleList()



        self.layers.append(DownsamplerBlock(chans, 64))



        for x in range(0, 5):

            self.layers.append(non_bottleneck_1d(64, 0.03, 1)) 



        self.layers.append(DownsamplerBlock(64, 128))



        for x in range(0, 2):

            self.layers.append(non_bottleneck_1d(128, 0.3, 2))

            self.layers.append(non_bottleneck_1d(128, 0.3, 4))

            self.layers.append(non_bottleneck_1d(128, 0.3, 8))

            self.layers.append(non_bottleneck_1d(128, 0.3, 16))



        #Only in encoder mode:

        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)



    def forward(self, input, predict=False):

        output = self.initial_block(input)



        for layer in self.layers:

            output = layer(output)



        if predict:

            output = self.output_conv(output)



        return output





class UpsamplerBlock (nn.Module):

    def __init__(self, ninput, noutput):

        super().__init__()

        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)

        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)



    def forward(self, input):

        output = self.conv(input)

        output = self.bn(output)

        return F.relu(output)





class Decoder (nn.Module):

    def __init__(self, num_classes):

        super().__init__()



        self.layer1 = UpsamplerBlock(128, 64)

        self.layer2 = non_bottleneck_1d(64, 0, 1)

        self.layer3 = non_bottleneck_1d(64, 0, 1) # 64x64x304



        self.layer4 = UpsamplerBlock(64, 32)

        self.layer5 = non_bottleneck_1d(32, 0, 1)

        self.layer6 = non_bottleneck_1d(32, 0, 1) # 32x128x608



        self.output_conv = nn.ConvTranspose2d(32, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)



    def forward(self, input):

        output = input

        output = self.layer1(output)

        output = self.layer2(output)

        output = self.layer3(output)

        em2 = output

        output = self.layer4(output)

        output = self.layer5(output)

        output = self.layer6(output)

        em1 = output



        output = self.output_conv(output)



        return output, em1, em2





class Net(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):  #use encoder to pass pretrained encoder

        super().__init__()

        self.encoder = Encoder(in_channels, out_channels)

        self.decoder = Decoder(out_channels)



    def forward(self, input, only_encode=False):

        if only_encode:

            return self.encoder.forward(input, predict=True)

        else:

            output = self.encoder(input)

            return self.decoder.forward(output)
# model



import torch

import torch.nn as nn

import torch.utils.data

import torch.nn.functional as F

import numpy as np

#from .ERFNet import Net



class uncertainty_net(nn.Module):

    def __init__(self, in_channels, out_channels=1, thres=15):

        super(uncertainty_net, self).__init__()

        out_chan = 2



        combine = 'concat'

        self.combine = combine

        self.in_channels = in_channels



        out_channels = 3

        self.depthnet = Net(in_channels=in_channels, out_channels=out_channels)



        local_channels_in = 2 if self.combine == 'concat' else 1

        self.convbnrelu = nn.Sequential(convbn(local_channels_in, 32, 3, 1, 1, 1),

                                        nn.ReLU(inplace=True))

        self.hourglass1 = hourglass_1(32)

        self.hourglass2 = hourglass_2(32)

        self.fuse = nn.Sequential(convbn(32, 32, 3, 1, 1, 1),

                                   nn.ReLU(inplace=True),

                                   nn.Conv2d(32, out_chan, kernel_size=3, padding=1, stride=1, bias=True))

        self.activation = nn.ReLU(inplace=True)

        self.thres = thres

        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, input, epoch=50):

        if self.in_channels > 1:

            rgb_in = input[:, 1:, :, :]

            lidar_in = input[:, 0:1, :, :]

        else:

            lidar_in = input



        # 1. GLOBAL NET

        embedding0, embedding1, embedding2 = self.depthnet(input)



        global_features = embedding0[:, 0:1, :, :]

        precise_depth = embedding0[:, 1:2, :, :]

        conf = embedding0[:, 2:, :, :]



        # 2. Fuse 

        if self.combine == 'concat':

            input = torch.cat((lidar_in, global_features), 1)

        elif self.combine == 'add':

            input = lidar_in + global_features

        elif self.combine == 'mul':

            input = lidar_in * global_features

        elif self.combine == 'sigmoid':

            input = lidar_in * nn.Sigmoid()(global_features)

        else:

            input = lidar_in



        # 3. LOCAL NET

        out = self.convbnrelu(input)

        out1, embedding3, embedding4 = self.hourglass1(out, embedding1, embedding2)

        out1 = out1 + out

        out2 = self.hourglass2(out1, embedding3, embedding4)

        out2 = out2 + out

        out = self.fuse(out2)

        lidar_out = out



        # 4. Late Fusion

        lidar_to_depth, lidar_to_conf = torch.chunk(out, 2, dim=1)

        lidar_to_conf, conf = torch.chunk(self.softmax(torch.cat((lidar_to_conf, conf), 1)), 2, dim=1)

        out = conf * precise_depth + lidar_to_conf * lidar_to_depth



        return out, lidar_out, precise_depth, global_features





def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):



    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False))

                         # nn.BatchNorm2d(out_planes))





class hourglass_1(nn.Module):

    def __init__(self, channels_in):

        super(hourglass_1, self).__init__()



        self.conv1 = nn.Sequential(convbn(channels_in, channels_in, kernel_size=3, stride=2, pad=1, dilation=1),

                                   nn.ReLU(inplace=True))



        self.conv2 = convbn(channels_in, channels_in, kernel_size=3, stride=1, pad=1, dilation=1)



        self.conv3 = nn.Sequential(convbn(channels_in*2, channels_in*2, kernel_size=3, stride=2, pad=1, dilation=1),

                                   nn.ReLU(inplace=True))



        self.conv4 = nn.Sequential(convbn(channels_in*2, channels_in*2, kernel_size=3, stride=1, pad=1, dilation=1))



        self.conv5 = nn.Sequential(nn.ConvTranspose2d(channels_in*4, channels_in*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),

                                   nn.BatchNorm2d(channels_in*2),

                                   nn.ReLU(inplace=True))



        self.conv6 = nn.Sequential(nn.ConvTranspose2d(channels_in*2, channels_in, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),

                                   nn.BatchNorm2d(channels_in))



    def forward(self, x, em1, em2):

        x = self.conv1(x)

        x = self.conv2(x)

        x = F.relu(x, inplace=True)

        x = torch.cat((x, em1), 1)



        x_prime = self.conv3(x)

        x_prime = self.conv4(x_prime)

        x_prime = F.relu(x_prime, inplace=True)

        x_prime = torch.cat((x_prime, em2), 1)



        out = self.conv5(x_prime)

        out = self.conv6(out)



        return out, x, x_prime





class hourglass_2(nn.Module):

    def __init__(self, channels_in):

        super(hourglass_2, self).__init__()



        self.conv1 = nn.Sequential(convbn(channels_in, channels_in*2, kernel_size=3, stride=2, pad=1, dilation=1),

                                   nn.BatchNorm2d(channels_in*2),

                                   nn.ReLU(inplace=True))



        self.conv2 = convbn(channels_in*2, channels_in*2, kernel_size=3, stride=1, pad=1, dilation=1)



        self.conv3 = nn.Sequential(convbn(channels_in*2, channels_in*2, kernel_size=3, stride=2, pad=1, dilation=1),

                                   nn.BatchNorm2d(channels_in*2),

                                   nn.ReLU(inplace=True))



        self.conv4 = nn.Sequential(convbn(channels_in*2, channels_in*4, kernel_size=3, stride=1, pad=1, dilation=1))



        self.conv5 = nn.Sequential(nn.ConvTranspose2d(channels_in*4, channels_in*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),

                                   nn.BatchNorm2d(channels_in*2),

                                   nn.ReLU(inplace=True))



        self.conv6 = nn.Sequential(nn.ConvTranspose2d(channels_in*2, channels_in, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),

                                   nn.BatchNorm2d(channels_in))



    def forward(self, x, em1, em2):

        x = self.conv1(x)

        x = self.conv2(x)

        x = x + em1

        x = F.relu(x, inplace=True)



        x_prime = self.conv3(x)

        x_prime = self.conv4(x_prime)

        x_prime = x_prime + em2

        x_prime = F.relu(x_prime, inplace=True)



        out = self.conv5(x_prime)

        out = self.conv6(out)



        return out 
#Training setttings



parser = argparse.ArgumentParser(description='KITTI Depth Completion Task TEST')

parser.add_argument('--dataset', type=str, default='kitti', 

                    choices = {'kitti'}, help='dataset to work with')

parser.add_argument('--mod', type=str, default='mod', choices = {'mod'}, help='Model for use')

parser.add_argument('--no_cuda', action='store_true', help='no gpu usage')

parser.add_argument('--input_type', type=str, default='rgb', help='use rgb for rgbdepth')

# Data augmentation settings

parser.add_argument('--crop_w', type=int, default=1216, help='width of image after cropping')

parser.add_argument('--crop_h', type=int, default=256, help='height of image after cropping')



# Paths settings

parser.add_argument('--save_path', type= str, 

                    default='/kaggle/input/pretainedmodel', 

                    help='save path')

parser.add_argument('--data_path', type=str, 

                    default='/kaggle/input/pretainedmodel/reflection_completion_test', 

                    help='path to desired datasets')



# Cudnn

parser.add_argument("--cudnn", type=str2bool, nargs='?', const=True, default=True, help="cudnn optimization active")

parser.add_argument('--multi', type=str2bool, nargs='?', const=True, default=False, help="use multiple gpus")

parser.add_argument('--normal', type=str2bool, nargs='?', const=True, default=False, help="Normalize input")

parser.add_argument('--max_depth', type=float, default=85.0, help="maximum depth of input")

parser.add_argument('--sparse_val', type=float, default=0.0, help="encode sparse values with 0")

parser.add_argument('--num_samples', default=0, type=int, help='number of samples')
def calculateRSME(prediction, gt):

    valid_mask = (gt > 0).detach()

    prediction = prediction[valid_mask]

    gt = gt[valid_mask]

    abs_diff = (prediction - gt).abs()

    return torch.sqrt(torch.mean(torch.pow(abs_diff, 2))).item()
def main():

    global args

    global dataset

    #args = parser.parse_args()

    args = parser.parse_known_args()[0]



    torch.backends.cudnn.benchmark = args.cudnn

    



    best_file_name = glob.glob(os.path.join(args.save_path, 'model_best*'))[0]



    save_root = '/kaggle/working/results'

    if not os.path.isdir(save_root):

        os.makedirs(save_root)



    print("==========\nArgs:{}\n==========".format(args))

    # INIT

    print("Init model: '{}'".format(args.mod))

    channels_in = 1 if args.input_type == 'depth' else 4

    model = uncertainty_net(in_channels=channels_in)

    print("Number of parameters in model {} is {:.3f}M".format(args.mod.upper(), sum(tensor.numel() for tensor in model.parameters())/1e6))

    if not args.no_cuda:

        # Load on gpu before passing params to optimizer

        if not args.multi:

            model = model.cuda()

        else:

            model = torch.nn.DataParallel(model).cuda()

    if os.path.isfile(best_file_name):

        print("=> loading checkpoint '{}'".format(best_file_name))

        checkpoint = torch.load(best_file_name)

        model.load_state_dict(checkpoint['state_dict'])

        lowest_loss = checkpoint['loss']

        best_epoch = checkpoint['best epoch']

        print('Lowest RMSE for selection validation set was {:.4f} in epoch {}'.format(lowest_loss, best_epoch))

    else:

        print("=> no checkpoint found at '{}'".format(best_file_name))

        return



    if not args.no_cuda:

        model = model.cuda()

    print("Initializing dataset {}".format(args.dataset))

    #dataset = Datasets.define_dataset(args.dataset, args.data_path, args.input_type)

    dataset = Kitti_preprocessing(args.data_path, args.input_type)

    dataset.prepare_dataset()

    to_pil = transforms.ToPILImage()

    to_tensor = transforms.ToTensor()

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    depth_norm = transforms.Normalize(mean=[14.97/args.max_depth], std=[11.15/args.max_depth])

    model.eval()

    print("===> Start testing")

    total_time = []

    

    with torch.no_grad():

        RSME_total = 0

        for i, (img, rgb, gt) in tqdm.tqdm(enumerate(zip(dataset.selected_paths['lidar_in'],

                                           dataset.selected_paths['img'], dataset.selected_paths['gt']))):



            raw_path = os.path.join(img)

            raw_pil = Image.open(raw_path)

            gt_path = os.path.join(gt)

            gt_pil = Image.open(gt)

            assert raw_pil.size == (1216, 352)



            crop = 352-args.crop_h

            raw_pil_crop = raw_pil.crop((0, crop, 1216, 352))

            gt_pil_crop = gt_pil.crop((0, crop, 1216, 352))



            raw = depth_read(raw_pil_crop, args.sparse_val)

            raw = to_tensor(raw).float()

            gt = depth_read(gt_pil_crop, args.sparse_val)

            gt = to_tensor(gt).float()

            valid_mask = (raw > 0).detach().float()



            #input = torch.unsqueeze(raw, 0)

            #gt = torch.unsqueeze(gt, 0)

            input = torch.unsqueeze(raw, 0).cuda()

            gt = torch.unsqueeze(gt, 0).cuda()





            if args.normal:

                # Put in {0-1} range and then normalize

                input = input/args.max_depth

                # input = depth_norm(input)



            if args.input_type == 'rgb':

                rgb_path = os.path.join(rgb)

                rgb_pil = Image.open(rgb_path)

                assert rgb_pil.size == (1216, 352)

                rgb_pil_crop = rgb_pil.crop((0, crop, 1216, 352))

                rgb = to_tensor(rgb_pil_crop).float()

                #rgb = torch.unsqueeze(rgb, 0)

                rgb = torch.unsqueeze(rgb, 0).cuda()

                if not args.normal:

                    rgb = rgb*255.0



                input = torch.cat((input, rgb), 1)



            torch.cuda.synchronize()

            a = time.perf_counter()

            output, _, _, _ = model(input)

            torch.cuda.synchronize()

            b = time.perf_counter()

            total_time.append(b-a)

            if args.normal:

                output = output*args.max_depth

            output = torch.clamp(output, min=0, max=85)

            RSME_total += calculateRSME(output[:, 0:1], gt)



            print(type(output))

            print(output.shape)

            output = output * 256.

            raw = raw * 256.

            output = output[0][0:1].cpu()

            data = output[0].numpy()

    

            if crop != 0:

                padding = (0, 0, crop, 0)

                output = torch.nn.functional.pad(output, padding, "constant", 0)

                output[:, 0:crop] = output[:, crop].repeat(crop, 1)



            pil_img = to_pil(output.int())

            assert pil_img.size == (1216, 352)

            pil_img.save(os.path.join(save_root, os.path.basename(img)))

            #print(os.path.join(save_root, os.path.basename(img)))

    print('average RSME: ', RSME_total / (i + 1))

    print('average_time: ', sum(total_time[100:])/(len(total_time[100:])))

    print('num imgs: ', i + 1)





if __name__ == '__main__':

    main()
#from pathlib import Path

#import zipfile

#img_root = Path('/kaggle/working/results')

#with zipfile.ZipFile('imgs.zip', 'w') as z:

#    for img_name in img_root.iterdir():

#        z.write(img_name)