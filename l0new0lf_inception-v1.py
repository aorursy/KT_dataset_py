import torch

import torch.nn as nn

import matplotlib.pyplot as plt 

import numpy as np
class ConvReLU(nn.Module):

    def __init__(self, config):

        super(ConvReLU, self).__init__()

        self.relu = nn.ReLU()

        self.conv = self._get_conv_plus_relu(config) 

        

    def forward(self, x):

        x = self.conv(x)

        x = self.relu(x)

        return x

        

    def _get_conv_plus_relu(self, config):

        in_channels   = config['in_ch'] 

        out_channels  = config['out_ch'] 

        kernel_size   = config['z'] 

        stride        = config['s'] if ('s' in config.keys()) else (1,1) 

        padding       = config['p'] if ('p' in config.keys()) else (0,0) 

        

        return nn.Conv2d(

            in_channels, out_channels, 

            kernel_size, stride, padding

        )
config = {'in_ch': 3, 'out_ch': 100, 'z': (1,1)}

cr = ConvReLU(config)



# for z=1: s=1,p=0 retain w,h

cr(torch.rand(1,3,224,224)).shape
class InceptionV1(nn.Module):

    def __init__(self, config):

        super(InceptionV1, self).__init__()

        

        self.conv_block = ConvReLU

        self.maxpool = nn.MaxPool2d(kernel_size = (3,3), stride = (2,2), padding = (1,1), ceil_mode=False) # halve w,h (floored approximation)

        self.avgpool = nn.AvgPool2d(kernel_size = (7,7), stride = (1,1), padding = (0,0))

        self.config = config

        

        # 0  : sample

        # 1  : filter

        # 2  : w 

        # 3  : h

        self.concat_dim = 1 # filter-wise / depth-wise

        

    def forward(self, x):

        # inception v1

        x_b1, x_b2, x_b3, x_b4 = self._get_inception_v1(x)

        x = torch.cat((x_b1, x_b2, x_b3, x_b4), self.concat_dim)

        

        # pooling (halves w,h) if mentioned

        if 'pool' in self.config:

            if self.config['pool'] == 'max':

                x = self.maxpool(x)

            elif self.config['pool'] == 'avg':

                x = self.avgpool(x)

        

        return x

    

    def _get_inception_v1(self, x):

        """ `config` corresponds to row in arch in paper 

        

        Throughout the module, height and width is CONSTANT

        Only depth changes! (which is concatenated)

        """

        in_ch    = self.config['in_ch'] # or use 1-th idx

        out_b1   = self.config['out_1x1']

        red_b2   = self.config['red_3x3']

        out_b2   = self.config['out_3x3']

        red_b3   = self.config['red_5x5']

        out_b3   = self.config['out_5x5']

        out_b4   = self.config['out_pool']

        

        # s, p such that they

        # retain spatial structure (w,h)

        s_1x1, p_1x1 = (1,1), (0,0)

        s_3x3, p_3x3 = (1,1), (1,1)

        s_5x5, p_5x5 = (1,1), (2,2)

        

        b1 = self.conv_block({'in_ch': in_ch, 'out_ch': out_b1, 'z': (1,1), 's': s_1x1, 'p': p_1x1})

        b2 = nn.Sequential(

            self.conv_block({'in_ch':  in_ch, 'out_ch': red_b2, 'z': (1,1), 's': s_1x1, 'p': p_1x1}),

            self.conv_block({'in_ch': red_b2, 'out_ch': out_b2, 'z': (3,3), 's': s_3x3, 'p': p_3x3})

        )

        b3 = nn.Sequential(

            self.conv_block({'in_ch':  in_ch, 'out_ch': red_b3, 'z': (1,1), 's': s_1x1, 'p': p_1x1}),

            self.conv_block({'in_ch': red_b3, 'out_ch': out_b3, 'z': (5,5), 's': s_5x5, 'p': p_5x5})

        )

        b4 = nn.Sequential(

            nn.MaxPool2d(kernel_size=(3,3), stride=s_3x3, padding=p_3x3), # does not affect depth

            self.conv_block({'in_ch':  in_ch, 'out_ch': out_b4, 'z': (1,1), 's': s_1x1, 'p': p_1x1})

        )

        

        return b1(x), b2(x), b3(x), b4(x)
# 28×28×192 >>> 28×28×256

config = {

    'in_ch'   : 192, 

    'out_1x1' :  64,

    'red_3x3' :  96, 'out_3x3': 128,

    'red_5x5' :  16, 'out_5x5':  32, # expensive

    'out_pool':  32

}



iv1 = InceptionV1(config)

iv1(torch.rand(1,192,28,28)).shape
# 56×56×192 >>> 28×28×256

config = {

    'in_ch'   : 192, 

    'out_1x1' :  64,

    'red_3x3' :  96, 'out_3x3': 128,

    'red_5x5' :  16, 'out_5x5':  32, # expensive

    'out_pool':  32,

    'pool'    : 'max'

}



iv1 = InceptionV1(config)

iv1(torch.rand(1,192,56,56)).shape

# Note: approximatio w/ interger flooring
# 7×7×832 >>> 1×1×1024

config = {

    'in_ch'   : 832, 

    'out_1x1' : 384,

    'red_3x3' : 192, 'out_3x3': 384,

    'red_5x5' :  48, 'out_5x5': 128, # expensive

    'out_pool': 128,

    'pool'    : 'avg'

}



iv1 = InceptionV1(config)

iv1(torch.rand(1,832,7,7)).shape
class GoogLeNet(nn.Module):

    def __init__(self, config, in_ch=3, out_nodes=1000):

        super(GoogLeNet, self).__init__()

        self.config = config

        self.in_ch = in_ch 

        self.out_nodes = out_nodes

        

        # helpers

        self.conv_block = ConvReLU

        self.inception_block = InceptionV1

        self.pool = nn.MaxPool2d(kernel_size = (3,3), stride = (2,2), padding = (1,1), ceil_mode=False) # halve w,h (floored approximation)

        self.drop = nn.Dropout(0.4)

        self.softmax = nn.Softmax(dim=1) 

        

        # before inception layers

        self.conv1 = self.conv_block({'in_ch': self.in_ch, 'out_ch':  64, 'z': (7,7), 's': (2,2), 'p': (3,3)}) # (floored approximation)

        self.conv2 = self.conv_block({'in_ch':         64, 'out_ch': 192, 'z': (3,3), 's': (1,1), 'p': (1,1)})

        

        # after inception layers

        self.fc = nn.Linear(1024, self.out_nodes)

        

    def forward(self, x):

        """ w, h of x is 224"""

        

        # before inception

        #x = self.lrn(self.pool(self.conv1(x)))

        x = self.pool(self.conv1(x))

        x = self.pool(self.conv2(x))

        

        # inception layers

        for cfg in self.config:

            inception_v1 = self.inception_block(cfg)

            x = inception_v1(x)

        

        # after inception

        x = self.drop(x)

        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)

        x = self.softmax(x)

        

        return x
inception_layers_configs = [

    # inception 3ab ---------------------------------------------------------------------------------------------------------------------

    {'in_ch'   : 192, 'out_1x1' :  64, 'red_3x3' :  96, 'out_3x3': 128, 'red_5x5' :  16, 'out_5x5':  32, 'out_pool':  32},

    {'in_ch'   : 256, 'out_1x1' : 128, 'red_3x3' : 128, 'out_3x3': 192, 'red_5x5' :  32, 'out_5x5':  96, 'out_pool':  64, 'pool': 'max'},

    # inception 4abcde ------------------------------------------------------------------------------------------------------------------

    {'in_ch'   : 480, 'out_1x1' : 192, 'red_3x3' :  96, 'out_3x3': 208, 'red_5x5' :  16, 'out_5x5':  48, 'out_pool':  64},

    {'in_ch'   : 512, 'out_1x1' : 160, 'red_3x3' : 112, 'out_3x3': 224, 'red_5x5' :  24, 'out_5x5':  64, 'out_pool':  64},

    {'in_ch'   : 512, 'out_1x1' : 128, 'red_3x3' : 128, 'out_3x3': 256, 'red_5x5' :  24, 'out_5x5':  64, 'out_pool':  64},

    {'in_ch'   : 512, 'out_1x1' : 112, 'red_3x3' : 144, 'out_3x3': 288, 'red_5x5' :  32, 'out_5x5':  64, 'out_pool':  64},

    {'in_ch'   : 528, 'out_1x1' : 256, 'red_3x3' : 160, 'out_3x3': 320, 'red_5x5' :  32, 'out_5x5': 128, 'out_pool': 128, 'pool': 'max'},

    # inception 5ab ---------------------------------------------------------------------------------------------------------------------

    {'in_ch'   : 832, 'out_1x1' : 256, 'red_3x3' : 160, 'out_3x3': 320, 'red_5x5' :  32, 'out_5x5': 128, 'out_pool': 128},

    {'in_ch'   : 832, 'out_1x1' : 384, 'red_3x3' : 192, 'out_3x3': 384, 'red_5x5' :  48, 'out_5x5': 128, 'out_pool': 128, 'pool': 'avg'},

    # -----------------------------------------------------------------------------------------------------------------------------------

]
model = GoogLeNet(inception_layers_configs, in_ch=3, out_nodes=1000)



data = torch.rand(64, 3, 224, 224)

model(data).shape # softmax(dim=1)