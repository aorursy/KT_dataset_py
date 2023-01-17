import torch

import torch.nn as nn

import matplotlib.pyplot as plt 

import numpy as np
configs = {

    'A': [

        {'in_ch':   3, 'out_ch':  64, 'activation': nn.ReLU, 'maxpool': True  },

        # ---------------------------------------------------------------------

        {'in_ch':  64, 'out_ch': 128, 'activation': nn.ReLU, 'maxpool': True  },

        # ---------------------------------------------------------------------

        {'in_ch': 128, 'out_ch': 256, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch': 256, 'out_ch': 256, 'activation': nn.ReLU, 'maxpool': True  },

        # ---------------------------------------------------------------------

        {'in_ch': 256, 'out_ch': 512, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch': 512, 'out_ch': 512, 'activation': nn.ReLU, 'maxpool': True  },

        # ---------------------------------------------------------------------

        {'in_ch': 512, 'out_ch': 512, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch': 512, 'out_ch': 512, 'activation': nn.ReLU, 'maxpool': True  },

        # ---------------------------------------------------------------------

    ],

    'B': [

        {'in_ch':   3, 'out_ch':  64, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch':  64, 'out_ch':  64, 'activation': nn.ReLU, 'maxpool': True  },

        # ---------------------------------------------------------------------

        {'in_ch':  64, 'out_ch': 128, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch': 128, 'out_ch': 128, 'activation': nn.ReLU, 'maxpool': True  },

        # ---------------------------------------------------------------------

        {'in_ch': 128, 'out_ch': 256, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch': 256, 'out_ch': 256, 'activation': nn.ReLU, 'maxpool': True  },

        # ---------------------------------------------------------------------

        {'in_ch': 256, 'out_ch': 512, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch': 512, 'out_ch': 512, 'activation': nn.ReLU, 'maxpool': True  },

        # ---------------------------------------------------------------------

        {'in_ch': 512, 'out_ch': 512, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch': 512, 'out_ch': 512, 'activation': nn.ReLU, 'maxpool': True  },

        # ---------------------------------------------------------------------

    ],

    'D': [

        {'in_ch':   3, 'out_ch':  64, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch':  64, 'out_ch':  64, 'activation': nn.ReLU, 'maxpool': True  },

        # ---------------------------------------------------------------------

        {'in_ch':  64, 'out_ch': 128, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch': 128, 'out_ch': 128, 'activation': nn.ReLU, 'maxpool': True  },

        # ---------------------------------------------------------------------

        {'in_ch': 128, 'out_ch': 256, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch': 256, 'out_ch': 256, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch': 256, 'out_ch': 256, 'activation': nn.ReLU, 'maxpool': True  },

        # ---------------------------------------------------------------------

        {'in_ch': 256, 'out_ch': 512, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch': 512, 'out_ch': 512, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch': 512, 'out_ch': 512, 'activation': nn.ReLU, 'maxpool': True  },

        # ---------------------------------------------------------------------

        {'in_ch': 512, 'out_ch': 512, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch': 512, 'out_ch': 512, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch': 512, 'out_ch': 512, 'activation': nn.ReLU, 'maxpool': True  },

        # ---------------------------------------------------------------------

    ],

    'E': [

        {'in_ch':   3, 'out_ch':  64, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch':  64, 'out_ch':  64, 'activation': nn.ReLU, 'maxpool': True  },

        # ---------------------------------------------------------------------

        {'in_ch':  64, 'out_ch': 128, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch': 128, 'out_ch': 128, 'activation': nn.ReLU, 'maxpool': True  },

        # ---------------------------------------------------------------------

        {'in_ch': 128, 'out_ch': 256, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch': 256, 'out_ch': 256, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch': 256, 'out_ch': 256, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch': 256, 'out_ch': 256, 'activation': nn.ReLU, 'maxpool': True  },

        # ---------------------------------------------------------------------

        {'in_ch': 256, 'out_ch': 512, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch': 512, 'out_ch': 512, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch': 512, 'out_ch': 512, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch': 512, 'out_ch': 512, 'activation': nn.ReLU, 'maxpool': True  },

        # ---------------------------------------------------------------------

        {'in_ch': 512, 'out_ch': 512, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch': 512, 'out_ch': 512, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch': 512, 'out_ch': 512, 'activation': nn.ReLU, 'maxpool': False },

        {'in_ch': 512, 'out_ch': 512, 'activation': nn.ReLU, 'maxpool': True  },

        # ---------------------------------------------------------------------        

    ],

}
class VggNet(nn.Module):

    def __init__(self, config, out_classes=1000):

        super(VggNet, self).__init__()

        

        # maxpool (halves input w,h)

        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))

        

        # conv series

        self.conv_series = self.get_conv_series(config)

        

        # fc series w/ dropout

        self.fc_series = self.get_fc_series([

             {'in_nodes': 512*7*7, 'out_nodes':         4096, 'activation':     nn.ReLU, 'dropout': 0.5   },

             {'in_nodes':    4096, 'out_nodes':         4096, 'activation':     nn.ReLU, 'dropout': 0.5   },

             {'in_nodes':    4096, 'out_nodes':  out_classes, 'activation':  nn.Softmax, },

        ])

        

    def forward(self, x):

        

        # input   : (num_samples, 3, 224, 224)

        # output  : (num_samples, 512, 7, 7) 

        # As spatial resolution is preserved, only maxpool down samples (by half)

        # i.e for 5 maxpools, (224) / (2^5) =  7

        # 512 comes from depth

        x = self.conv_series(x)

        

        # reshape for lin fcs

        x = x.reshape(num_samples, -1)

        

        # input   : (num_samples, -1) i.e (num_samples, 512*7*7)

        # output  : (num_samples, out_classes)

        x = self.fc_series(x)

        

        return x

    

    

    # ==============    

    # Helpers

    # ==============

    def get_conv_series(self, config):

        """

        Gives out sequential model of convs

        """

        

        # accumultor

        series = []

        

        # common to all

        # s=p=1 for k=3 preserves spatial

        # resolution

        k = (3,3)

        s = (1,1)

        p = (1,1)

        

        for layer_cfg in config:

            

            in_ch, out_ch  = layer_cfg['in_ch'], layer_cfg['out_ch']

            actvn          = layer_cfg['activation']

            is_pool        = layer_cfg['maxpool']

            

            # add conv layer

            series.append(

                nn.Conv2d(

                    in_channels=in_ch, 

                    out_channels=out_ch, 

                    kernel_size=k, 

                    stride=s, 

                    padding=p

                )

            )

            

            # add activation if specified

            if actvn is not False:

                series.append(actvn())

            

            # add pool if specified

            if is_pool is not False:

                series.append(self.pool)

                

        # return seq from accumulator

        return nn.Sequential(*series)

    

    

    

    def get_fc_series(self, config):

        """

        Gives out sequential model of lin units

        """

        # accumulator

        series = []

        

        for layer_cfg in config:

            

            in_nodes   = layer_cfg['in_nodes']

            out_nodes  = layer_cfg['out_nodes']

            activation = layer_cfg['activation']

            dropout    = layer_cfg['dropout'] if 'dropout' in layer_cfg else None

            

            # add lin layer

            series.append(nn.Linear(in_nodes, out_nodes))

            

            # add activation

            if activation: series.append(activation())

                

            # add dropout

            if dropout is not None:

                series.append( nn.Dropout(p=dropout))

                

        # return seq

        return nn.Sequential(*series)
Vgg16 = VggNet(configs['D'])



num_samples = 8

INPUT = torch.randn(num_samples, 3, 224, 224)



print(Vgg16(INPUT).shape)
del Vgg16
Vgg19 = VggNet(configs['E'])



num_samples = 8

INPUT = torch.randn(num_samples, 3, 224, 224)



print(Vgg19(INPUT).shape)
del Vgg19
Vgg13 = VggNet(configs['B'])



num_samples = 8

INPUT = torch.randn(num_samples, 3, 224, 224)



print(Vgg13(INPUT).shape)
del Vgg13
Vgg11 = VggNet(configs['A'])



num_samples = 8

INPUT = torch.randn(num_samples, 3, 224, 224)



print(Vgg11(INPUT).shape)