import torch

import torch.nn as nn

import matplotlib.pyplot as plt 

import numpy as np
class AlexNet(nn.Module):

    """

    Input dims: (num_samples, 3, 227, 227)

    """

    def __init__(self):

        super(AlexNet, self).__init__()

        

        # Activation unit(s) and pooling unit

        self.softmax = nn.Softmax()

        self.relu = nn.ReLU()

        self.overlapped_max_pool = nn.MaxPool2d(kernel_size = (3,3), stride = (2,2))

        

        # conv w/ z=11, s=4 i.e 

        # (1,3,227,227) >> (1,96,55,55) >> pool

        self.conv11 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11,11), stride=(4,4), padding=(0,0))



        # conv w/ z=5, pad=2 ,s=1

        # (1,96,27,27) >> (1,256,27,27) >> pool

        self.conv5 = nn.Conv2d(in_channels = 96, out_channels = 256, kernel_size = (5,5), stride = (1,1), padding = (2,2))

        

        # series of 3 convs (note: spatial resolution preserved. s=1, p=1 for z=3)

        # (1, 256, 13, 13) >> 

        # (1, 384, 13, 13) >> (1, 384, 13, 13) >> (1, 256, 13, 13) >> pool

        self.conv3_series = self.get_conv3_series([

            {'in_ch': 256, 'out_ch': 384, 'act': nn.ReLU},

            {'in_ch': 384, 'out_ch': 384, 'act': nn.ReLU},

            {'in_ch': 384, 'out_ch': 256, 'act': nn.ReLU},

        ])

        

        # fc linear units

        # (1, 256, 6, 6) >> reshape(1, -1) >> (1, 9262) >>

        # (in:9216, out: 4096) >> (in: 4096, out: 1000) >> softmax

        self.fc_lin_series = self.get_fc_lin_series([

            {'in_nodes': 256*6*6, 'out_nodes': 4096, 'dropout': 0.5, 'act': nn.ReLU},

            {'in_nodes':    4096, 'out_nodes': 4096, 'dropout':   0, 'act': nn.ReLU},

            {'in_nodes': 4096   , 'out_nodes': 1000, 'dropout': 0.5, 'act': nn.Softmax},

        ])

        

    def forward(self, x):

        

        # conv, max

        x = self.relu(self.conv11(x))

        x = self.overlapped_max_pool(x)

        

        # conv, max

        x = self.relu(self.conv5(x))

        x = self.overlapped_max_pool(x)

        

        # conv, conv, conv, max

        x = self.conv3_series(x)

        x = self.overlapped_max_pool(x)

        

        # reshape to (num_samples, -1)

        x = x.reshape(x.shape[0], -1)

        

        # fc1, drop >> fc2, drop >> softmax

        x = self.fc_lin_series(x)

        

        return x

        

    

    # ====================================

    # helpers

    # ====================================

    def get_conv3_series(self, configs):

        conv3_series = []

        

        for config in configs:

            in_ch, out_ch   = config['in_ch'], config['out_ch']

            activation_unit = config['act']

            

            conv3_series.append(

                nn.Conv2d(

                    in_channels   = in_ch, 

                    out_channels  = out_ch, 

                    kernel_size   = (3,3), 

                    stride        = (1,1), 

                    padding       = (1,1)

                )

            )

            conv3_series.append( activation_unit() )

            

        return nn.Sequential(*conv3_series)

    

    

    def get_fc_lin_series(self, configs):

        lin_series = []

        

        for config in configs:

            in_nodes, out_nodes, dropout_val = config['in_nodes'], config['out_nodes'], config['dropout']

            activation_unit = config['act']



            lin_series.append( nn.Linear(in_nodes, out_nodes) )

            if activation_unit == nn.ReLU:

                lin_series.append( activation_unit() )

            elif activation_unit == nn.ReLU:

                lin_series.append( activation_unit(dim=out_nodes) )

            lin_series.append( nn.Dropout(p=dropout_val) )

            

        return nn.Sequential(*lin_series)
NUM_SAMPLES = 32

INPUT       = torch.randn(NUM_SAMPLES, 3, 227, 227)
model = AlexNet()

print(model(INPUT).shape)