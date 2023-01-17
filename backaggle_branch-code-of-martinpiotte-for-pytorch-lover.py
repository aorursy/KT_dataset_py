# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/whale-pytorch-weight-of-branch/"))

# Any results you write to the current directory are saved as output.
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
import pickle
import imp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):

    
    def __init__(self):
        super(KitModel, self).__init__()
#         global __weights_dict
#         __weights_dict = load_weights(weight_file)

        self.conv2d_58 = self.__conv(2, name='conv2d_58', in_channels=1, out_channels=64, kernel_size=(9, 9), stride=(2, 2), groups=1, bias=True)
        self.batch_normalization_55 = self.__batch_normalization(2, 'batch_normalization_55', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_59 = self.__conv(2, name='conv2d_59', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_56 = self.__batch_normalization(2, 'batch_normalization_56', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_60 = self.__conv(2, name='conv2d_60', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_57 = self.__batch_normalization(2, 'batch_normalization_57', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_61 = self.__conv(2, name='conv2d_61', in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_58 = self.__batch_normalization(2, 'batch_normalization_58', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_62 = self.__conv(2, name='conv2d_62', in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_59 = self.__batch_normalization(2, 'batch_normalization_59', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_63 = self.__conv(2, name='conv2d_63', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_60 = self.__batch_normalization(2, 'batch_normalization_60', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_64 = self.__conv(2, name='conv2d_64', in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_61 = self.__batch_normalization(2, 'batch_normalization_61', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_65 = self.__conv(2, name='conv2d_65', in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_62 = self.__batch_normalization(2, 'batch_normalization_62', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_66 = self.__conv(2, name='conv2d_66', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_63 = self.__batch_normalization(2, 'batch_normalization_63', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_67 = self.__conv(2, name='conv2d_67', in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_64 = self.__batch_normalization(2, 'batch_normalization_64', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_68 = self.__conv(2, name='conv2d_68', in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_65 = self.__batch_normalization(2, 'batch_normalization_65', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_69 = self.__conv(2, name='conv2d_69', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_66 = self.__batch_normalization(2, 'batch_normalization_66', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_70 = self.__conv(2, name='conv2d_70', in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_67 = self.__batch_normalization(2, 'batch_normalization_67', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_71 = self.__conv(2, name='conv2d_71', in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_68 = self.__batch_normalization(2, 'batch_normalization_68', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_72 = self.__conv(2, name='conv2d_72', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_69 = self.__batch_normalization(2, 'batch_normalization_69', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_73 = self.__conv(2, name='conv2d_73', in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_70 = self.__batch_normalization(2, 'batch_normalization_70', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_74 = self.__conv(2, name='conv2d_74', in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_71 = self.__batch_normalization(2, 'batch_normalization_71', num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_75 = self.__conv(2, name='conv2d_75', in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_72 = self.__batch_normalization(2, 'batch_normalization_72', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_76 = self.__conv(2, name='conv2d_76', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_73 = self.__batch_normalization(2, 'batch_normalization_73', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_77 = self.__conv(2, name='conv2d_77', in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_74 = self.__batch_normalization(2, 'batch_normalization_74', num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_78 = self.__conv(2, name='conv2d_78', in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_75 = self.__batch_normalization(2, 'batch_normalization_75', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_79 = self.__conv(2, name='conv2d_79', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_76 = self.__batch_normalization(2, 'batch_normalization_76', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_80 = self.__conv(2, name='conv2d_80', in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_77 = self.__batch_normalization(2, 'batch_normalization_77', num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_81 = self.__conv(2, name='conv2d_81', in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_78 = self.__batch_normalization(2, 'batch_normalization_78', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_82 = self.__conv(2, name='conv2d_82', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_79 = self.__batch_normalization(2, 'batch_normalization_79', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_83 = self.__conv(2, name='conv2d_83', in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_80 = self.__batch_normalization(2, 'batch_normalization_80', num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_84 = self.__conv(2, name='conv2d_84', in_channels=256, out_channels=64, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_81 = self.__batch_normalization(2, 'batch_normalization_81', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_85 = self.__conv(2, name='conv2d_85', in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_82 = self.__batch_normalization(2, 'batch_normalization_82', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_86 = self.__conv(2, name='conv2d_86', in_channels=64, out_channels=256, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_83 = self.__batch_normalization(2, 'batch_normalization_83', num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_87 = self.__conv(2, name='conv2d_87', in_channels=256, out_channels=384, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_84 = self.__batch_normalization(2, 'batch_normalization_84', num_features=384, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_88 = self.__conv(2, name='conv2d_88', in_channels=384, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_85 = self.__batch_normalization(2, 'batch_normalization_85', num_features=96, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_89 = self.__conv(2, name='conv2d_89', in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_86 = self.__batch_normalization(2, 'batch_normalization_86', num_features=96, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_90 = self.__conv(2, name='conv2d_90', in_channels=96, out_channels=384, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_87 = self.__batch_normalization(2, 'batch_normalization_87', num_features=384, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_91 = self.__conv(2, name='conv2d_91', in_channels=384, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_88 = self.__batch_normalization(2, 'batch_normalization_88', num_features=96, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_92 = self.__conv(2, name='conv2d_92', in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_89 = self.__batch_normalization(2, 'batch_normalization_89', num_features=96, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_93 = self.__conv(2, name='conv2d_93', in_channels=96, out_channels=384, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_90 = self.__batch_normalization(2, 'batch_normalization_90', num_features=384, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_94 = self.__conv(2, name='conv2d_94', in_channels=384, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_91 = self.__batch_normalization(2, 'batch_normalization_91', num_features=96, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_95 = self.__conv(2, name='conv2d_95', in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_92 = self.__batch_normalization(2, 'batch_normalization_92', num_features=96, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_96 = self.__conv(2, name='conv2d_96', in_channels=96, out_channels=384, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_93 = self.__batch_normalization(2, 'batch_normalization_93', num_features=384, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_97 = self.__conv(2, name='conv2d_97', in_channels=384, out_channels=96, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_94 = self.__batch_normalization(2, 'batch_normalization_94', num_features=96, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_98 = self.__conv(2, name='conv2d_98', in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_95 = self.__batch_normalization(2, 'batch_normalization_95', num_features=96, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_99 = self.__conv(2, name='conv2d_99', in_channels=96, out_channels=384, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_96 = self.__batch_normalization(2, 'batch_normalization_96', num_features=384, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_100 = self.__conv(2, name='conv2d_100', in_channels=384, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_97 = self.__batch_normalization(2, 'batch_normalization_97', num_features=512, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_101 = self.__conv(2, name='conv2d_101', in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_98 = self.__batch_normalization(2, 'batch_normalization_98', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_102 = self.__conv(2, name='conv2d_102', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_99 = self.__batch_normalization(2, 'batch_normalization_99', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_103 = self.__conv(2, name='conv2d_103', in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_100 = self.__batch_normalization(2, 'batch_normalization_100', num_features=512, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_104 = self.__conv(2, name='conv2d_104', in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_101 = self.__batch_normalization(2, 'batch_normalization_101', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_105 = self.__conv(2, name='conv2d_105', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_102 = self.__batch_normalization(2, 'batch_normalization_102', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_106 = self.__conv(2, name='conv2d_106', in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_103 = self.__batch_normalization(2, 'batch_normalization_103', num_features=512, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_107 = self.__conv(2, name='conv2d_107', in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_104 = self.__batch_normalization(2, 'batch_normalization_104', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_108 = self.__conv(2, name='conv2d_108', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_105 = self.__batch_normalization(2, 'batch_normalization_105', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_109 = self.__conv(2, name='conv2d_109', in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_106 = self.__batch_normalization(2, 'batch_normalization_106', num_features=512, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_110 = self.__conv(2, name='conv2d_110', in_channels=512, out_channels=128, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_107 = self.__batch_normalization(2, 'batch_normalization_107', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_111 = self.__conv(2, name='conv2d_111', in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_108 = self.__batch_normalization(2, 'batch_normalization_108', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_112 = self.__conv(2, name='conv2d_112', in_channels=128, out_channels=512, kernel_size=(1, 1), stride=(1, 1), groups=1, bias=True)

    def forward(self, x):
        conv2d_58_pad   = F.pad(x, (3, 4, 3, 4))
        conv2d_58       = self.conv2d_58(conv2d_58_pad)
        conv2d_58_activation = F.relu(conv2d_58)
        max_pooling2d_6 = F.max_pool2d(conv2d_58_activation, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        batch_normalization_55 = self.batch_normalization_55(max_pooling2d_6)
        conv2d_59_pad   = F.pad(batch_normalization_55, (1, 1, 1, 1))
        conv2d_59       = self.conv2d_59(conv2d_59_pad)
        conv2d_59_activation = F.relu(conv2d_59)
        batch_normalization_56 = self.batch_normalization_56(conv2d_59_activation)
        conv2d_60_pad   = F.pad(batch_normalization_56, (1, 1, 1, 1))
        conv2d_60       = self.conv2d_60(conv2d_60_pad)
        conv2d_60_activation = F.relu(conv2d_60)
        max_pooling2d_7 = F.max_pool2d(conv2d_60_activation, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        batch_normalization_57 = self.batch_normalization_57(max_pooling2d_7)
        conv2d_61       = self.conv2d_61(batch_normalization_57)
        conv2d_61_activation = F.relu(conv2d_61)
        batch_normalization_58 = self.batch_normalization_58(conv2d_61_activation)
        conv2d_62       = self.conv2d_62(batch_normalization_58)
        conv2d_62_activation = F.relu(conv2d_62)
        batch_normalization_59 = self.batch_normalization_59(conv2d_62_activation)
        conv2d_63_pad   = F.pad(batch_normalization_59, (1, 1, 1, 1))
        conv2d_63       = self.conv2d_63(conv2d_63_pad)
        conv2d_63_activation = F.relu(conv2d_63)
        batch_normalization_60 = self.batch_normalization_60(conv2d_63_activation)
        conv2d_64       = self.conv2d_64(batch_normalization_60)
        add_17          = batch_normalization_58 + conv2d_64
        activation_17   = F.relu(add_17)
        batch_normalization_61 = self.batch_normalization_61(activation_17)
        conv2d_65       = self.conv2d_65(batch_normalization_61)
        conv2d_65_activation = F.relu(conv2d_65)
        batch_normalization_62 = self.batch_normalization_62(conv2d_65_activation)
        conv2d_66_pad   = F.pad(batch_normalization_62, (1, 1, 1, 1))
        conv2d_66       = self.conv2d_66(conv2d_66_pad)
        conv2d_66_activation = F.relu(conv2d_66)
        batch_normalization_63 = self.batch_normalization_63(conv2d_66_activation)
        conv2d_67       = self.conv2d_67(batch_normalization_63)
        add_18          = batch_normalization_61 + conv2d_67
        activation_18   = F.relu(add_18)
        batch_normalization_64 = self.batch_normalization_64(activation_18)
        conv2d_68       = self.conv2d_68(batch_normalization_64)
        conv2d_68_activation = F.relu(conv2d_68)
        batch_normalization_65 = self.batch_normalization_65(conv2d_68_activation)
        conv2d_69_pad   = F.pad(batch_normalization_65, (1, 1, 1, 1))
        conv2d_69       = self.conv2d_69(conv2d_69_pad)
        conv2d_69_activation = F.relu(conv2d_69)
        batch_normalization_66 = self.batch_normalization_66(conv2d_69_activation)
        conv2d_70       = self.conv2d_70(batch_normalization_66)
        add_19          = batch_normalization_64 + conv2d_70
        activation_19   = F.relu(add_19)
        batch_normalization_67 = self.batch_normalization_67(activation_19)
        conv2d_71       = self.conv2d_71(batch_normalization_67)
        conv2d_71_activation = F.relu(conv2d_71)
        batch_normalization_68 = self.batch_normalization_68(conv2d_71_activation)
        conv2d_72_pad   = F.pad(batch_normalization_68, (1, 1, 1, 1))
        conv2d_72       = self.conv2d_72(conv2d_72_pad)
        conv2d_72_activation = F.relu(conv2d_72)
        batch_normalization_69 = self.batch_normalization_69(conv2d_72_activation)
        conv2d_73       = self.conv2d_73(batch_normalization_69)
        add_20          = batch_normalization_67 + conv2d_73
        activation_20   = F.relu(add_20)
        max_pooling2d_8 = F.max_pool2d(activation_20, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        batch_normalization_70 = self.batch_normalization_70(max_pooling2d_8)
        conv2d_74       = self.conv2d_74(batch_normalization_70)
        conv2d_74_activation = F.relu(conv2d_74)
        batch_normalization_71 = self.batch_normalization_71(conv2d_74_activation)
        conv2d_75       = self.conv2d_75(batch_normalization_71)
        conv2d_75_activation = F.relu(conv2d_75)
        batch_normalization_72 = self.batch_normalization_72(conv2d_75_activation)
        conv2d_76_pad   = F.pad(batch_normalization_72, (1, 1, 1, 1))
        conv2d_76       = self.conv2d_76(conv2d_76_pad)
        conv2d_76_activation = F.relu(conv2d_76)
        batch_normalization_73 = self.batch_normalization_73(conv2d_76_activation)
        conv2d_77       = self.conv2d_77(batch_normalization_73)
        add_21          = batch_normalization_71 + conv2d_77
        activation_21   = F.relu(add_21)
        batch_normalization_74 = self.batch_normalization_74(activation_21)
        conv2d_78       = self.conv2d_78(batch_normalization_74)
        conv2d_78_activation = F.relu(conv2d_78)
        batch_normalization_75 = self.batch_normalization_75(conv2d_78_activation)
        conv2d_79_pad   = F.pad(batch_normalization_75, (1, 1, 1, 1))
        conv2d_79       = self.conv2d_79(conv2d_79_pad)
        conv2d_79_activation = F.relu(conv2d_79)
        batch_normalization_76 = self.batch_normalization_76(conv2d_79_activation)
        conv2d_80       = self.conv2d_80(batch_normalization_76)
        add_22          = batch_normalization_74 + conv2d_80
        activation_22   = F.relu(add_22)
        batch_normalization_77 = self.batch_normalization_77(activation_22)
        conv2d_81       = self.conv2d_81(batch_normalization_77)
        conv2d_81_activation = F.relu(conv2d_81)
        batch_normalization_78 = self.batch_normalization_78(conv2d_81_activation)
        conv2d_82_pad   = F.pad(batch_normalization_78, (1, 1, 1, 1))
        conv2d_82       = self.conv2d_82(conv2d_82_pad)
        conv2d_82_activation = F.relu(conv2d_82)
        batch_normalization_79 = self.batch_normalization_79(conv2d_82_activation)
        conv2d_83       = self.conv2d_83(batch_normalization_79)
        add_23          = batch_normalization_77 + conv2d_83
        activation_23   = F.relu(add_23)
        batch_normalization_80 = self.batch_normalization_80(activation_23)
        conv2d_84       = self.conv2d_84(batch_normalization_80)
        conv2d_84_activation = F.relu(conv2d_84)
        batch_normalization_81 = self.batch_normalization_81(conv2d_84_activation)
        conv2d_85_pad   = F.pad(batch_normalization_81, (1, 1, 1, 1))
        conv2d_85       = self.conv2d_85(conv2d_85_pad)
        conv2d_85_activation = F.relu(conv2d_85)
        batch_normalization_82 = self.batch_normalization_82(conv2d_85_activation)
        conv2d_86       = self.conv2d_86(batch_normalization_82)
        add_24          = batch_normalization_80 + conv2d_86
        activation_24   = F.relu(add_24)
        max_pooling2d_9 = F.max_pool2d(activation_24, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        batch_normalization_83 = self.batch_normalization_83(max_pooling2d_9)
        conv2d_87       = self.conv2d_87(batch_normalization_83)
        conv2d_87_activation = F.relu(conv2d_87)
        batch_normalization_84 = self.batch_normalization_84(conv2d_87_activation)
        conv2d_88       = self.conv2d_88(batch_normalization_84)
        conv2d_88_activation = F.relu(conv2d_88)
        batch_normalization_85 = self.batch_normalization_85(conv2d_88_activation)
        conv2d_89_pad   = F.pad(batch_normalization_85, (1, 1, 1, 1))
        conv2d_89       = self.conv2d_89(conv2d_89_pad)
        conv2d_89_activation = F.relu(conv2d_89)
        batch_normalization_86 = self.batch_normalization_86(conv2d_89_activation)
        conv2d_90       = self.conv2d_90(batch_normalization_86)
        add_25          = batch_normalization_84 + conv2d_90
        activation_25   = F.relu(add_25)
        batch_normalization_87 = self.batch_normalization_87(activation_25)
        conv2d_91       = self.conv2d_91(batch_normalization_87)
        conv2d_91_activation = F.relu(conv2d_91)
        batch_normalization_88 = self.batch_normalization_88(conv2d_91_activation)
        conv2d_92_pad   = F.pad(batch_normalization_88, (1, 1, 1, 1))
        conv2d_92       = self.conv2d_92(conv2d_92_pad)
        conv2d_92_activation = F.relu(conv2d_92)
        batch_normalization_89 = self.batch_normalization_89(conv2d_92_activation)
        conv2d_93       = self.conv2d_93(batch_normalization_89)
        add_26          = batch_normalization_87 + conv2d_93
        activation_26   = F.relu(add_26)
        batch_normalization_90 = self.batch_normalization_90(activation_26)
        conv2d_94       = self.conv2d_94(batch_normalization_90)
        conv2d_94_activation = F.relu(conv2d_94)
        batch_normalization_91 = self.batch_normalization_91(conv2d_94_activation)
        conv2d_95_pad   = F.pad(batch_normalization_91, (1, 1, 1, 1))
        conv2d_95       = self.conv2d_95(conv2d_95_pad)
        conv2d_95_activation = F.relu(conv2d_95)
        batch_normalization_92 = self.batch_normalization_92(conv2d_95_activation)
        conv2d_96       = self.conv2d_96(batch_normalization_92)
        add_27          = batch_normalization_90 + conv2d_96
        activation_27   = F.relu(add_27)
        batch_normalization_93 = self.batch_normalization_93(activation_27)
        conv2d_97       = self.conv2d_97(batch_normalization_93)
        conv2d_97_activation = F.relu(conv2d_97)
        batch_normalization_94 = self.batch_normalization_94(conv2d_97_activation)
        conv2d_98_pad   = F.pad(batch_normalization_94, (1, 1, 1, 1))
        conv2d_98       = self.conv2d_98(conv2d_98_pad)
        conv2d_98_activation = F.relu(conv2d_98)
        batch_normalization_95 = self.batch_normalization_95(conv2d_98_activation)
        conv2d_99       = self.conv2d_99(batch_normalization_95)
        add_28          = batch_normalization_93 + conv2d_99
        activation_28   = F.relu(add_28)
        max_pooling2d_10 = F.max_pool2d(activation_28, kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=False)
        batch_normalization_96 = self.batch_normalization_96(max_pooling2d_10)
        conv2d_100      = self.conv2d_100(batch_normalization_96)
        conv2d_100_activation = F.relu(conv2d_100)
        batch_normalization_97 = self.batch_normalization_97(conv2d_100_activation)
        conv2d_101      = self.conv2d_101(batch_normalization_97)
        conv2d_101_activation = F.relu(conv2d_101)
        batch_normalization_98 = self.batch_normalization_98(conv2d_101_activation)
        conv2d_102_pad  = F.pad(batch_normalization_98, (1, 1, 1, 1))
        conv2d_102      = self.conv2d_102(conv2d_102_pad)
        conv2d_102_activation = F.relu(conv2d_102)
        batch_normalization_99 = self.batch_normalization_99(conv2d_102_activation)
        conv2d_103      = self.conv2d_103(batch_normalization_99)
        add_29          = batch_normalization_97 + conv2d_103
        activation_29   = F.relu(add_29)
        batch_normalization_100 = self.batch_normalization_100(activation_29)
        conv2d_104      = self.conv2d_104(batch_normalization_100)
        conv2d_104_activation = F.relu(conv2d_104)
        batch_normalization_101 = self.batch_normalization_101(conv2d_104_activation)
        conv2d_105_pad  = F.pad(batch_normalization_101, (1, 1, 1, 1))
        conv2d_105      = self.conv2d_105(conv2d_105_pad)
        conv2d_105_activation = F.relu(conv2d_105)
        batch_normalization_102 = self.batch_normalization_102(conv2d_105_activation)
        conv2d_106      = self.conv2d_106(batch_normalization_102)
        add_30          = batch_normalization_100 + conv2d_106
        activation_30   = F.relu(add_30)
        batch_normalization_103 = self.batch_normalization_103(activation_30)
        conv2d_107      = self.conv2d_107(batch_normalization_103)
        conv2d_107_activation = F.relu(conv2d_107)
        batch_normalization_104 = self.batch_normalization_104(conv2d_107_activation)
        conv2d_108_pad  = F.pad(batch_normalization_104, (1, 1, 1, 1))
        conv2d_108      = self.conv2d_108(conv2d_108_pad)
        conv2d_108_activation = F.relu(conv2d_108)
        batch_normalization_105 = self.batch_normalization_105(conv2d_108_activation)
        conv2d_109      = self.conv2d_109(batch_normalization_105)
        add_31          = batch_normalization_103 + conv2d_109
        activation_31   = F.relu(add_31)
        batch_normalization_106 = self.batch_normalization_106(activation_31)
        conv2d_110      = self.conv2d_110(batch_normalization_106)
        conv2d_110_activation = F.relu(conv2d_110)
        batch_normalization_107 = self.batch_normalization_107(conv2d_110_activation)
        conv2d_111_pad  = F.pad(batch_normalization_107, (1, 1, 1, 1))
        conv2d_111      = self.conv2d_111(conv2d_111_pad)
        conv2d_111_activation = F.relu(conv2d_111)
        batch_normalization_108 = self.batch_normalization_108(conv2d_111_activation)
        conv2d_112      = self.conv2d_112(batch_normalization_108)
        add_32          = batch_normalization_106 + conv2d_112
        activation_32   = F.relu(add_32)
        global_max_pooling2d_2 = F.max_pool2d(input = activation_32, kernel_size = activation_32.size()[2:])
        global_max_pooling2d_2_flatten = global_max_pooling2d_2.view(global_max_pooling2d_2.size(0), -1)
        return global_max_pooling2d_2_flatten


    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

#         layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['weights']))
#         if 'bias' in __weights_dict[name]:
#             layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

#         if 'scale' in __weights_dict[name]:
#             layer.state_dict()['weight'].copy_(torch.from_numpy(__weights_dict[name]['scale']))
#         else:
#             layer.weight.data.fill_(1)

#         if 'bias' in __weights_dict[name]:
#             layer.state_dict()['bias'].copy_(torch.from_numpy(__weights_dict[name]['bias']))
#         else:
#             layer.bias.data.fill_(0)

#         layer.state_dict()['running_mean'].copy_(torch.from_numpy(__weights_dict[name]['mean']))
#         layer.state_dict()['running_var'].copy_(torch.from_numpy(__weights_dict[name]['var']))
        return layer

def compute_head_features(x, y):
    x1 = x * y
    x2 = x + y
    x3 = (x - y).abs_()
    x4 = (x - y) * (x - y)
    x = torch.cat([x1, x2, x3, x4], 1)
    return x


class Siamese(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        
        if pretrained:
            pickle.load = partial(pickle.load, encoding="latin1")
            pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

            MainModel = imp.load_source('MainModel', "../input/whale-pytorch-weight-of-branch/keras_branch.py")
            backbone = torch.load("../input/whale-pytorch-weight-of-branch/keras_branch.pth", map_location=lambda storage, loc: storage, pickle_module=pickle)
        else:
            backbone = KitModel()
        
        self.backbone = backbone
        self.feature_dims = 512

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.feature_dims, out_features=1),
        )

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 4), padding=0, stride=1)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(32, 1), padding=0, stride=1)

    def forward(self, xa, xb):
        # Get features
        feature_a = self.get_features(xa)
        feature_b = self.get_features(xb)

        score = self.get_score(feature_a, feature_b)
        return score, feature_a, feature_b

    def get_features(self, x):
        x = self.backbone(x)
        return x

    def get_score(self, feature_a, feature_b):
        # Make head features
        head_features = compute_head_features(feature_a, feature_b)

        head_features = head_features.view(-1, 1, self.feature_dims, 4)
        head_features = F.relu(self.conv1(head_features))
        head_features = head_features.view(-1, 1, 32, self.feature_dims)
        head_features = F.relu(self.conv2(head_features))
        head_features = head_features.view(-1, self.feature_dims)

        score = self.classifier(head_features)
        score = torch.sigmoid(score)
        return score
my_siamese = Siamese(pretrained=False)
my_siamese = Siamese(pretrained=True)