import torch
print(torch.__version__)
!nvcc --version
!cat /usr/include/x86_64-linux-gnu/cudnn_v*.h | grep CUDNN_MAJOR -A 2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
nn.GRUCell(10,10).cuda()
nn.RNNCell(10,10).cuda()
nn.LSTMCell(10,10).cuda()
nn.GRU(10,10).cuda()
nn.LSTM(10,10).cuda()
nn.RNN(10,10).cuda()

