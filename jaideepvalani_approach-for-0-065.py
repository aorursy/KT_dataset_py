# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))

    None



# Any results you write to the current directory are saved as output.
def _sigmoid(x):

    

    

    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)

    return y



class focal_loss(nn.Module):

  def __init__(self, gamma=2.0):

        super().__init__()

  def forward(self,pred, gt):

    ''' Modified focal loss. Exactly the same as CornerNet.

        Runs faster and costs a little bit more memory

      Arguments:

        pred (batch x c x h x w)

        gt_regr (batch x c x h x w)

    '''

    pred=_sigmoid(pred)

    pos_inds = gt.eq(1).float()

    pos_inds=pos_inds.unsqueeze(1)

    #print(pos_inds.size())

    neg_inds = gt.lt(1).float().unsqueeze(1)



    neg_weights = torch.pow(1 - gt, 4).unsqueeze(1)



    loss = 0

    #print(neg_weights)

    pos_loss = torch.log(pred+1e-7) * torch.pow(1 - pred, 2) * pos_inds

    neg_loss = torch.log(1 - pred+1e-7) * torch.pow(pred, 2) * neg_weights * neg_inds



    

    #.float().sum()

    pos_loss = pos_loss.view(pred.size(0),-1).sum(-1)

    neg_loss = neg_loss.view(gt.size(0),-1).sum(-1)

    #neg_loss.sum(-1)

    num_pos  = pos_inds.sum()

    if num_pos == 0:

      loss = loss - neg_loss

    else:

      loss = loss - (pos_loss + neg_loss) #/ num_pos

    num_pos  = pos_inds.view(gt.size(0),-1).sum(-1)

    #print('loss',loss.size(),pos_loss.size(),loss.size(),'loss_sum',loss.sum(-1).mean(0),num_pos.size())

    return loss.mean(0)
