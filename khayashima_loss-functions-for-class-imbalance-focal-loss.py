# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
class ImbalancedClassesNumpy:

    def __init__(self,y):

        #y has to be binary vectors

        num_negative = y[y == 0].shape

        num_positive = y[y == 1].shape

        self.alpha = num_negative / (num_negative + num_positive)

        self.gamma = 2

    def _return_pt(self,target,pred):

        return np.add(np.multiply(np.subtract(1, target) , np.subtract(1 , pred)) ,np.multiply( target , pred))

    

    def class_imbalanced_loss(self,target,pred,alpha = None):

        if alpha is None:

            alpha = self.alpha

        pt = self._return_pt(target,pred)

        return - alpha * np.sum(np.log(pt))

    

    def forcal_loss(self,target,pred,alpha = None):

        if alpha is None:

            alpha = self.alpha

        pt = self._return_pt(target,pred)

        return - alpha * np.sum(np.multiply(np.subtract(1 , pt) ** gamma,np.log(pt)))

    
import torch
ten_a = torch.zeros([2,3],dtype=torch.int32)

ten_b = torch.tensor([[1,0,3],[2,0,1]],dtype=torch.int32)

torch.masked_select(ten_b,torch.eq(ten_a,ten_b)).shape[0]

torch.zeros(ten_a.size())
class ImbalancedClassPytorch:

    def __init__(self,y):

        num_negative = torch.masked_select(y,torch.eq(y,torch.zeros(y.size(),dtype = torch.uint8))).shape[0]

        #num_negative = y[y == 0].shape

        num_positive = torch.masked_select(y,torch.eq(y,torch.ones(y.size(),dtype = torch.uint8))).shape[0]

        self.alpha = num_negative / (num_negative + num_positive)

    def _return_pt(self,target,pred):

        pt = torch.add(torch.mul(torch.sub(1, target) , torch.sub(1 , pred)) ,torch.mul( target , pred))

        return pt

    def class_imbalanced_loss(self,target,pred,alpha = None):

        if alpha is None:

            alpha = self.alpha

        pt = self._return_pt(target,pred)

        return - alpha * torch.sum(torch.mul(torch.sub(1 , pt),torch.log(pt)))

    def focal_loss(self,target,pred,alpha = None,gamma = 2):

        if alpha is None:

            alpha = self.alpha

        pt = self._return_pt(target,pred)

        return - alpha * torch.sum(torch.mul(torch.sub(1 , pt).pow(gamma),torch.log(pt)))