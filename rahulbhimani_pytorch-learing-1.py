# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import torch

import numpy as np
np_data=np.arange(6).reshape((2,3))

torch_data=torch.from_numpy(np_data)

tensor2array=torch_data.numpy()

print(

    '\nnumpy array:',np_data,

    '\ntorch tensor',torch_data,

    '\ntensor to array:',tensor2array

)
data=[-1,-2,1,2]

tensor=torch.FloatTensor(data)

print(

    '\nabs',

    '\nnumpy: ',np.abs(data),

    '\ntorch: ',torch.abs(tensor)

)
print(

    '\nsin',

    '\nnumpy: ', np.sin(data),

    '\ntorch: ', torch.sin(tensor)

)
tensor.sigmoid()
tensor.exp()
print(

    'mean',

    '\nnumpy: ',np.mean(data),

    '\ntorch: ',torch.mean(tensor)

)
data=[[1,2],

     [3,4]]

tensor=torch.FloatTensor(data)

print(

    'multiplication (matmul)',

    '\nnumpy :',np.matmul(data,data),

    '\ntorch :',torch.mm(tensor,tensor)

)