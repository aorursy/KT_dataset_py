# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import collections

from datetime import datetime, timedelta

os.environ["XRT_TPU_CONFIG"] = "tpu_worker;0;10.0.0.2:8470"

!export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
!dpkg -i /kaggle/input/torchxla/*.deb
_VersionConfig = collections.namedtuple('_VersionConfig', 'wheels,server')

VERSION = "torch_xla==nightly"

CONFIG = {

    'torch_xla==nightly': _VersionConfig('nightly', 'XRT-dev20200311')

}[VERSION]

DIST_BUCKET = '/kaggle/input/torchxla'

TORCH_WHEEL = 'torch-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)

TORCH_XLA_WHEEL = 'torch_xla-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)

TORCHVISION_WHEEL = 'torchvision-{}-cp36-cp36m-linux_x86_64.whl'.format(CONFIG.wheels)





!pip uninstall -y torch torchvision

!cp "$DIST_BUCKET/$TORCH_WHEEL" .

!cp "$DIST_BUCKET/$TORCH_XLA_WHEEL" .

!cp "$DIST_BUCKET/$TORCHVISION_WHEEL" .

!pip install "$TORCH_WHEEL"

!pip install "$TORCH_XLA_WHEEL"

!pip install "$TORCHVISION_WHEEL"
import torch

import torch_xla

import torch_xla.core.xla_model as xm



t = torch.randn(2, 2, device=xm.xla_device())

print(t.device)

print(t)
x = np.arange(1000000)
xm.save(x, 'x.pt')
y = torch.load('x.pt')
y