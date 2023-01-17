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
!pip install torchxrayvision
import torch
import torchvision
import torchxrayvision as xrv
from tqdm import tqdm
import sys
d_covid19 = xrv.datasets.COVID19_Dataset(views=["PA", "AP", "AP Supine"],
                                         imgpath="../input/covid19-image-data-collection/images",
                                         csvpath="../input/covid19-image-data-collection/metadata.csv")
print(d_covid19)

for i in tqdm(range(len(d_covid19))):
    try:
        a = d_covid19[i]
    except KeyboardInterrupt:
        break;
    except:
        print("Error with {}".format(i) + d_covid19.csv.iloc[i].filename)
        print(sys.exc_info()[1])