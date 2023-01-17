# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import urllib.request

import shutil



url = "http://www.igb.uci.edu/~pfbaldi/physics/data/hepjets/images/test_no_pile_5000000.h5"

output_file = "../input/test_no_pile_5000000.h5"

with urllib.request.urlopen(url) as response, open(output_file, 'wb') as out_file:

    shutil.copyfileobj(response, out_file)

# Any results you write to the current directory are saved as output.