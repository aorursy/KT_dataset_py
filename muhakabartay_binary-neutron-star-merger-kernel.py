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
#!ls /kaggle/input/



DIR='/kaggle/input/binary-neutron-star-merger/'



!echo $DIR
import csv



# read flash.dat to a list of lists

datContent = [i.strip().split() for i in open("/kaggle/input/binary-neutron-star-merger/FABER.stuff/gw3p3.dat").readlines()]