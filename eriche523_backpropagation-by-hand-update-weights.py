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
from IPython.display import Image

import os
Image("../input/20200105_233950.jpg")
Image("../input/20200105_234013.jpg")
Image("../input/20200105_234028.jpg")
Image("../input/20200105_234036.jpg")
Image("../input/20200105_234044.jpg")
Image("../input/20200105_234105.jpg")
Image("../input/20200105_234122.jpg")
Image("../input/20200105_234149.jpg")