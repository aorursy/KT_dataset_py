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
#i got this address from this link: https://www.backblaze.com/b2/hard-drive-test-data.html on the bottom, Download raw data, then right click "download data file" and copy link address. Paste it here

# do the same for other years if you want

!wget https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_2013.zip
!unzip data_2013.zip
!ls 2013
import pandas as pd
data_2013_06_02 = pd.read_csv("2013/2013-06-02.csv")