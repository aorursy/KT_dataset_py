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


dic = {'user1':{'jan':[10,5,5],'feb':[10,5],'mar':[15,20],'apr':[15,20],'may':[15,20],'jun':[15,20],'jul':[15,20],'aug':[15,20],'sep':[15,20],'oct':[15,20],'nov':[15,20],'dec':[15,20]}}

dic['user1']['jan']