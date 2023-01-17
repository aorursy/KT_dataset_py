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
import numpy as np

import pandas as pd
# 加载全部语料

reviews =pd.read_excel('../input/job-information-for-it/51job.xlsx')
for i in reviews:

    print(i)
# 打印公司经营范围

for i in set(reviews['company_area']):

    print(i)
# 打印公司类型

for i in set(reviews['company_type']):

    print(i)