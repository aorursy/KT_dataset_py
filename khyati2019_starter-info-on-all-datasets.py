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
# Creating df for all csv files and extracting the info in a text file. The info files will be saved in the local directories once the below code is run.

import pandas as pd

import io

from openpyxl.workbook import Workbook

dataset =0

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print('\n')

        dataset =dataset+1

        print(os.path.join(dirname, filename))

        if(filename.lower().endswith(('csv'))):

            df= pd.read_csv(os.path.join(dirname, filename))

       

        df.info()

        

        buffer = io.StringIO()

        df.info(buf=buffer)

        s = buffer.getvalue()

        with open(str(dataset) + "_" + filename +"_df_info.txt", "w", encoding="utf-8") as f:

            f.write(s)

# Any results you write to the current directory are saved as output.