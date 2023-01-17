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
def ReadCSV(file_name):
    print("-----------ReadCSV-----------")
    print(file_name)
    df = pd.read_csv(file_name)
    print("-----------INFO-----------")
    print(df.info())
    print("-----------HEAD-----------")
    print(df.head())

    
main_path = "../input"
list_of_files = os.listdir("../input")
# os.path.join(main_path,list_of_files[0])
[ReadCSV(os.path.join(main_path,file)) for file in list_of_files]

