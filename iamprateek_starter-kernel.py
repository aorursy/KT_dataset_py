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
Hackathon_Working_Data = pd.read_csv("/kaggle/input/store-transaction-data/Hackathon_Working_Data.csv")

Hackathon_Validation_Data = pd.read_csv("/kaggle/input/store-transaction-data/Hackathon_Validation_Data.csv")

Hackathon_Ideal_Data = pd.read_csv("/kaggle/input/store-transaction-data/Hackathon_Ideal_Data.csv")

Hackathon_Mapping_File = pd.read_csv("/kaggle/input/store-transaction-data/Hackathon_Mapping_File.csv")
Hackathon_Working_Data.head()
Hackathon_Validation_Data.head()
Hackathon_Ideal_Data.head()
Hackathon_Mapping_File.head()