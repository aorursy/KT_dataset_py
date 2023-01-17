# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# İlk olarak uzerinde işlem yapacagımız data dosyasını seçelim



data = pd.read_csv("/kaggle/input/protein-data-set/pdb_data_no_dups.csv")
# data hakkında genel bilgiler



data.info()
# Satır ve sutun sayıları 



data.shape
# Sutun isimleri



data.columns
data.dtypes
# sutunlarda hangi degerlerden kaç tane oldugunu bulma



def deger_sayisi(variable):

    

    var = data[variable]  # columnun  degişkenlerni atadık 

    varValue = var.value_counts()

    

    print("{} : \n {}" .format(variable,varValue))
category = ["classification" , "experimentalTechnique" , "macromoleculeType" , "crystallizationMethod" , "phValue"]



for i in category:

    deger_sayisi(i)