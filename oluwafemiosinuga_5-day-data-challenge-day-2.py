# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import pandas as pd
data_final = pd.read_excel('../input/millenniumofdata_v3_final.xlsx',sheetname='A4. Ind Production 1270-1870',skiprows=7)

data_final
data_final.describe()
_ = plt.hist(data_final['Leather'],bins=20)

_ = plt.xlabel('Leather')

_ = plt.ylabel('Count')

_ = plt.title('Histogram of Leather')

plt.show()