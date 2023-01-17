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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import seaborn as sns

%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
crops_prod_data = pd.read_csv("../input/agricuture-crops-production-in-india/datafile (3).csv")

print(crops_prod_data.info())
crops_prod_data.isnull().sum()
crops_prod_data.dropna(subset=['Season/ duration in days'],inplace= True)
del crops_prod_data['Unnamed: 4']
crops_prod_data.isnull().sum()
plt.figure(figsize=(40,9)) 

ax =sns.countplot(x='Crop', data=crops_prod_data)
