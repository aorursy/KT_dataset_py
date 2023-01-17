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
import matplotlib.pyplot as plt

%matplotlib inline
wine_data = pd.read_csv('/kaggle/input/wine-quality/winequalityN.csv')
wine_data.head()
wine_data.describe()
wine_data.info()
wine_data.isnull().sum()
#Function Handling all NULL values

def fill_na_mean(col_name):

    for i in col_name:

        mean_val = wine_data[i].mean()

        wine_data[i].fillna(mean_val,inplace=True)

        
lst = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','pH','sulphates']

fill_na_mean(lst)

wine_data.isnull().sum()

        
#Plotting graph with features alcohal and fixed acidity

plt.figure(figsize=(10,7))

plt.scatter(x='alcohol',y='fixed acidity',data=wine_data,marker= 'o',c="g")

plt.xlabel('Alcohol',fontsize=13)

plt.ylabel('fixed acidity',fontsize=13)

plt.title('alcohol and fixed acidity',fontsize=18)

plt.show()