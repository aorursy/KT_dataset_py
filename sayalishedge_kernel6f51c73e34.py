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
#import required packages for further program







import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#to read our datafile using package pandas



data = pd.read_csv("/kaggle/input/german-credit-data-with-risk/german_credit_data.csv")

print (data.columns)

data.head(10)
#Let`s clean our data 

#Encoding the all categorical data variables and missing values   



data['Sex'] = data['Sex'].map({'male':0,'female':1})

data['Housing'] = data['Housing'].map({'own':0, 'rent':2,  'free':1})

data['Saving accounts'] = data['Saving accounts'].map({'little':0,  'moderate':1,   'quite rich':2,  'rich':3,  'NaN':4})

data['Checking account'] = data['Checking account'].map({'little':0, 'moderate':1, 'rich':2,'NaN':3})

data['Purpose'] = data['Purpose'].map({'car':0, 'furniture/equipment':1, 'radio/TV':2, 'domestic appliances':3,'repairs':4, 'education':5, 'business':6, 'vacation/others':7})

data['Risk'] = data['Risk'].map({'good':0,'bad':1})

data["Saving accounts"].fillna(4,inplace=True)

data["Checking account"].fillna(3,inplace=True)

data.head(10)
#you can save/export  your cleaned file anyweare on your PC  

#enter your path/location with new file name in following command 

#data.to_csv("C:/Users/stat 123/Desktop/data coding in python.csv",index=False)


hmap = data.corr()

plt.subplots(figsize=(10, 9))

sns.heatmap(hmap, vmax=.8,annot=True,cmap="coolwarm", square=True)