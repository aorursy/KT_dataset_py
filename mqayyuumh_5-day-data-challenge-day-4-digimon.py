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
df = pd.read_csv(r'../input/DigiDB_digimonlist.csv')

df.info()
df.describe(include=['O'])
df.describe()
import seaborn as sns

import matplotlib.pyplot as plt



f, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(20,5))

sns.countplot(df['Type'], ax = ax1).set_title('Digimon Type')

sns.countplot(df['Stage'], ax = ax2).set_title('Digimon Stage')

sns.countplot(df['Attribute'], ax = ax3).set_title('Digimon Attribute')