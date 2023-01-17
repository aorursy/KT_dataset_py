# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/countries.csv')

print(df.shape)

df.head()
df[df.Country == 'United States of America']
HDI_deficit=df[np.isfinite(df['HDI']) & np.isfinite(df['Biocapacity Deficit or Reserve'] )]

HDI_deficit=HDI_deficit[['Country','HDI','Biocapacity Deficit or Reserve']]



HDI_deficit['Biocapacity Deficit or Reserve'] = HDI_deficit['Biocapacity Deficit or Reserve']+np.abs(HDI_deficit['Biocapacity Deficit or Reserve'].min())

HDI_deficit['Biocapacity Deficit or Reserve'] = HDI_deficit['Biocapacity Deficit or Reserve']/HDI_deficit['Biocapacity Deficit or Reserve'].max()

HDI_deficit.head()
%no correlation here

plt.scatter(HDI_deficit['HDI'],HDI_deficit['Biocapacity Deficit or Reserve'])

plt.show()
HDI_deficit['Biocapacity Deficit or Reserve'].hist(alpha=0.5, bins=10)