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
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

%matplotlib inline
data_level = pd.read_csv('../input/chennai_reservoir_levels.csv')

data_level.info()
data_level.describe()
data_rainfall = pd.read_csv('../input/chennai_reservoir_rainfall.csv')
data_rainfall.info()
data_rainfall.describe()
data_level.head(3)

data_level['Date'] = pd.to_datetime(data_level['Date'])
data_level.index = pd.to_datetime(data_level['Date'])
data_level.head(3)
del data_level['Date']

data_level.head(3)
data_rainfall['Date'] = pd.to_datetime(data_rainfall['Date'])
data_rainfall.index = pd.to_datetime(data_rainfall['Date'])
data_rainfall.head(3)
del data_rainfall['Date']



data_rainfall.head(3)
data_level.corr()
data_rainfall.corr()
# Correlation matrix

def plotCorrelationMatrix(df, graphWidth, filename):

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] 

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()
plotCorrelationMatrix(data_level, graphWidth=10, filename='reservoir level')
plotCorrelationMatrix(data_rainfall, graphWidth=10, filename='rainfall level')
data_level.POONDI.plot(kind='line', title='POONDI-Water Level',color='r', figsize=(11,5.5), style='--' )
data_level.CHOLAVARAM.plot(kind='line', title='CHOLAVARAM-Water Level',color='g', figsize=(11,5.5), style='--' )
data_level.REDHILLS.plot(kind='line', title='REDHILLS-Water Level',color='m', figsize=(11,5.5), style='--' )
data_level.CHEMBARAMBAKKAM.plot(kind='line', title='CHEMBARAMBAKKAM-Water Level',color='b', figsize=(11,5.5), style='--' )
data_level['TOTAL'] = data_level['POONDI'] + data_level['CHOLAVARAM'] + data_level['REDHILLS'] + data_level['CHEMBARAMBAKKAM']



data_level.head(3)
data_level.TOTAL.plot(kind='line', title='Total - Water Level in Chennai',color='c', figsize=(11,5.5), style='--')
data_rainfall.POONDI.plot(kind='line', title='POONDI - Rainfall',color='R', figsize=(11,5.5), style='--' )
data_rainfall.CHOLAVARAM.plot(kind='line', title='CHOLAVARAM - Rainfall',color='G', figsize=(11,5.5), style='--' )
data_rainfall.REDHILLS.plot(kind='line', title='REDHILLS - Rainfall',color='B', figsize=(11,5.5), style='--' )
data_rainfall.CHEMBARAMBAKKAM.plot(kind='line', title='CHEMBARAMBAKKAM - Rainfall',color='M', figsize=(11,5.5), style='--' )
data_rainfall['TOTAL'] = data_rainfall['POONDI'] + data_rainfall['CHOLAVARAM'] + data_rainfall['REDHILLS'] + data_rainfall['CHEMBARAMBAKKAM']



data_rainfall.head(3)
data_rainfall.TOTAL.plot(kind='line', title='Total - Rainfall in Chennai',color='c', figsize=(11,5.5), style='--')