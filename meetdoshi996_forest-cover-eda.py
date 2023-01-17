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
import seaborn as sns
plt.rcParams.update({'figure.max_open_warning': 0})
data = pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')
data.columns
data.head()
# removing the id column
X = data.drop(columns='Id')
Y1 = X.loc[:, 'Elevation':'Slope' ]
for i,col in enumerate(Y1):
    plt.figure(i, figsize=(12,5))
    sns.boxplot(y=X[col],x=X['Cover_Type'])
Y1 = pd.concat([Y1, X['Cover_Type']], axis=1)
sns.pairplot(Y1, hue ='Cover_Type')
Y2 = X[['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points']]
for i,col in enumerate(Y2):
    plt.figure(i, figsize=(12,5))
    sns.boxplot(y=X[col],x=X['Cover_Type'])
Y2 = pd.concat([Y2, X['Cover_Type']], axis=1)
sns.pairplot(Y2, hue ='Cover_Type')
Y3= X[['Hillshade_9am', 'Hillshade_Noon','Hillshade_3pm']]
for i,col in enumerate(Y3):
    plt.figure(i, figsize=(12,5))
    sns.boxplot(y=X[col],x=X['Cover_Type'])
Y3 = pd.concat([Y3, X['Cover_Type']], axis=1)
sns.pairplot(Y3, hue ='Cover_Type')
Y4 = X[['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3','Wilderness_Area4']]
for i,col in enumerate(Y4):
    plt.figure(i, figsize=(12,5))
    y1 = []
    x1 = []
    for x, y in enumerate(X.groupby(by='Cover_Type')[col].sum()):
        y1.append(y)
        x1.append(x + 1)
    plt.bar(x1,y1 )
    plt.title(col)
Y5 = X[['Soil_Type1', 'Soil_Type2', 'Soil_Type3',
       'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
       'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
       'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
       'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
       'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
       'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
       'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
       'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
       'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40',]]
for i,col in enumerate(Y5):
    plt.figure(i, figsize=(6,4))
    y1 = []
    x1 = []
    for x, y in enumerate(X.groupby(by='Cover_Type')[col].sum()):
        y1.append(y)
        x1.append(x + 1)
    plt.bar(x1,y1)
    plt.title(col)
