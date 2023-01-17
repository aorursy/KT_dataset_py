# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from pycaret.regression import *
data = pd.read_csv("/kaggle/input/top50spotify2019/top50.csv",encoding="ISO-8859-1")
data.head()
data.tail()
data.columns
data['Genre'].unique()
#Catplot
#It gives count of genre in spotify top 50 list. 
sns.catplot(y = "Genre", kind = "count",
            palette = "colorblind", edgecolor = ".6",
            data = data)
plt.show()
plt.figure(figsize=(12,12))
sns.jointplot(x= data["Beats.Per.Minute"].values, y= data['Popularity'].values, size=10, kind="hex",)
plt.ylabel('Popularity', fontsize=12)
plt.xlabel("Beats.Per.Minute", fontsize=12)
plt.title("Beats.Per.Minute Vs Popularity", fontsize=15);
#The purpose of this graph is to show connection among Beats and Popularity
data.columns
plt.figure(figsize=(12,12))
sns.jointplot(x= data['Loudness..dB..'].values, y= data['Popularity'].values, size=10, kind="kde",)
plt.ylabel('Popularity', fontsize=12)
plt.xlabel("Beats.Per.Minute", fontsize=12)
plt.title("Beats.Per.Minute Vs Popularity", fontsize=15);
#The purpose of this graph is to show connection among loudness and Popularity
plt.figure(figsize=(16, 6))
heatmap=sns.heatmap(data.corr(),vmax=1, vmin=-1,annot=True)
#Pie charts 
labels = data['Artist.Name'].value_counts().index
sizes = data['Artist.Name'].value_counts().values
colors = ['red', 'yellowgreen', 'lightcoral', 'lightskyblue','cyan', 'green', 'black','yellow']
plt.figure(figsize = (10,10))
plt.pie(sizes, labels=labels, colors=colors)
autopct=('%1.1f%%')
plt.axis('equal')
plt.show()
for i in data['Genre']:
    if "pop" in i:
        data['Genre'] = data['Genre'].replace(i,"pop")
    if "edm" in i:
        data['Genre'] = data['Genre'].replace(i,"edm")
    if "hip hop" in i: 
        data['Genre'] = data['Genre'].replace(i,"hip hop")
    if "room" in i:
        data['Genre'] = data['Genre'].replace(i,"other")
    if "r&b"  in i:  
        data['Genre'] = data['Genre'].replace(i,"other")
    if "reggae" in i:  
        data['Genre'] = data['Genre'].replace(i,"reggae")
    if "rap" in i:
        data['Genre'] = data['Genre'].replace(i,"hip hop")
    if "boy band" in i:
        data['Genre'] = data['Genre'].replace(i,"pop")
    if "brostep" in i:
        data['Genre'] =  data['Genre'].replace(i,"other")
data['Genre'].unique()
plt.figure(figsize=(16, 6))
data['Genre'].value_counts().plot(kind="bar", color = 'r')
data.drop(['Track.Name', 'Artist.Name'], axis = 1, inplace = True)
data
reg = setup(data = data, 
             target = 'Popularity',
             numeric_imputation = 'mean',
             categorical_features = ['Genre']  , 
             normalize = True,
             silent = True)
compare_models()
svm = create_model('svm')
plot_model(svm)
