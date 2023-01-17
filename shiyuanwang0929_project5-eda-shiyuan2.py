import numpy as np 
import pandas as pd 
from PIL import Image
from PIL import ImageStat
import PIL
import matplotlib.pyplot as plt
import glob
import cv2
import math
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set_style("white")
sns.set(font_scale=1.5)
Train_imageinfo = pd.read_csv('../input/train-imageinfo/Train_imageinfo (4).csv')
# Delete those useless columns since I have already done the subset in previous work.
del Train_imageinfo['Errors']
del Train_imageinfo['Alarm Triggered']
del Train_imageinfo['Dilution Factor']
del Train_imageinfo['Dead Time']
Train_imageinfo.head()
# Overview of the dataframe
Train_imageinfo.describe()
# We can see that the column 'Rel. Humidity' is meaningless since it has all zero values, so we can delete it.
del Train_imageinfo['Rel. Humidity']
Train_imageinfo.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8); # ; avoid having the matplotlib verbose informations
Train_weather = Train_imageinfo[['Temp(C)','Pressure(kPa)','Wind_Speed','Total']]
Train_weather.corr()
sns.heatmap(Train_weather.corr(),cmap=plt.cm.viridis, linecolor='white', annot=True);
gridobj = sns.lmplot(x="Temp(C)", y="Total", data=Train_weather,scatter_kws={'s':1})
plt.title("Scatterplot with line of best fit (Temp VS Total concentration)", fontsize=15)
plt.show()
gridobj = sns.lmplot(x="Pressure(kPa)", y="Total", data=Train_weather,scatter_kws={'s':1})
plt.title("Scatterplot with line of best fit (Pressure VS Total concentration)", fontsize=15)
plt.show()
gridobj = sns.lmplot(x="Wind_Speed", y="Total", data=Train_weather,scatter_kws={'s':1})
plt.title("Scatterplot with line of best fit (Wind_Speed VS Total concentration)", fontsize=15)
plt.show()
def pressure_group(x):
    if 0 < x <= 99.5:
        return '0-99.5'
    elif x > 99.5:
        return '>99.5'
def wind_group(x):
    if 0 <= x <= 6:
        return '0-6'
    elif x > 6:
        return '>6'
Train_weather['WindGroup'] = Train_weather['Wind_Speed'].apply(lambda x: wind_group(x))
Train_weather['PressureGroup'] = Train_weather['Pressure(kPa)'].apply(lambda x: pressure_group(x))
Train_weather.head()
sns.lmplot(x="Temp(C)", y="Total", row="WindGroup", col="PressureGroup",data=Train_weather)
Train_imageinfo_08052020 = Train_imageinfo[Train_imageinfo['Image_day'] == 'video08052020']
Train_imageinfo_08052020.describe()
Train_imageinfo_08052020 = Train_imageinfo_08052020[['Temp(C)','Pressure(kPa)','Contrast_minmax','Contrast_RMS','Red','Green','Blue','Luminance','Entropy','Haze_removed','Transmission','Total']]
Train_imageinfo_08052020.head()
Train_imageinfo_08052020.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8);
Train_imageinfo_08052020.corr()
plt.figure(figsize=(15, 10))
sns.set(font_scale=1.2)
sns.heatmap(Train_imageinfo_08052020.corr('spearman'),cmap=plt.cm.viridis, linecolor='white', annot=True);
# Scatter plot between independent variables and Total concentration
xx = ['Temp(C)','Pressure(kPa)','Contrast_minmax','Contrast_RMS','Red','Green','Blue','Luminance','Entropy','Haze_removed','Transmission']
for i in xx:
    gridobj = sns.lmplot(x=i, y="Total", data=Train_imageinfo_08052020,scatter_kws={'s':1})
    plt.title("Scatterplot with Line of Best Fit "+ i + " VS Total Concentration", fontsize=15)
    plt.show()
# Kernel density estimate (KDE) plot between independent variables and Total concentration
for i in xx:
    gridobj = sns.kdeplot(Train_imageinfo_08052020[i], Train_imageinfo_08052020['Total'])
    plt.title("KDE Plot of "+ i + " and Total Concentration", fontsize=15)
    plt.show()
# Pair scatter plot among all variables
sns.pairplot(Train_imageinfo_08052020[xx], kind = 'scatter')
plt.show()