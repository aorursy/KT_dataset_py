import pandas as pd
from PIL import Image, ImageStat
import numpy as np
import os
import glob

import cv2 
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor 
from sklearn import metrics
from sklearn.metrics import mean_squared_error
%matplotlib inline

from xgboost import XGBRegressor
from xgboost import plot_tree
from sklearn.model_selection import train_test_split #cannot use train test split as data is time series
from sklearn.metrics import mean_squared_error
from xgboost import plot_importance
import seaborn as sns

from statsmodels.tsa.ar_model import AR
Crude_oil_trend=pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/Crude_oil_trend.csv')
Crude_oil_trend.index=Crude_oil_trend.iloc[:,0]
Crude_oil_trend.drop(columns=['Date'],inplace=True)
Crude_oil_trend.tail()
Train_data=pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/COVID-19_and_Price_dataset.csv')
Train_data.index=Train_data.iloc[:,0]
Train_data.drop(columns=['Date'],inplace=True)
Train_data.tail()
fig, axs = plt.subplots(1,2)
axs[0].set_title('Price')
axs[0].plot('Price', data=Train_data, marker='', color='skyblue', linewidth=2)
axs[1].set_title('World Total Cases')
axs[1].plot('World_total_cases', data=Train_data, marker='', color='olive', linewidth=2)
plt.figure(figsize=(40,40))
plt1 = sns.heatmap(Train_data.corr())
plt1.plot()
a=glob.glob(r"../input/ntt-data-global-ai-challenge-06-2020\NTL-dataset\tif\*.tif")
brightness_array=[]
for i in a:
    image = Image.open(i).convert('L')
    stat = ImageStat.Stat(image)
    brightness_array.append(stat.mean[0])
brightness_array=pd.Series(brightness_array)
brightness_array.to_excel('brightness_array.xls')