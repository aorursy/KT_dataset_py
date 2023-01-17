import pandas as pd
import numpy as np
import scipy

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
#IPython.notebook.get_cell(0).class_config.set('cm_config',{'lineWrapping':true})

from datetime import datetime

import warnings
warnings.filterwarnings('ignore')
crimeData = pd.read_csv("../input/train.csv")
crimeData.head(5)
#Converting date from string type to datetime
crimeData["Dates"] = pd.to_datetime(crimeData["Dates"])
#Gettin some early insights for every column except the geo-location and dates.
colnames = crimeData.columns.values
colnames = colnames[1:(len(colnames)-2)]
colnames
print(crimeData[colnames].describe())
#Plotting the number of crimes per day of week 
plt.figure(figsize=(16,10))
sns.set_context("poster",)

ax1 =  plt.subplot2grid((2,1),(0,0))
WeeklyData = pd.DataFrame(crimeData["DayOfWeek"].value_counts())
sns.barplot(x=WeeklyData.index, y="DayOfWeek", data=WeeklyData, palette="Blues", order=['Monday','Tuesday','Wednesday',
                                                                       'Thursday','Friday','Saturday','Sunday'])
sns.axlabel("","")

ax2 =  plt.subplot2grid((2,1),(1,0))
sns.countplot(x="DayOfWeek", hue="PdDistrict", data=crimeData, palette="hls", order=['Monday','Tuesday','Wednesday',
                                                                       'Thursday','Friday','Saturday','Sunday']);
sns.axlabel("","")
ax2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,fontsize=12)
#WeeklyDummy = pd.get_dummies(crimeData["DayOfWeek"])
#crimeData = crimeData.join(WeeklyDummy)
#crimeData.drop("DayOfWeek")
CatData = pd.DataFrame(crimeData["Category"].value_counts())
#CatData[0:10]
fig = plt.figure()
plt.figure(figsize=(20,8))
sns.set_context("poster")

ax1 =  plt.subplot2grid((1,2),(0,0))
ax1.set_title('Top 10 crime Categories',color='green',size='20')
sns.barplot(x=CatData[0:10].index, y="Category", data=CatData[0:10], palette = "viridis")
sns.axlabel("","")
for label in ax1.xaxis.get_ticklabels():
        label.set_rotation(90)

ax2 =  plt.subplot2grid((1,2),(0,1))
ax2.set_title('Bottom 10 crime Categories',color='green',size='20')
sns.axlabel("","")
sns.barplot(x=CatData[len(CatData)-10:len(CatData)].index, y="Category", data=CatData[(len(CatData)-10):len(CatData)])
for label in ax2.xaxis.get_ticklabels():
        label.set_rotation(90)