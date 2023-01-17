# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Plots Libraries
import folium
from matplotlib import  pyplot as plt
%matplotlib inline

plt.rcParams['axes.labelcolor'] = 'blue'
plt.rcParams['axes.titlesize'] = 16

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
speed_camera = pd.read_csv("../input/speed-camera-locations.csv",parse_dates=['GO-LIVE DATE'])
speed_cam_violation = pd.read_csv('../input/speed-camera-violations.csv',parse_dates=['VIOLATION DATE'])
red_light_cam_loc = pd.read_csv("../input/red-light-camera-locations.csv",parse_dates=['GO LIVE DATE'])
red_light_violation = pd.read_csv("../input/red-light-camera-violations.csv", parse_dates=['VIOLATION DATE'])


speed_camera.rename(columns={'GO-LIVE DATE':'GO LIVE DATE'}, inplace=True)
red_light_cam_loc.rename(columns={'INTERSECTION':'ADDRESS'},inplace=True)

def camera_location(dataset):
    m = folium.Map(location=[41.859531,-87.693233],zoom_start=10.5,)

    for i in range(0,len(dataset)):
        log, lat,addr = dataset.iloc[i]['LONGITUDE'], dataset.iloc[i]['LATITUDE'],dataset.iloc[i]['ADDRESS']
        folium.Marker([lat,log],popup=addr).add_to(m)
    return m

def get_year_month_day_data(dataset):
    latest_year_violation = dataset[
        dataset['VIOLATION DATE'].dt.year == dataset['VIOLATION DATE'].dt.year.max()] 

    latest_month_violation = latest_year_violation[
        latest_year_violation['VIOLATION DATE'].dt.month == latest_year_violation['VIOLATION DATE'].dt.month.max()]  

    latest_date_violation= latest_month_violation[
        latest_month_violation['VIOLATION DATE'].dt.day == latest_month_violation['VIOLATION DATE'].dt.day.max()] 
    
    return (latest_year_violation,latest_month_violation, latest_date_violation)

def plot_violation(dataset, title=None):
    plt.figure(figsize=(16,4))
    top_10 = dataset.groupby('ADDRESS').sum().sort_values(
        by = 'VIOLATIONS',ascending=False).head(10)['VIOLATIONS'].plot(kind='bar')
    plt.grid()
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("No of Violations");
    
def plot_camera_go_live(dataset, title = None):
    plt.figure(figsize=(16,4))
    dataset['GO LIVE DATE'].dt.year.value_counts().sort_index().plot(kind='bar');
    plt.title(title)
    plt.ylabel('No of Camera')
    plt.grid();
print('No of Speed Camera :{}'.format(speed_camera.shape[0]))
print('No of Red Light Camera :{}'.format(red_light_cam_loc.shape[0]))
camera_location(speed_camera)
plot_camera_go_live(speed_camera,'Speed Camera Go Live By Year')
year, month,date = get_year_month_day_data(speed_cam_violation)
plot_violation(year,'Top 10 Current Year Speed Violation Spots')
plot_violation(month,'Top 10 Current Month Speed Violation Spots')
plot_violation(date,'Top 10 Current Day Speed Violation Spots')
camera_location(red_light_cam_loc)
plot_camera_go_live(red_light_cam_loc)
year, month,date = get_year_month_day_data(red_light_violation)
plot_violation(year,'Top 10 Current Year Red Light Violation Spots')
plot_violation(month,'Top 10 Current Month Red Light Violation Spots')
plot_violation(date,'Top 10 Current Day Red Light Violation Spots')
def violation_by_month(dataset,title=None):
    import calendar
    
    year,_,_= get_year_month_day_data(dataset)
    dataset_18 = dataset
    dataset_18['MONTH'] = dataset_18['VIOLATION DATE'].dt.month.apply(
        lambda x: calendar.month_abbr[x])
    dataset_18[['MONTH','VIOLATIONS']].groupby(['MONTH']).sum().sort_values(
         by='VIOLATIONS',ascending=False).plot(kind='Bar');
    plt.grid();
    plt.title(title)
    
violation_by_month(speed_cam_violation,'Speed Violation By Month')
violation_by_month(red_light_violation, 'Red Light Violation By Month')