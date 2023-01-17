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
#loading Dataset

df = pd.read_csv("/kaggle/input/hospitals-and-beds-in-india/Number of Government Hospitals and Beds in Rural and Urban Areas .csv")
df_Govt_hospitals = df.copy()
#datset sample

df_Govt_hospitals.head()
#shape of dataset

df_Govt_hospitals.shape
#rename columns

df_Govt_hospitals = df_Govt_hospitals.rename(columns={"Rural hospitals":"Rural Number of hospitals","Unnamed: 2":"Rural hospitals Number of Beds","Urban hospitals":"Urban Number of hospitals","Unnamed: 4":"Urban hospitals Number of Beds"})
#slicing dataset

df_Govt_hospitals = df_Govt_hospitals[1:-1]
#features of dataset

df_Govt_hospitals.columns
#new column Total Number of Hospitals = Rural Number of hospitals + Urban Number of hospitals

df_Govt_hospitals["Total Number of Hospitals"] = df_Govt_hospitals["Rural Number of hospitals"].astype('int64')+df_Govt_hospitals["Urban Number of hospitals"].astype('int64')
#new column Total hospitals Number of Beds = Rural hospitals Number of Beds + Urban hospitals Number of Beds

df_Govt_hospitals["Total hospitals Number of Beds"] = df_Govt_hospitals["Rural hospitals Number of Beds"].astype('int64')+df_Govt_hospitals["Urban hospitals Number of Beds"].astype('int64')
import matplotlib.pyplot as plt
def plot_figure(X_axis,Y_axis,title):

    plt.figure(figsize=(15,8))

    plt.bar(df_Govt_hospitals[X_axis],df_Govt_hospitals[Y_axis].astype('int64'))

    plt.xticks(rotation=90)

    plt.title(title)
plot_figure("States/UTs","Total Number of Hospitals","Total Number of Hospitals for each state")
plot_figure("States/UTs","Total hospitals Number of Beds","Total hospitals Number of Beds for each state")
plot_figure("States/UTs","Rural Number of hospitals","Rural Number of hospitals for each state")
plot_figure("States/UTs","Rural hospitals Number of Beds","Rural hospitals Number of Beds for each state")
plot_figure("States/UTs","Urban Number of hospitals","Urban Number of hospitals for each state")
plot_figure("States/UTs","Urban hospitals Number of Beds","Urban hospitals Number of Beds for each state")
df_Total_Govt_hospitals = df_Govt_hospitals.sort_values(by='Total Number of Hospitals', ascending=True)
plt.figure(figsize=(15,20))

plt.barh(df_Total_Govt_hospitals["States/UTs"],df_Total_Govt_hospitals["Total Number of Hospitals"].astype('int64'),align='center')

plt.xticks(rotation=90)

plt.title("Total Number of Hospitals Ranks")
df_Govt_hospitals["Rural Number of hospitals"] = df_Govt_hospitals["Rural Number of hospitals"].astype('int64')

df_Rural_Govt_hospitals = df_Govt_hospitals.sort_values(by='Rural Number of hospitals', ascending=True)
plt.figure(figsize=(15,20))

plt.barh(df_Rural_Govt_hospitals["States/UTs"],df_Rural_Govt_hospitals["Rural Number of hospitals"].astype('int64'),align='center')

plt.xticks(rotation=90)

plt.title("Total Number of Hospitals in Rural Area Ranks")
df_Govt_hospitals["Urban Number of hospitals"] = df_Govt_hospitals["Urban Number of hospitals"].astype('int64')

df_Urban_Govt_hospitals = df_Govt_hospitals.sort_values(by='Urban Number of hospitals', ascending=True)
plt.figure(figsize=(15,20))

plt.barh(df_Urban_Govt_hospitals["States/UTs"],df_Urban_Govt_hospitals["Urban Number of hospitals"].astype('int64'),align='center')

plt.xticks(rotation=90)

plt.title("Total Number of Hospitals in Urban Area Ranks")